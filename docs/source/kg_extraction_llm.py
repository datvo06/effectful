"""LLM knowledge-graph extraction with provenance checking (issue #664, code-synthesis).

This is the LLM-in-the-loop companion to ``kg_extraction.py``. The model *writes the
extraction program* -- it answers by synthesizing a function that reads spans from the
document and builds triples -- and that synthesized program is **provenance-checked before
its answer is accepted**: every triple must be built from a span the model actually found
(``find_span``), not one it invented. A hallucinated triple is rejected and the model is
asked to revise, with no ground-truth labels.

The stack, from the effectful LLM handlers plus :class:`~effectful.handlers.llm.governance.CheckProvenance`:

* ``SynthesizeAndCall`` -- the model answers by writing a function (see ``codeadapt.py``);
* ``CheckProvenance`` -- reifies that synthesized function to a term and folds it with
  ``check_requires`` (the ``Requires`` refinement below), rejecting an answer that would
  build a triple from an ungrounded span;
* ``RetryLLMHandler`` -- feeds the rejection back so the model revises.

Because ``CheckProvenance`` raises the standard ``ToolCallExecutionError``, it plugs into the
existing retry path with no changes to the synthesis handler. The check reads the *term*
(the synthesized program), so it is sound; it assumes the synthesized function is
straight-line over its operations (the same reifiability precondition as ``reachable_tools``).

Run it (needs an API key for the chosen model)::

    python -m docs.source.kg_extraction_llm --model anthropic/claude-3-5-sonnet-20241022

The provenance check itself is model-agnostic and is verified without any LLM in
``tests/test_handlers_llm_governance.py::test_check_provenance_rejects_a_synthesized_answer_that_hallucinates``.
"""

import argparse
import contextlib
import dataclasses
from typing import Annotated

import tenacity

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    RetryLLMHandler,
    SynthesizeAndCall,
    TerminalRenderer,
)
from effectful.handlers.llm.evaluation import UnsafeEvalProvider
from effectful.handlers.llm.governance import CheckProvenance
from effectful.ops.effects import Requires
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled


@dataclasses.dataclass(frozen=True)
class Span:
    """A substring located in a document, with its character offsets."""

    text: str
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class Triple:
    """A knowledge-graph edge: ``(subject) --relation--> (object)``."""

    subject: Span
    relation: str
    object: Span


@defop
def find_span(document: str, query: str) -> Span:
    """Locate ``query`` in ``document`` and return the grounded :class:`Span`.

    Raises ``ValueError`` if ``query`` is not present verbatim, so a span can only be
    produced from text that actually occurs in the document.
    """
    start = document.find(query)
    if start < 0:
        raise ValueError(f"{query!r} does not occur in the document")
    return Span(query, start, start + len(query))


@defop
def make_triple(
    subject: Annotated[Span, Requires(find_span)],
    relation: str,
    object: Annotated[Span, Requires(find_span)],
) -> Triple:
    """Build a :class:`Triple`. Both endpoints carry ``Requires(find_span)``, so
    :class:`~effectful.handlers.llm.governance.CheckProvenance` rejects a triple whose
    subject or object was not produced by :func:`find_span`."""
    return Triple(subject, relation, object)


@Template.define
def extract(document: str) -> list[Triple]:
    """Extract a knowledge graph from the document below as a list of triples.

    Use `find_span(document, query)` to locate each entity as a grounded span, and
    `make_triple(subject, relation, object)` to build each edge. Every subject and object
    must be a span you obtained from `find_span` on this document -- do not construct spans
    directly, and do not invent entities that are not present in the text.

    Document:
    {document}
    """
    raise NotHandled


def run(document: str, *, model: str, num_retries: int = 4, render: bool = True):
    """Run the LLM extraction with provenance checking and return the grounded triples.

    ``CheckProvenance`` is installed *inside* ``RetryLLMHandler`` so that a rejected
    (ungrounded) synthesized answer is caught and fed back for revision.
    """
    with (
        handler(LiteLLMProvider(model=model)),
        handler(UnsafeEvalProvider()),
        handler(SynthesizeAndCall()),
        handler(
            CheckProvenance()
        ),  # reject a synthesized answer that hallucinates a span
        handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(num_retries))),
        handler(TerminalRenderer()) if render else contextlib.nullcontext(),
    ):
        return extract(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="anthropic/claude-3-5-sonnet-20241022")
    parser.add_argument(
        "--document",
        default="Marie Curie was born in Warsaw. Paris is the capital of France.",
    )
    parser.add_argument("--num-retries", type=int, default=4)
    args = parser.parse_args()

    triples = run(args.document, model=args.model, num_retries=args.num_retries)
    print(f"\nExtracted {len(triples)} grounded triple(s):")
    for t in triples:
        print(f"  ({t.subject.text}) --{t.relation}--> ({t.object.text})")


if __name__ == "__main__":
    main()
