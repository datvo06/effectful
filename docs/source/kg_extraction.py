"""Provenance-checked knowledge-graph extraction (issue #664).

An LLM that extracts a knowledge graph from a document must stay *grounded in the source*:
every entity and relation in a triple should come from text the model actually located,
not text it invented. This example expresses that requirement as a **refinement type** and
enforces it statically, with no ground-truth labels.

The mechanism is :class:`~effectful.ops.effects.Requires`, the precondition companion to
``Uses``. Where ``Uses[op, ...]`` on a *return* type is a postcondition (the effects an
operation performs), ``Requires(op)`` on an *argument* is a precondition on the value's
**dataflow provenance**: "you may only pass me a value that ``op`` produced." Here
``make_triple``'s span arguments are annotated ``Requires(find_span)``, so a span is
acceptable only if ``find_span`` is in its provenance.

:func:`~effectful.ops.effects.check_requires` reads that provenance off a *term*. So the
check applies to an extraction expressed as an effectful program (for example, the function
an LLM writes under ``SynthesizeAndCall`` — see ``codeadapt.py``): reify the program to a
term and :func:`check_grounded` rejects any triple built from a span the model did not find,
*before* the triple is trusted. In a notebook: write the extraction program, call
:func:`check_grounded` on it, and only run it if the provenance holds.

Note the scope. This grounds a *synthesized program* (a term), where provenance is
structural. It does not, on its own, ground values a model returns through eager JSON
tool-calling, where a returned value arrives with no provenance term attached.
"""

import dataclasses
from collections.abc import Callable
from typing import Annotated

from effectful.internals.runtime import interpreter
from effectful.ops.effects import Requires, check_requires
from effectful.ops.semantics import apply
from effectful.ops.syntax import defdata, defop


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

    Raises ``ValueError`` if the query is not present verbatim, so a span can only
    be produced from text that is actually in the document.
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
    """Build a :class:`Triple`. The subject and object must be *found* spans: each
    carries ``Requires(find_span)``, so :func:`check_grounded` rejects a triple whose
    endpoints were not produced by :func:`find_span`."""
    return Triple(subject, relation, object)


def check_grounded(
    extraction: Callable[[], object],
) -> dict:
    """Provenance violations in an ``extraction`` program: ``{}`` means every triple it
    builds is grounded (each span came from :func:`find_span`), so the extraction can be
    trusted; a non-empty result names the operation, argument, and missing provenance.

    The program is reified to a term under ``defdata`` (no operation body runs, no model is
    called) and folded by :func:`~effectful.ops.effects.check_requires`. This is the
    ``Requires`` analogue of a decode-time validator: run it on the function an LLM
    synthesizes before executing the extraction it describes.
    """
    with interpreter({apply: defdata}):
        term = extraction()
    return check_requires(term)


if __name__ == "__main__":
    document = "Paris is the capital of France."

    def grounded():
        # both endpoints are spans found in the document -> provenance holds
        return make_triple(
            find_span(document, "Paris"),
            "capital_of",
            find_span(document, "France"),
        )

    def hallucinated():
        # the object is a span the model invented, not one it found -> provenance fails
        return make_triple(
            find_span(document, "Paris"),
            "capital_of",
            Span("Atlantis", 0, 8),
        )

    print("grounded:    ", check_grounded(grounded) or "OK — every triple is grounded")
    print("hallucinated:", check_grounded(hallucinated) or "OK")
    # Only run an extraction once its provenance is verified:
    assert not check_grounded(grounded)
    with interpreter({apply: lambda op, *a, **k: op.__default_rule__(*a, **k)}):
        print("result:      ", grounded())
