"""Static tool-graph governance for LLM :class:`~effectful.handlers.llm.template.Template` s.

A :class:`~effectful.handlers.llm.template.Tool` **is** an
:class:`~effectful.ops.types.Operation` (``class Tool(Operation)``,
``class Template(Tool)``), so a template's tools *are* part of its effect row. These
compute the tool graph without ever calling the LLM:

* :func:`toolsof` — the tools transitively reachable from a tool/template through the
  lexical scope it captured, via :func:`~effectful.handlers.llm.completions._tools_in_scope`.
  Fully static: it reads captured lexical contexts, nothing is executed.
* :func:`reachable_tools` — the tools a zero-arg function can reach, *through* templates,
  by reifying it to a :class:`~effectful.ops.types.Term` (never running the LLM or any
  tool body) and walking it for the tools it mentions, then expanding via :func:`toolsof`.
* :func:`check_tools` — the leak check ``reachable_tools(fn) - allowed``.

Only real :class:`Tool` / :class:`Template` instances are in a template's lexical scope;
synthetic lexical readers and other handler capabilities are injected as ``tool_types`` at
``call_system`` time, not captured in scope, so they never appear in this graph.

**Soundness precondition (important).** :func:`reachable_tools` obtains the term by
*running ``fn``'s own Python body* under a reifying interpretation. Operation calls become
term nodes, but **native Python control flow in ``fn`` is resolved at analysis time** — an
untaken ``if``/``for``/``try`` branch contributes no tools. So ``check_tools(fn) == set()``
is a tool-safety guarantee **only for a straight-line ``fn``** (or one whose branching is
expressed with reifying conditional *operations*, which fold both arms). It is not a proof
over arbitrary Python control flow; for a branchy ``fn`` it is the set reached *on this
reification*, which may under-approximate. Keep governed entry points straight-line.
"""

import collections.abc
from collections.abc import Callable
from typing import Any

from effectful.handlers.llm.completions import (
    ToolCallExecutionError,
    _tools_in_scope,
    call_assistant,
    call_tool,
)
from effectful.handlers.llm.encoding import DecodedToolCall
from effectful.handlers.llm.template import FinalTool, Template, Tool
from effectful.internals.runtime import interpreter
from effectful.ops.effects import check_requires
from effectful.ops.semantics import apply, fwd, handler
from effectful.ops.syntax import ObjectInterpretation, Uses, defdata, implements
from effectful.ops.types import Term

__all__ = [
    "toolsof",
    "reachable_tools",
    "check_tools",
    "RestrictTools",
    "CheckProvenance",
]


def _tools_in(term: Any) -> frozenset[Tool]:
    """Every :class:`Tool` appearing *anywhere* in a reified ``term`` — as an operation or
    as an argument. ``Tool`` / ``Template`` subclass :class:`~effectful.ops.types.Operation`
    and define their own ``__apply__``, so a called tool sits in the ``args`` of an
    ``apply`` node rather than being the node's ``op``; a structural walk catches both.
    """
    found: set[Tool] = set()

    def walk(x: Any) -> None:
        if isinstance(x, Tool):
            found.add(x)
        if isinstance(x, Term):
            walk(x.op)
            for a in x.args:
                walk(a)
            for v in x.kwargs.values():
                walk(v)
        elif isinstance(x, collections.abc.Mapping):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple, set, frozenset)):
            for e in x:
                walk(e)

    walk(term)
    return frozenset(found)


def toolsof(tool: Tool) -> frozenset[Tool]:
    """The tools transitively reachable from ``tool`` through the lexical scope it captured
    (a template's tools are themselves tools, so this closes over sub-agents too).

    Fully static — it reads each template's captured ``__context__`` via
    :func:`~effectful.handlers.llm.completions._tools_in_scope`, never calls the LLM. A plain
    :class:`Tool` captures no scope, so it is a leaf. ``tool`` itself is *not* included (it is
    the root, not something it reaches) even though a template sees itself in its own scope.
    """
    seen: set[Tool] = set()
    stack: list[Tool] = [tool]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, Template):
            continue  # only a Template captures lexical scope; a plain Tool is a leaf
        for sub in _tools_in_scope(cur.__context__):
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return frozenset(seen) - {tool}  # exclude the root; a template sees itself in scope


def reachable_tools(fn: Callable[[], Any]) -> frozenset[Tool]:
    """Every tool a zero-arg ``fn`` can reach, *including through templates it calls*,
    without ever running a tool body or the LLM.

    ``fn`` is reified to a :class:`~effectful.ops.types.Term` under ``defdata`` — so even
    tools with real implementations become term nodes rather than executing — and the term
    is walked for the :class:`Tool` s it mentions (:func:`_tools_in`). Those are the
    *directly* reached tools; :func:`toolsof` expands each to the tools it in turn reaches.
    A template's body is ``raise NotHandled`` so it performs no tools directly, but the
    tools it captured lexically are recovered statically through :func:`toolsof`.

    The reifier uses ``interpreter`` (*replace*), never ``handler`` (*merge*): merging
    would let an ambient ``Tool.__apply__`` handler win dispatch, so the tool would run
    concretely instead of reifying and the walk would silently miss it — the static
    guarantee must hold regardless of what is installed at the call site.
    """
    with interpreter({apply: defdata}):
        term = fn()  # reify — no tool body runs, no LLM call, ambient handlers ignored
    direct = _tools_in(term)
    return direct.union(*(toolsof(t) for t in direct))


def check_tools(fn: Callable[[], Any], *allowed: Tool) -> frozenset[Tool]:
    """Tools ``fn`` can reach that are not in the ``allowed`` set — the static tool-safety
    leak check, computed with **no LLM call**. Empty == ``fn`` reaches no tool outside
    ``allowed`` *on this reification* (including through nested templates); this is a
    guarantee only for a straight-line ``fn`` — see :func:`reachable_tools` for the
    precondition (a branchy ``fn`` may under-approximate).

    The tool-graph analogue of :func:`~effectful.ops.effects.check_uses`:
    ``reachable_tools(fn) - allowed``.
    """
    return reachable_tools(fn) - frozenset(allowed)


class RestrictTools(ObjectInterpretation):
    """Off-by-default handler enforcing a template's ``Uses[...]`` allow-list over its
    **lexical tools** at run time.

    When a :class:`~effectful.handlers.llm.template.Template` whose return type declares
    ``Uses[tool, ...]`` is called, the real :class:`Tool` / :class:`Template` instances it
    captured in lexical scope are intersected with the allow-list before being offered to
    the LLM: the ``tools`` set reaching
    :func:`~effectful.handlers.llm.completions.call_assistant` is filtered. A template with
    no ``Uses`` annotation is unrestricted, so installing this handler is backward-compatible.

    **Guarantee (verified).** An unlisted *lexical tool* is provably never offered — it is
    absent from the ``tools`` set at ``call_assistant`` and therefore from the model's
    tool specs and the decode allow-list, so the model cannot invoke it. This holds even
    alongside downstream ``call_assistant`` handlers that union tools back in
    (``LexicalReaders``, ``SynthesizeAndCall``, ``PythonRepl``): those only add
    handler-injected capabilities (synthetic readers, a final tool, a code-exec tool), not
    the filtered-out real tools.

    **Scope (important).** This bounds the *real lexical tools only*, which is the effect
    row ``Uses`` declares. It does **not** bound those handler-injected capabilities — they
    are governed by whether their own handler is installed, not by this allow-list. In
    particular, installing ``LexicalReaders`` alongside this handler still exposes in-scope
    *values* as synthetic readers regardless of ``Uses``; for a hard boundary over lexical
    values, do not also install ``LexicalReaders``. Governing readers/other capabilities by
    the same allow-list is a separate, deferred feature.

    The restriction is scoped to the individual ``Template.__apply__`` by a per-call handler
    on ``call_assistant`` — the effectful-native pattern (a fresh handler whose dynamic
    extent is the call), so a template gets the narrowed set only for its own completion.
    """

    @implements(Template.__apply__)
    def _restrict[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        allowed = Uses.declared(template.__signature__)
        if allowed is None:
            return fwd()  # no allow-list declared -> unrestricted

        def _only_allowed(env, response_type, tools=frozenset(), anchor=None):
            return fwd(
                env,
                response_type,
                frozenset(t for t in tools if t in allowed),
                anchor=anchor,
            )

        with handler({call_assistant: _only_allowed}):
            return fwd()


def _ungrounded_message(
    violations: dict[Any, dict[str, frozenset]],
) -> str:
    """A retry-feedback message naming each ungrounded argument and the operation that must
    have produced it."""
    parts = []
    for op, unmet in violations.items():
        for arg, missing in unmet.items():
            need = " / ".join(sorted(o.__name__ for o in missing))
            parts.append(
                f"argument {arg!r} of {op.__name__}() must be a value produced by "
                f"{need}(), not one constructed directly"
            )
    return (
        "Ungrounded value(s) in your answer: "
        + "; ".join(parts)
        + ". Rebuild those values by calling the required operation(s)."
    )


class CheckProvenance(ObjectInterpretation):
    """Off-by-default handler that provenance-checks a *synthesized* final answer before it
    is accepted, enforcing ``Requires`` refinement types on the code an LLM writes.

    Compose it with ``SynthesizeAndCall`` (and ``RetryLLMHandler``): when the model answers
    by calling the finalizing tool with a function it wrote, this handler reifies that
    function to a term — running its operations symbolically under ``defdata`` rather than
    executing them — and folds it with :func:`~effectful.ops.effects.check_requires`. If the
    function would build a value that violates a ``Requires`` precondition (for example a
    knowledge-graph triple from a span the model *invented* rather than *found*), the
    finalizing call is rejected with a :class:`ToolCallExecutionError` describing the
    ungrounded argument. ``RetryLLMHandler`` feeds that back and the model revises, so the
    answer only stands once its provenance holds — with no ground-truth labels and no
    trusting the model's own claim.

    This is sound because the check reads a *term* (the synthesized program), not the eager
    values the model reports. It assumes the synthesized function is straight-line over its
    operations (it passes found values to operations rather than inspecting them) — the same
    reifiability precondition as :func:`reachable_tools`.
    """

    @implements(call_tool)
    def _check_provenance[T](self, tool_call: DecodedToolCall[T]):
        if isinstance(tool_call.tool, FinalTool):
            # Reify the finalizing application (implementation applied to the inputs): its
            # operations become term nodes instead of executing, so provenance is preserved.
            with interpreter({apply: defdata}):
                term = tool_call.tool.__default__(
                    *tool_call.bound_args.args, **tool_call.bound_args.kwargs
                )
            if violations := check_requires(term):
                raise ToolCallExecutionError(
                    raw_tool_call=tool_call,
                    original_error=ValueError(_ungrounded_message(violations)),
                )
        return fwd()
