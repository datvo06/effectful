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

from effectful.handlers.llm.completions import _tools_in_scope, call_assistant
from effectful.handlers.llm.template import Template, Tool
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, fwd, handler
from effectful.ops.syntax import ObjectInterpretation, Uses, defdata, implements
from effectful.ops.types import Term

__all__ = ["toolsof", "reachable_tools", "check_tools", "RestrictTools"]


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
        context = getattr(
            cur, "__context__", None
        )  # only Templates capture lexical scope
        if context is None:
            continue
        for sub in _tools_in_scope(context):
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
    """Off-by-default handler enforcing a template's ``Uses[...]`` allow-list at run time.

    When a :class:`~effectful.handlers.llm.template.Template` whose return type declares
    ``Uses[tool, ...]`` is called, the LLM is offered *only* the declared tools: the
    lexical tools reaching :func:`~effectful.handlers.llm.completions.call_assistant` are
    intersected with the allow-list. A template with no ``Uses`` annotation is unrestricted,
    so installing this handler is backward-compatible.

    Enforcement is by construction and sound (no LLM introspection): an unlisted tool is
    never *offered*, and the decode boundary rejects a call to a tool that was not offered,
    so the model physically cannot invoke it. The restriction is scoped to the individual
    ``Template.__apply__`` by a per-call handler on ``call_assistant`` — the effectful-native
    pattern (a fresh handler whose dynamic extent is the call), so a template gets the
    narrowed set only for its own completion.

    Only real lexical tools are filtered; handler-injected capabilities (synthetic readers,
    a final tool, a code-exec tool) are ``tool_types`` unioned in downstream and are left
    untouched.
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
