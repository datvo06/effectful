"""Static tool governance: ``toolsof`` / ``reachable_tools`` / ``check_tools`` — no LLM call.

These check the static tool graph: which tools a template, or a function that calls one,
can reach — computed by reading captured lexical scope and by reifying to a Term, never
running the LLM.
"""

import contextlib
from typing import Annotated

from effectful.handlers.llm.completions import (
    LexicalReaders,
    LiteLLMProvider,
    completion,
)
from effectful.handlers.llm.governance import (
    RestrictTools,
    check_tools,
    reachable_tools,
    toolsof,
)
from effectful.handlers.llm.template import Template, Tool
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, handler
from effectful.ops.syntax import ObjectInterpretation, Uses, implements


class _StopBeforeRequest(Exception):
    pass


def _model_tool_names(template, *extra_handlers):
    """The tool *names* actually offered to the model when ``template`` is called under
    ``RestrictTools`` (+ ``extra_handlers``), captured at the true boundary: the ``tools``
    handed to :func:`~effectful.handlers.llm.completions.completion`. We intercept
    ``completion`` and raise before any request, so no LLM is called — and, unlike a capture
    at the outer ``call_assistant`` seam, this observes the *final* tool set after every
    downstream handler (``LexicalReaders`` etc.) has unioned its capabilities in.
    """
    names: set[str] = set()

    class _CaptureRequest(ObjectInterpretation):
        @implements(completion)
        def _c(self, *args, **kwargs):
            names.update(t["function"]["name"] for t in kwargs.get("tools", []))
            raise _StopBeforeRequest

    with contextlib.ExitStack() as stack:
        stack.enter_context(handler(LiteLLMProvider(model="gpt-4o")))
        stack.enter_context(handler(RestrictTools()))
        for h in extra_handlers:
            stack.enter_context(handler(h))
        stack.enter_context(handler(_CaptureRequest()))
        with contextlib.suppress(_StopBeforeRequest):
            template()
    return names


def _trip_planner():
    """Build a template with a captured tool graph and a caller of it.

    Returns ``(suggest_city, delete_everything, my_fn)`` where ``suggest_city`` is a
    template that lexically captures ``cities``/``weather``/``delete_everything``, and
    ``my_fn`` is a plain function that calls the template.
    """

    @Tool.define
    def cities() -> list[str]:
        """Return a list of cities."""
        return ["Chicago", "Barcelona"]

    @Tool.define
    def weather(city: str) -> str:
        """Return the weather in a city."""
        return "sunny"

    @Tool.define
    def delete_everything() -> None:
        """Dangerous: wipe all state."""
        raise RuntimeError("boom")

    @Template.define
    def suggest_city() -> str:
        """Use the `cities` and `weather` tools to suggest a city."""
        raise NotImplementedError

    def my_fn() -> str:
        return suggest_city()

    return suggest_city, delete_everything, my_fn


def test_toolsof_is_the_static_tool_graph():
    suggest_city, delete_everything, _ = _trip_planner()
    reached = toolsof(suggest_city)
    # every lexically-captured tool is reachable, including the dangerous one
    assert delete_everything in reached
    # the root itself is not one of the tools it reaches
    assert suggest_city not in reached


def test_reachable_tools_sees_through_a_template_without_calling_the_llm():
    suggest_city, delete_everything, my_fn = _trip_planner()
    reached = reachable_tools(my_fn)
    # the template it calls, and (transitively) that template's captured tools
    assert suggest_city in reached
    assert delete_everything in reached
    assert toolsof(suggest_city) <= reached


def test_reachable_tools_is_the_leak_check():
    # `reachable_tools(fn) <= declared` is the static tool-safety guarantee.
    suggest_city, delete_everything, my_fn = _trip_planner()
    declared = {suggest_city} | (toolsof(suggest_city) - {delete_everything})
    leak = reachable_tools(my_fn) - declared
    assert leak == frozenset({delete_everything})  # flagged, LLM never called


def test_tool_graph_counts_only_real_tools_not_handler_capabilities():
    # Governance counts only the real Tool/Template instances a template captures in its
    # lexical scope. Handler-injected capabilities (LexicalReaders' synthetic readers, a
    # final tool, a code-exec tool) are `tool_types` assembled at `call_system` time, never
    # in scope, so they never enter the graph. Law: `toolsof` is invariant to whether such a
    # handler is installed, and a plain lexical value never becomes a reachable tool.
    @Tool.define
    def real_tool() -> int:
        """A real tool."""
        return 0

    favorite_city = (
        "Paris"  # a plain lexical value the LLM could read via a synthetic tool
    )

    @Template.define
    def t() -> str:
        """Use {favorite_city} with the real_tool."""
        raise NotImplementedError

    assert (
        t.__context__["favorite_city"] == favorite_city
    )  # the plain value IS in scope...

    baseline = toolsof(t)
    with handler(LexicalReaders()):
        with_readers = toolsof(t)

    assert baseline == frozenset(
        {real_tool}
    )  # ...but is not counted as a reachable tool
    assert (
        with_readers == baseline
    )  # installing LexicalReaders does not change the graph


def test_restrict_tools_withholds_a_disallowed_lexical_tool_at_the_model_boundary():
    # L1: a disallowed real lexical tool is provably never offered to the model, even
    # alongside LexicalReaders (which unions capabilities into the tool set downstream of
    # the filter). Asserted at the true boundary: the tools handed to `completion`.
    @Tool.define
    def cities() -> list[str]:
        """Cities."""
        return ["A"]

    @Tool.define
    def weather(city: str) -> str:
        """Weather."""
        return "sunny"

    @Tool.define
    def delete_everything() -> None:
        """Dangerous."""
        raise RuntimeError

    @Template.define
    def restricted() -> Annotated[str, Uses[cities, weather]]:
        """Use cities and weather to suggest a city."""
        raise NotImplementedError

    # even with LexicalReaders installed (the handler that re-expands scope downstream):
    names = _model_tool_names(restricted, LexicalReaders())
    assert {"cities", "weather"} <= names  # the allow-listed tools are offered
    assert (
        "delete_everything" not in names
    )  # the disallowed real tool is provably withheld


def test_restrict_tools_leaves_unrestricted_templates_alone():
    # Backward-compatible: a template with no `Uses` annotation is unrestricted, so every
    # lexically-captured tool still reaches the model even with RestrictTools installed.
    @Tool.define
    def cities() -> list[str]:
        """Cities."""
        return ["A"]

    @Tool.define
    def delete_everything() -> None:
        """Dangerous."""
        raise RuntimeError

    @Template.define
    def unrestricted() -> str:
        """Suggest a city."""
        raise NotImplementedError

    names = _model_tool_names(unrestricted)
    assert {
        "cities",
        "delete_everything",
    } <= names  # nothing withheld without a Uses row


def test_check_tools_flags_the_leak():
    # check_tools = reachable_tools - allowed; the L2 tool-safety check (no LLM).
    suggest_city, delete_everything, my_fn = _trip_planner()
    allowed = {suggest_city} | (toolsof(suggest_city) - {delete_everything})
    assert check_tools(my_fn, *allowed) == frozenset({delete_everything})
    # allowing everything reachable -> no leak
    assert check_tools(my_fn, *reachable_tools(my_fn)) == frozenset()


def test_reachable_tools_ignores_ambient_apply_handler():
    # Soundness law: static reachability must not depend on what is installed at the call
    # site. With `handler` (merge) an ambient apply interpretation would win dispatch, so a
    # called tool runs concretely instead of reifying and is silently missed. The reifier
    # uses `interpreter` (replace), so the row is identical either way and the reified
    # tool ops never reach the ambient handler.
    suggest_city, delete_everything, my_fn = _trip_planner()
    baseline = reachable_tools(my_fn)

    ran = []

    def ambient(op, *a, **k):  # a valid ambient interpretation (concrete execution)
        ran.append(op)
        return op.__default_rule__(*a, **k)

    with interpreter({apply: ambient}):
        under_ambient = reachable_tools(my_fn)

    assert under_ambient == baseline  # row unaffected by the ambient handler
    assert delete_everything in under_ambient
    # reification isolated the tool calls — none of the trip tools executed concretely
    assert not ({suggest_city, delete_everything} & set(ran))
