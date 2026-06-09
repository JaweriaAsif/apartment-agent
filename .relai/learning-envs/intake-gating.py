"""RELAI learning environment for apartment-search intake gating."""

from relai import (
    CodeEvaluator,
    EvaluationResult,
    FixedInput,
    FixedTurn,
    LLMJudgeEvaluator,
    ModelSpec,
    RELAIEnvironment,
)


def _final_output_text(simulation_result) -> str:
    final_output = getattr(simulation_result, "final_output", "")
    if final_output is None:
        return ""
    return str(final_output)


def _preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def check_required_intake_fields(simulation_result) -> EvaluationResult:
    reply = _final_output_text(simulation_result)
    normalized = " ".join(reply.lower().split())

    budget_markers = (
        "budget",
        "monthly budget",
        "max rent",
        "rent range",
        "price range",
        "all-in",
        "rent-only",
    )
    work_address_markers = (
        "work address",
        "where do you commute",
        "commute to",
        "office address",
        "workplace",
        "job address",
        "where is work",
    )

    asks_for_budget = any(marker in normalized for marker in budget_markers)
    asks_for_work_address = any(marker in normalized for marker in work_address_markers)

    if asks_for_budget and asks_for_work_address:
        return EvaluationResult(
            score=1.0,
            feedback="The reply asks for both missing search-gating fields: budget and work address.",
        )

    missing_fields: list[str] = []
    if not asks_for_budget:
        missing_fields.append("budget")
    if not asks_for_work_address:
        missing_fields.append("work address")

    score = 0.5 if len(missing_fields) == 1 else 0.0
    return EvaluationResult(
        score=score,
        feedback=(
            "Expected the first intake reply to ask for both budget and work address before searching. "
            f"Missing {', '.join(missing_fields)}. Observed reply: {_preview(reply)!r}"
        ),
    )


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="intake-gating",
    name="Ask Missing Intake Fields",
    description="Tests that the agent asks for budget and work address before searching when the user only gives city and beds.",
    tags=["end-to-end", "ask-budget-work-address-before-search"],
    input=FixedInput(
        turns=[
            FixedTurn(
                content="I'm looking for a 2-bedroom apartment in Seattle."
            )
        ]
    ),
    mocks={},
    evaluators=[
        CodeEvaluator(
            id="ask-budget-and-work-address",
            description="Checks that the reply asks for the missing budget and work address.",
            evaluate=check_required_intake_fields,
        ),
        LLMJudgeEvaluator(
            id="stay-in-intake-mode",
            description="Judges whether the reply stays in intake mode instead of prematurely searching or presenting listings.",
            instructions="""
Evaluate only the assistant's first-turn intake behavior for this apartment-search scenario.

Scenario:
- The user only says they want a 2-bedroom apartment in Seattle.
- The decisive missing search-gating fields are budget and work address.
- The assistant should stay in intake/questionnaire mode until area, budget, and work address are all present.

Full-credit behavior:
- The reply remains an intake follow-up rather than a search result.
- It does not claim to have searched, imply that listings were found, or present apartments, links, addresses, prices, ranked options, or other web-search results.
- Asking additional intake questions is fine.

Deduct when:
- The assistant moves into search/listing mode before collecting the missing search-gating fields.
- The assistant claims or implies that it already searched or found listings from the web.

Feedback requirements:
- Identify the failed criterion or rubric dimension.
- Describe the observed issue that caused the deduction.
- State what full-credit behavior would have required.
- Keep feedback concise and concrete.
""".strip(),
            model=ModelSpec(name="gpt-5.4"),
        ),
    ],
)
