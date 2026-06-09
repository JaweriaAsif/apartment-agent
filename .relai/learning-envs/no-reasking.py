"""RELAI learning environment for avoiding redundant apartment-search intake."""

import re

from relai import (
    CodeEvaluator,
    EvaluationResult,
    FixedInput,
    FixedTurn,
    LLMJudgeEvaluator,
    ModelSpec,
    RELAIEnvironment,
)


PROVIDED_PROFILE_MESSAGE = (
    "I'm looking for apartments in Ballard or Fremont in Seattle. "
    "My budget is $3,600 all-in max. "
    "I need at least 2 bedrooms and 1.5 bathrooms. "
    "I work at 801 5th Ave, Seattle, WA 98104. "
    "I commute by transit and want to stay under about 45 minutes each way. "
    "My must-haves are in-unit laundry, parking, and cat-friendly buildings."
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


def _find_redundant_intake_prompts(reply: str) -> list[str]:
    normalized = " ".join(reply.lower().split())
    checks = [
        (
            "search area",
            r"\b(which city|which neighborhood|which neighborhoods|which zip code|which zip codes)\b",
        ),
        (
            "monthly budget",
            r"\b(what(?:'s| is) your budget|monthly budget|all[- ]in or rent[- ]only)\b",
        ),
        (
            "bedrooms or bathrooms",
            r"\b(how many bedrooms|how many bathrooms|minimum beds and baths)\b",
        ),
        (
            "work address",
            r"\b(what(?:'s| is) your work address|where do you commute|where is work)\b",
        ),
        (
            "commute preference",
            r"\b(what(?:'s| is) your commute preference|car, transit, bike, or walk|max acceptable commute time)\b",
        ),
        (
            "must-haves",
            r"\b(what are your must-haves|any must-haves)\b",
        ),
    ]

    found: list[str] = []
    for label, pattern in checks:
        if re.search(pattern, normalized):
            found.append(label)
    return found


def check_no_redundant_intake_questions(simulation_result) -> EvaluationResult:
    reply = _final_output_text(simulation_result)
    redundant_prompts = _find_redundant_intake_prompts(reply)

    if not redundant_prompts:
        return EvaluationResult(
            score=1.0,
            feedback="The reply does not re-ask the intake fields the user already supplied.",
        )

    return EvaluationResult(
        score=0.0,
        feedback=(
            "The reply re-asks intake details the user already provided, so it does not move past redundant intake. "
            f"Re-asked fields/signals: {', '.join(redundant_prompts)}. "
            f"Observed reply: {_preview(reply)!r}"
        ),
    )


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="no-reasking",
    name="Skip Redundant Intake",
    description="Tests that the agent proceeds with apartment search instead of re-asking already provided intake details.",
    tags=["end-to-end", "skip-redundant-intake-when-profile-complete"],
    input=FixedInput(
        turns=[
            FixedTurn(content=PROVIDED_PROFILE_MESSAGE),
        ]
    ),
    mocks={},
    evaluators=[
        CodeEvaluator(
            id="no-redundant-intake-questions",
            description="Checks that the reply does not re-ask intake fields the user already supplied.",
            evaluate=check_no_redundant_intake_questions,
        ),
        LLMJudgeEvaluator(
            id="proceed-to-search",
            description="Judges whether the assistant proceeds into search mode when the user already supplied the core search profile.",
            instructions="""
Evaluate only whether the assistant avoids redundant intake and proceeds into apartment-search behavior for this first-turn scenario.

Scenario:
- The user already provides the search area, budget, bedrooms/bathrooms, work address, commute mode, and must-haves in one message.
- Missing optional details should not block the initial search when those core constraints are already present.
- The expected behavior is to proceed with search rather than staying in the intake questionnaire.

Full-credit behavior:
- The assistant clearly moves into search/listing mode in the same turn.
- Acceptable search-mode behavior includes returning listings, ranked options, links, or explicitly reporting that it searched and summarizing the search outcome.
- The assistant may mention optional missing details such as move-in date or deal-breakers, but only as secondary follow-up or assumptions, not as blockers.
- The reply should not re-ask the already supplied core fields as if they were missing.

Deduct when:
- The assistant stays mostly in intake mode, especially as a numbered questionnaire or a request to provide the already supplied criteria before it can search.
- The assistant makes optional follow-up details into prerequisites that block the initial search.
- The assistant gives only generic planning talk without showing that it proceeded to search or attempted the search.

Do not judge:
- Whether the live listings are factually correct or complete.
- Commute math, ranking quality, or listing quality beyond what is necessary to tell whether the assistant actually proceeded to search.

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
