"""RELAI learning environment for complete intake without re-asking."""

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


def _final_output_text(simulation_result) -> str:
    final_output = getattr(simulation_result, "final_output", "")
    if final_output is None:
        return ""
    return str(final_output)


def _preview(text: str, limit: int = 260) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def check_no_reasking(simulation_result) -> EvaluationResult:
    reply = _final_output_text(simulation_result)
    normalized = " ".join(reply.lower().split())

    reask_patterns = {
        "search area": (
            r"\b(search area|which city|which neighborhood|which neighbourhood|which zip|what area are you looking)\b",
        ),
        "budget": (
            r"\b(monthly budget|max rent|price range|budget range|is that all[- ]in|rent[- ]only)\b",
        ),
        "beds/baths": (
            r"\b(min(?:imum)? beds?|bedrooms? / bathrooms?|how many beds?|how many bathrooms?|beds and baths)\b",
        ),
        "work address": (
            r"\b(work address|where do you commute|where is work|office address|workplace)\b",
        ),
        "commute preference": (
            r"\b(car, transit, bike, or walk|max acceptable commute|how long (?:of )?(?:a )?commute)\b",
        ),
        "move-in date": (
            r"\b(move[- ]in date|when do you need to be in|when do you need to move|when are you looking to move)\b",
        ),
        "must-haves": (
            r"\b(must[- ]haves?|what amenities do you need|required amenities)\b",
        ),
        "deal-breakers": (
            r"\b(deal[- ]breakers?|anything that disqualifies)\b",
        ),
    }
    prompt_markers = (
        "can you share",
        "could you share",
        "please share",
        "please provide",
        "provide",
        "do you have",
        "tell me",
        "let me know",
        "what's",
        "what is",
        "which",
        "when do you",
        "how long",
    )

    triggered_fields: list[str] = []
    for field, patterns in reask_patterns.items():
        if any(re.search(pattern, normalized) for pattern in patterns) and any(
            marker in normalized for marker in prompt_markers
        ):
            triggered_fields.append(field)

    if not triggered_fields:
        return EvaluationResult(
            score=1.0,
            feedback="The reply does not re-ask intake questions that were already provided or clarified as non-blocking.",
        )

    unique_fields = sorted(set(triggered_fields))
    return EvaluationResult(
        score=0.0,
        feedback=(
            "Expected the agent to use the provided intake details and avoid another intake round. "
            f"Observed re-asking for {', '.join(unique_fields)}. Reply excerpt: {_preview(reply)!r}"
        ),
    )


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="no-reasking",
    name="Search Without Re-Asking",
    description="Tests that the agent starts apartment search when the user already provides the core intake details.",
    tags=["end-to-end", "start-search-with-complete-core-intake"],
    input=FixedInput(
        turns=[
            FixedTurn(
                content=(
                    "I'm looking for an apartment in Ballard or Fremont in Seattle. "
                    "My budget is $3,200 all-in max. "
                    "I need at least 2 bedrooms and 2 bathrooms. "
                    "I work at 801 5th Ave in Seattle, commute by transit, and my must-haves are in-unit laundry and parking."
                )
            )
        ]
    ),
    mocks={},
    evaluators=[
        CodeEvaluator(
            id="no-intake-reasking",
            description="Checks that the reply does not ask the user to repeat intake details or other clarified non-blocking intake gaps.",
            evaluate=check_no_reasking,
        ),
        LLMJudgeEvaluator(
            id="proceed-to-search",
            description="Judges whether the assistant uses the provided intake details to move into search mode in the same turn.",
            instructions="""
Evaluate only whether the assistant proceeds appropriately after receiving a nearly complete apartment-search intake message.

Scenario:
- The user already provides search area, budget, beds/baths, work address, commute mode, and must-haves in one message.
- Treat move-in date, deal-breakers, and the exact max commute time as non-blocking gaps for this scenario.
- The assistant should not re-ask the intake questionnaire before searching.

Full-credit behavior:
- The assistant uses the supplied criteria and moves into search/listing mode in the same turn.
- It may present search-informed listings, ranked options, or a concise search-derived summary.
- It does not block on missing move-in date, deal-breakers, or exact max commute time.
- It does not ask the user to restate already supplied intake details.

Deduct when:
- The assistant stays in intake mode or asks another intake questionnaire instead of searching.
- The assistant asks for move-in date, deal-breakers, or exact max commute time before proceeding.
- The assistant merely says it will search later without actually moving into a search-oriented response for this turn.

Do not judge exact listing correctness, exact wording, or the specific websites used.

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
