"""RELAI learning environment for apartment listing commute coverage."""

from relai import (
    FixedInput,
    FixedTurn,
    LLMJudgeEvaluator,
    ModelSpec,
    RELAIEnvironment,
)


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="commute-reported",
    name="Show Commute Details",
    description="Tests that each returned apartment listing includes approximate distance and estimated commute time to the user's workplace.",
    tags=["end-to-end", "include-distance-and-commute-time-for-each-listing"],
    input=FixedInput(
        turns=[
            FixedTurn(
                content=(
                    "I'm looking for a 2-bedroom apartment in Ballard or Fremont in Seattle. "
                    "My budget is $3,600 all-in max. "
                    "I need at least 2 bedrooms and 1.5 bathrooms. "
                    "I work at 801 5th Ave, Seattle, WA 98104. "
                    "I commute by transit and want to stay under about 45 minutes each way. "
                    "My move-in date is July 1. "
                    "My must-haves are in-unit laundry, parking, and cat-friendly buildings. "
                    "I have one indoor cat. "
                    "My deal-breakers are income-restricted units and basement apartments. "
                    "Please show me a few current options."
                )
            )
        ]
    ),
    mocks={},
    evaluators=[
        LLMJudgeEvaluator(
            id="listing-commute-coverage",
            description="Judges whether every returned listing includes distance and commute details to the workplace.",
            instructions="""
Evaluate only whether the assistant provides the required commute coverage for each apartment listing it returns.

Scenario:
- The user already provides a complete apartment-search profile, including the work address and commute mode.
- The evaluation intent is only whether each returned listing includes both approximate distance and an estimated commute time to the provided work address using the chosen mode.
- Do not evaluate whether those commute estimates are factually correct.

Full-credit behavior:
- The assistant returns one or more apartment listings, ranked options, or clearly separated candidate apartments.
- Every returned listing explicitly includes:
  1. approximate distance from the provided work address, and
  2. estimated commute time by the user's chosen mode, which here is transit.
- Distance and commute details are presented per listing, not only as a general note.
- If one of those details truly cannot be determined, the listing makes that explicit with unknown or unavailable rather than silently omitting it.

Deduct when:
- The assistant returns listings but one or more listings omit distance, omit commute time, or omit both.
- The assistant gives commute details only for some listings.
- The assistant mentions distance or commute only in a general summary instead of per listing.
- The assistant uses the wrong commute mode or fails to tie the estimate to the provided workplace.
- The assistant stays in intake mode or otherwise does not return listings despite the complete profile.

Do not judge:
- Whether the listings, prices, commute times, or distances are factually correct.
- Cost-breakdown quality, ranking quality, or website choice unless they directly prevent evaluating commute-detail coverage.

Feedback requirements:
- Identify the failed criterion or rubric dimension.
- Describe the observed issue that caused the deduction.
- State what full-credit behavior would have required.
- When possible, mention which listing or section omitted approximate distance, estimated commute time, the transit mode, or the tie to the provided work address.
- Keep feedback concise and concrete.
""".strip(),
            model=ModelSpec(name="gpt-5.4"),
        ),
    ],
)
