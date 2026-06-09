"""RELAI learning environment for apartment listing commute details."""

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
    name="Include Commute Details",
    description="Tests that each returned apartment listing includes estimated distance and commute time to the user's workplace.",
    tags=["end-to-end", "include-distance-and-commute-for-each-listing"],
    input=FixedInput(
        turns=[
            FixedTurn(
                content=(
                    "I'm looking for a 2-bedroom apartment in Ballard or Fremont in Seattle. "
                    "My budget is $3,500 all-in max. "
                    "I need at least 2 bedrooms and 1.5 bathrooms. "
                    "I work at 801 5th Ave, Seattle, WA 98104. "
                    "I commute by transit and want to stay under about 45 minutes each way. "
                    "My move-in date is July 1. "
                    "My must-haves are in-unit laundry and parking. "
                    "My deal-breakers are income-restricted units and basement apartments."
                )
            )
        ]
    ),
    mocks={},
    evaluators=[
        LLMJudgeEvaluator(
            id="listing-commute-details",
            description="Judges whether every returned listing includes estimated distance and commute time to the user's workplace.",
            instructions="""
Evaluate only whether the assistant's listing output includes commute detail coverage for each returned apartment listing.

Scenario:
- The user already provides a complete apartment-search profile, including a work address and transit as the commute mode.
- The assistant should search and return apartment listings in the same turn.
- The evaluation intent is coverage, not numerical accuracy.

Full-credit behavior:
- The assistant returns one or more apartment listings or ranked options.
- Every returned listing includes both:
  1. an approximate distance to the provided work address, and
  2. an estimated commute time by the user's chosen mode of transit.
- These values are presented as estimates or approximations rather than as certain facts.

Deduct when:
- The assistant returns listings but one or more listings omit distance.
- The assistant returns listings but one or more listings omit commute time.
- The assistant gives only one of the two fields for a listing.
- The assistant mentions commute or distance only in a general summary instead of per listing.
- The assistant presents the values without indicating that they are approximate or estimated.
- The assistant stays in intake mode or otherwise fails to return listings despite the complete profile.

Do not judge:
- Whether the commute or distance numbers are actually correct.
- Broader listing quality, ranking quality, or website/source choice unless they directly prevent evaluating the commute-detail requirement.

Feedback requirements:
- Identify the failed criterion or rubric dimension.
- Describe the observed issue that caused the deduction.
- State what full-credit behavior would have required.
- When possible, mention which listing or portion of the answer lacked distance, commute time, or estimate labeling.
- Keep feedback concise and concrete.
""".strip(),
            model=ModelSpec(name="gpt-5.4"),
        ),
    ],
)
