"""RELAI learning environment for apartment listing cost-breakdown coverage."""

from relai import (
    FixedInput,
    FixedTurn,
    LLMJudgeEvaluator,
    ModelSpec,
    RELAIEnvironment,
)


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="cost-breakdown",
    name="Show Cost Breakdown",
    description="Tests that each returned apartment listing includes a total monthly cost broken down into key cost components.",
    tags=["end-to-end", "include-total-cost-breakdown-for-each-listing"],
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
                    "My deal-breakers are income-restricted units and basement apartments."
                )
            )
        ]
    ),
    mocks={},
    evaluators=[
        LLMJudgeEvaluator(
            id="listing-cost-breakdown",
            description="Judges whether every returned listing includes a useful monthly cost breakdown.",
            instructions="""
Evaluate only whether the assistant provides the required total monthly cost breakdown for each apartment listing it returns.

Scenario:
- The user already provides a complete apartment-search profile, so the assistant should search and return listings in the same turn.
- The evaluation intent is per-listing cost-breakdown coverage, not broader listing quality or price accuracy.
- A listing may use zero, none, not applicable, or unknown for a cost component when that is appropriate or unavailable, but the component still needs to be explicitly addressed.

Full-credit behavior:
- The assistant returns one or more apartment listings, ranked options, or clearly separated candidate apartments.
- Every returned listing explicitly reports total monthly cost with a breakdown that covers:
  1. rent,
  2. estimated utilities,
  3. parking, and
  4. pet rent.
- Estimated amounts are clearly labeled as estimates or approximations when they are not directly known.
- If a component is not applicable or unavailable, the listing still makes that explicit rather than silently omitting the component.

Deduct when:
- The assistant returns listings but one or more listings omit any of the required cost components.
- The assistant gives only a single total price without the component breakdown.
- The assistant mentions utilities, parking, or pet rent only in a general note rather than per listing.
- The assistant fails to make clear when a non-rent amount is estimated, approximate, unknown, or not applicable.
- The assistant stays in intake mode or otherwise does not return listings despite the complete profile.

Do not judge:
- Whether the actual listing prices are correct.
- Commute quality, ranking quality, or website choice unless they directly prevent evaluating the cost-breakdown requirement.

Feedback requirements:
- Identify the failed criterion or rubric dimension.
- Describe the observed issue that caused the deduction.
- State what full-credit behavior would have required.
- When possible, mention which listing or part of the answer omitted rent, estimated utilities, parking, pet rent, or estimate labeling.
- Keep feedback concise and concrete.
""".strip(),
            model=ModelSpec(name="gpt-5.4"),
        ),
    ],
)
