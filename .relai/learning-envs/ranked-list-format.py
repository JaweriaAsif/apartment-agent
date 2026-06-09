"""RELAI learning environment for ranked apartment result formatting."""

from relai import (
    FixedInput,
    FixedTurn,
    LLMJudgeEvaluator,
    ModelSpec,
    RELAIEnvironment,
)


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="ranked-list-format",
    name="Ranked Apartment Results",
    description="Tests that the agent returns apartment options as a clearly ranked list with the requested fields and a brief best-picks summary.",
    tags=["end-to-end", "rank-results-by-fit-with-fields-and-summary"],
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
                    "Please show me current options as a clearly ranked list by best fit. "
                    "For each listing include the address, rent, total monthly cost, beds/baths, distance from work, estimated commute time, and a link. "
                    "End with a short best-picks summary."
                )
            )
        ]
    ),
    mocks={},
    evaluators=[
        LLMJudgeEvaluator(
            id="ranked-list-format-and-summary",
            description="Judges whether the reply presents apartment results as a clearly ranked list with the required fields and a brief ending summary.",
            instructions="""
Evaluate only whether the assistant presents apartment-search results in the requested ranked-result format.

Scenario:
- The user already provides a complete apartment-search profile, so the assistant should search and return results in the same turn.
- Evaluation intent comes from the scenario request and clarification assumption: semantic compliance is sufficient.
- Do not require exact wording, a specific markup format, or an exact number of listings.

Full-credit behavior:
- The assistant returns one or more apartment listings, candidate options, or clearly separated results rather than staying in intake mode.
- The results are clearly ranked or ordered by fit, such as explicit numbering, best-to-worst ordering, or wording that makes the ranking unambiguous.
- Each returned listing includes all of the following, either with concrete values or an explicit unknown/unavailable marker when the information cannot be confirmed:
  1. address or clear neighborhood/location,
  2. rent,
  3. total monthly cost,
  4. beds/baths,
  5. distance from work,
  6. estimated commute time, and
  7. a link.
- The reply ends with a short best-picks summary that briefly identifies the best one or two options and why they stand out.

Deduct when:
- The assistant does not return listings despite the complete profile.
- The results are not clearly ranked or ordered by fit.
- One or more listings omit any required field without explicitly marking it unknown or unavailable.
- The reply gives rent but not total monthly cost.
- Distance from work or estimated commute time is missing, only mentioned globally, or not tied to each listing.
- Links are missing from one or more listings.
- The answer lacks the brief ending best-picks summary, or the closing section is not actually a concise summary of the top picks.

Do not judge:
- Whether listing facts, prices, commute estimates, or links are accurate.
- Whether the exact formatting is a table, bullets, or paragraphs, as long as the ranking and required fields are clear.
- Broader answer quality beyond what is necessary to assess ranking, field coverage, and the ending summary.

Feedback requirements:
- Identify the failed criterion or rubric dimension.
- Describe the observed issue that caused the deduction.
- State what full-credit behavior would have required.
- When useful, include expected-versus-observed detail such as which required field was missing, whether the ranking was unclear, or whether the ending best-picks summary was absent.
- Keep feedback concise and concrete.
""".strip(),
            model=ModelSpec(name="gpt-5.4"),
        ),
    ],
)
