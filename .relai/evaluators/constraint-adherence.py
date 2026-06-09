
from relai import LLMJudgeEvaluator, ModelSpec

evaluator = LLMJudgeEvaluator(
    id="constraint-adherence",
    scope="end-to-end",
    description="Judges whether the agent's recommendations respect the user's budget, must-haves, and deal-breakers.",
    instructions="""Evaluate whether the agent respects the user's stated budget, must-haves, and deal-breakers when making recommendations.

Judge the full interaction and final recommendation set using these criteria:
- Budget adherence: Recommended listings should fit the user's stated budget or clearly disclosed acceptable range. Penalize listings that exceed the budget without explicit user approval.
- Must-have adherence: Recommended listings should satisfy the user's explicit required criteria. Penalize any recommendation that misses a stated must-have unless the agent clearly explains that no compliant option exists.
- Deal-breaker exclusion: Any listing that violates an explicit deal-breaker should be treated as a serious failure and should not be included in the recommendations.
- Constraint handling quality: Reward the agent for accurately filtering options, clearly explaining tradeoffs, and calling out uncertainty when a listing's compliance is unclear.

Scoring guidance:
- 1.0: All recommended listings respect the user's budget, must-haves, and deal-breakers, with clear reasoning where helpful.
- 0.5: Mostly compliant, but includes minor misses, weak filtering, or unclear handling of one constraint.
- 0.0: Recommends one or more listings that clearly violate the user's budget, miss must-haves, or break stated deal-breakers.

Base your judgment on the user's explicit constraints in the interaction. Do not invent new preferences or rely on learning-environment-specific assumptions.""",
    model=ModelSpec(name="gpt-5.4"),
)
