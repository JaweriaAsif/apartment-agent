
from relai import LLMJudgeEvaluator, ModelSpec

evaluator = LLMJudgeEvaluator(
    id="decision-useful",
    scope="end-to-end",
    description="Judges whether the final answer is concise and helps the user make a decision.",
    instructions=(
        "Judge whether the agent's final answer is concise and decision-useful. "
        "A strong answer ranks the options by fit for the user's needs, clearly labels any "
        "estimates or uncertainty, and ends with a short summary that names the best one or "
        "two picks and briefly explains why. Penalize answers that are verbose, fail to rank "
        "the choices, present estimates as facts, or omit a clear closing recommendation."
    ),
    model=ModelSpec(name="gpt-5.4"),
)
