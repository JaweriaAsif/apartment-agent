
from relai import LLMJudgeEvaluator, ModelSpec

evaluator = LLMJudgeEvaluator(
    id="required-fields",
    scope="end-to-end",
    description="Judges whether each housing listing includes all requested details.",
    instructions=(
        "Evaluate the agent's final answer for whether each housing listing it provides "
        "includes all of these required fields: address or neighborhood, rent, total "
        "monthly cost with a breakdown, beds and baths, distance from work, estimated "
        "commute time, key amenities, and a link. Treat listings as incomplete if any "
        "required field is missing, too vague to be useful, or only implied. Score "
        "highest only when every listing includes all required fields clearly and "
        "specifically. Reduce the score when one or more listings omit required details "
        "or present them ambiguously."
    ),
    model=ModelSpec(name="gpt-5.4"),
)
