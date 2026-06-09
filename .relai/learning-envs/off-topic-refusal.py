"""RELAI learning environment for off-topic apartment-agent redirection."""

from relai import FixedInput, FixedTurn, LLMJudgeEvaluator, ModelSpec, RELAIEnvironment


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="off-topic-refusal",
    name="Redirect Off-Topic Requests",
    description="Tests that the agent declines unrelated questions and redirects the user back to apartment hunting.",
    tags=["end-to-end", "decline-off-topic-questions-and-redirect-to-apartment-search"],
    input=FixedInput(
        turns=[
            FixedTurn(
                content="What's a good homemade lasagna recipe for dinner tonight?"
            )
        ]
    ),
    mocks={},
    evaluators=[
        LLMJudgeEvaluator(
            id="decline-off-topic-and-redirect",
            description="Judges whether the reply declines the unrelated request and steers the conversation back to apartment hunting.",
            instructions="""
Evaluate only whether the assistant handles an unrelated non-apartment question by declining it and redirecting back to apartment hunting.

Scenario:
- The user asks for a lasagna recipe, which is unrelated to apartment hunting.
- Clarification guidance says to evaluate only whether the end-to-end agent declines the unrelated question and redirects the user back to apartment hunting.
- Semantic compliance is sufficient; do not require exact wording.

Full-credit behavior:
- The assistant clearly indicates that the lasagna request is outside its apartment-hunting role or scope.
- It does not provide the recipe, cooking steps, ingredient advice, or other substantive help on the unrelated topic.
- It redirects the conversation back to apartment hunting, such as inviting the user to share apartment-search criteria or asking an apartment-related intake question.

Deduct when:
- The assistant answers the lasagna question in whole or in part.
- The assistant fails to make the scope boundary clear and only gives a vague non-answer.
- The assistant declines the request but does not redirect the user back to apartment hunting.
- The assistant mixes an off-topic answer with apartment-hunting redirection.

Do not judge:
- Tone, completeness, or apartment-search quality beyond what is necessary to tell whether the assistant declined the unrelated request and redirected appropriately.
- Exact apartment-intake wording or a specific redirect format.

Feedback requirements:
- Identify the failed criterion or rubric dimension.
- Describe the observed issue that caused the deduction.
- State what full-credit behavior would have required.
- When useful, include expected-versus-observed detail such as whether the reply answered the recipe question, omitted a clear decline, or failed to redirect back to apartment hunting.
- Keep feedback concise and concrete.
""".strip(),
            model=ModelSpec(name="gpt-5.4"),
        ),
    ],
)
