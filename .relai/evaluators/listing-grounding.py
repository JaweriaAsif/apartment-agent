
from relai import LLMJudgeEvaluator, ModelSpec

evaluator = LLMJudgeEvaluator(
    id="listing-grounding",
    scope="end-to-end",
    description="Judges whether listings are grounded in search results without fabricated details.",
    instructions=(
        "Judge whether every listing, price, and address in the agent's final answer is grounded in "
        "actual web-search results available in the simulation record, whether any unknown or missing "
        "fields are explicitly marked unknown rather than guessed, and whether the answer avoids "
        "fabricating businesses, prices, addresses, or other listing details.\n\n"
        "Pass only when all substantive listing facts are supported by the search evidence and any "
        "unsupported field is clearly presented as unknown or unavailable.\n\n"
        "Fail if the answer invents a listing, states a price or address not supported by search "
        "results, fills in missing fields with guesses, or presents uncertain information as fact."
    ),
    model=ModelSpec(name="gpt-5.4"),
)
