# apartment-agent

A small terminal chat agent, built with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/),
that helps you find an apartment. It asks a fixed set of intake questions about your
housing needs, searches the web for current rental listings in your target area, and
returns a ranked shortlist annotated with **total monthly cost** and **distance /
commute time to your workplace**.

It keeps multi-turn conversation state in a local SQLite session and writes a
readable JSON conversation log for every run.

---

## How it works

The agent follows a four-step flow (see `src/apartment_agent/agent.py`):

1. **Intake questionnaire** — on the first turn (or whenever a requirement is
   missing) it asks a fixed numbered list of questions. It will not start searching
   until it has at least your **search area**, **budget**, and **work address**, and
   it skips any questions you've already answered.
2. **Search** — uses the SDK's hosted `WebSearchTool` to find current/new rental
   postings (apartments.com, Zillow, Zumper, HotPads, Craigslist, etc.), discarding
   anything over budget or hitting a deal-breaker.
3. **Cost + commute** — for each kept listing it works out a **total monthly cost**
   (rent + estimated utilities + parking + pet rent) and the **distance and estimated
   commute time** to your work address by your chosen mode. Estimated figures are
   labeled as estimates.
4. **Ranked list** — returns a list ranked by best fit, each entry showing address,
   rent, total cost, beds/baths, distance, commute time, amenities, and a link,
   followed by a short best-picks summary.

### Intake questions

1. Search area — city / neighborhoods / ZIP codes
2. Monthly budget — max rent, and whether that's all-in or rent-only
3. Bedrooms / bathrooms — minimums
4. Work address — where you commute to
5. Commute preference — car, transit, bike, or walk + max acceptable time
6. Move-in date
7. Must-haves — e.g. in-unit laundry, parking, pet-friendly, dishwasher, A/C, gym
8. Deal-breakers — anything that disqualifies a listing outright

### Guardrails

- Never invents listings, prices, or addresses — only reports what web search
  returns, and marks unknown fields as "unknown".
- Always labels estimated costs and commute times as estimates.
- Stays honest when results are thin instead of padding the list with poor matches.

---

## Requirements

- **Python ≥ 3.11**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An **OpenAI API key** with access to the model and hosted web search

## Setup

Install dependencies:

```bash
uv sync
```

Configure your API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | yes | — | Authenticates the agent and hosted web search |
| `OPENAI_MODEL` | no | `gpt-5.4` | Model used by the agent |
| `AGENT_SESSION_ID` | no | new random id | Reuse a specific conversation session under `.agent_sessions/` |
| `AGENT_LOG_FILE` | no | next `logs/conversation-NNN.json` | Explicit conversation-log path override |

## Run

```bash
uv run apartment-agent
```

Optional flags:

```bash
uv run apartment-agent --session-id my-search      # resume/name a session
uv run apartment-agent --log-file logs/search.json # explicit log file
```

In-chat commands:

| Command | Action |
|---------|--------|
| `/help` | Show commands |
| `/new` | Start a fresh conversation session |
| `/quit`, `/exit` | Exit |

### Example session

```
You: I'm looking in Capitol Hill, Seattle. Budget $2,200 all-in, 1BR,
     I work at 400 Broad St and commute by transit (max 30 min).

Assistant: Here are current 1BR listings under $2,200 ranked by fit:
  1. <address> — $1,950 rent (~$2,120 total: +$120 utils, +$50 parking)
     · 1bd/1ba · 2.1 mi from work · ~22 min by transit · in-unit laundry · <link>
  ...
  Best pick: #1 — closest to your transit limit and includes laundry.
```

(Listings are live web results, so exact output varies by day.)

## Sessions & logs

- **Sessions** persist multi-turn state in a local SQLite database under
  `.agent_sessions/conversations.db`. Each run gets a fresh random session id unless
  you pass `--session-id` / `AGENT_SESSION_ID`. `/new` starts a clean session.
- **Logs** are written as readable JSON to `logs/conversation-NNN.json` (auto-numbered),
  each containing a `rounds` list of user/assistant messages.

## Daily new-postings digest

The agent runs only when invoked, so a true "new listings every day" digest needs a
scheduled job (e.g. a `cron` entry) that launches `uv run apartment-agent` each
morning with your saved criteria. Ask the agent in-chat and it will walk you through
setting one up.

## Project layout

```
apartment-agent/
├── pyproject.toml              # project + apartment-agent script entry point
├── src/apartment_agent/
│   ├── agent.py                # agent definition + instructions + WebSearchTool
│   ├── cli.py                  # terminal chat loop, sessions, logging
│   └── logging.py              # JSON conversation log helpers
├── logs/                       # conversation logs (gitignored)
└── .relai/                     # RELAI simulator/eval config (local files gitignored)
```

## Testing with RELAI

This project is set up as a [RELAI](https://relai.ai) agent so you can simulate,
evaluate, and optimize its behavior. After `relai init` builds the simulator harness:

```bash
relai learning-env create --prompt "..."   # define a test scenario
relai simulate --tags all-envs              # run scenarios
relai optimize --tags all-envs --total-rollouts 30
```

Local RELAI files (`.relai/config.toml`, `.relai/simulator.env`, runs, caches) are
gitignored; collaborators run `relai init` to regenerate their own harness.

## Development

```bash
uv run pytest
```

Runtime artifacts ignored by git:

- `.env`
- `.agent_sessions/`
- `logs/`

## License

No license specified yet. Add one (e.g. MIT) before sharing or accepting contributions.
