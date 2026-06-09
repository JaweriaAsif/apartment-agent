# apartment-agent

A small terminal chat demo using the OpenAI Agents SDK for apartment hunting.

The agent asks a fixed set of intake questions about your housing needs (area,
budget, beds/baths, work address, commute, move-in date, must-haves, deal-breakers),
then uses the SDK's hosted `WebSearchTool` to find current rental listings in the
area and returns a ranked list with **total monthly cost** and **distance / commute
time to your workplace**. It keeps multi-turn conversation state in a local SQLite
session and writes a readable JSON conversation log.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Set your API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY.
```

Optional environment variables:

- `OPENAI_MODEL`: model used by the agent, default `gpt-5.4`
- `AGENT_SESSION_ID`: reuse a specific terminal conversation session stored under
  `.agent_sessions/`
- `AGENT_LOG_FILE`: explicit JSON conversation log path override. By default
  each run reserves the next numbered file, such as `logs/conversation-001.json`.

## Run

```bash
uv run apartment-agent
```

Useful commands inside the chat:

- `/help`: show commands
- `/new`: start a fresh conversation session
- `/quit` or `/exit`: stop the program

## Daily new-postings digest

The agent runs only when invoked, so a true "new listings every day" digest needs a
scheduled job (e.g. cron) that launches `uv run apartment-agent` each morning with
your saved criteria. Ask the agent and it will walk you through setting one up.

## Development

```bash
uv run pytest
```

Runtime artifacts are ignored by git:

- `.agent_sessions/`
- `logs/`
