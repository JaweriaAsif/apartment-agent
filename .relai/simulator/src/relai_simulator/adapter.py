from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path

from agents import Runner, SQLiteSession

from apartment_agent.agent import build_agent
from relai_simulator.adapter_contract import AgentAdapter, AgentTurnResult


class ProjectAgentAdapter:
    def __init__(self) -> None:
        self.agent = build_agent()
        self.agent_or_tools = self.agent
        self._session_tmpdir = tempfile.TemporaryDirectory(prefix="relai-apartment-agent-")
        self._session = SQLiteSession(
            f"relai-sim-{uuid.uuid4().hex[:12]}",
            str(Path(self._session_tmpdir.name) / "simulator-session.db"),
        )

    async def run_turn(self, user_message: str) -> AgentTurnResult:
        result = await asyncio.to_thread(
            Runner.run_sync,
            self.agent,
            user_message,
            session=self._session,
        )
        return AgentTurnResult(
            assistant_message=_stringify_output(getattr(result, "final_output", None)),
        )


def build_agent_adapter() -> AgentAdapter:
    return ProjectAgentAdapter()


def _stringify_output(final_output: object) -> str | None:
    if final_output is None:
        return None
    if isinstance(final_output, str):
        return final_output
    return str(final_output)
