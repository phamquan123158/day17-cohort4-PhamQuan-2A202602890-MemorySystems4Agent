from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """A deliberately simple within-thread-only agent.

    ``thread_id`` is the only memory key.  The same user therefore starts
    with an empty context when they open a new thread.
    """

    def __init__(
        self, config: LabConfig | None = None, force_offline: bool = False
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return a response and cumulative token accounting for a thread."""

        # user_id is intentionally not used as a memory key in the baseline.
        del user_id
        if self.langchain_agent is not None and not self.force_offline:
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            response = self._response_text(result)
            state = self.sessions.setdefault(thread_id, SessionState())
            state.prompt_tokens_processed += estimate_tokens(message)
            state.token_usage += estimate_tokens(response)
            state.messages.extend(
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ]
            )
            return self._result(thread_id, response)

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative output tokens for one thread."""

        state = self.sessions.get(thread_id)
        return state.token_usage if state else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative tokens processed as prompt context."""

        state = self.sessions.get(thread_id)
        return state.prompt_tokens_processed if state else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        del thread_id
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Generate a deterministic response without contacting an API."""

        state = self.sessions.setdefault(thread_id, SessionState())

        # Baseline resends its complete thread history on every turn.  This is
        # intentionally different from AdvancedAgent's compacted context.
        state.prompt_tokens_processed += (
            sum(estimate_tokens(item["content"]) for item in state.messages)
            + estimate_tokens(message)
        )

        response = self._offline_response(state.messages, message)
        state.messages.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
        )
        state.token_usage += estimate_tokens(response)
        return self._result(thread_id, response)

    def _maybe_build_langchain_agent(self):
        """Return a provider model when live mode is explicitly requested."""

        if self.force_offline:
            return None
        return build_chat_model(self.config.model)

    @staticmethod
    def _response_text(result: Any) -> str:
        """Normalize common LangChain response shapes to plain text."""

        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            messages = result.get("messages")
            if messages:
                content = getattr(messages[-1], "content", messages[-1])
                if isinstance(content, str):
                    return content
            content = result.get("content")
            if isinstance(content, str):
                return content
        content = getattr(result, "content", None)
        return content if isinstance(content, str) else str(result)

    @staticmethod
    def _offline_response(
        messages: list[dict[str, str]], message: str
    ) -> str:
        """Answer simple recall questions using this thread's prior messages."""

        normalized = message.casefold()
        previous = [
            item["content"]
            for item in messages
            if item.get("role") == "user"
        ]

        if "tên gì" in normalized or "tên mình" in normalized:
            for item in reversed(previous):
                if "tên là" in item.casefold():
                    return f"Mình nhớ trong cuộc trò chuyện này: {item}"

        if "ở đâu" in normalized or "nơi ở" in normalized:
            for item in reversed(previous):
                lowered = item.casefold()
                if "mình ở" in lowered or "đang ở" in lowered:
                    return f"Mình nhớ trong cuộc trò chuyện này: {item}"

        if "nghề" in normalized or "công việc" in normalized:
            for item in reversed(previous):
                lowered = item.casefold()
                if "làm " in lowered or "nghề" in lowered:
                    return f"Mình nhớ trong cuộc trò chuyện này: {item}"

        return "Mình đã ghi nhận thông tin trong cuộc trò chuyện hiện tại."

    def _result(self, thread_id: str, response: str) -> dict[str, Any]:
        return {
            "response": response,
            "thread_id": thread_id,
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": self.prompt_token_usage(thread_id),
            "compactions": 0,
        }
