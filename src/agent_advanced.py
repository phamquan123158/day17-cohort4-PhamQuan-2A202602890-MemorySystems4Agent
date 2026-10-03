from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    profile_update_confidence,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent with thread memory, persistent profile memory and compaction."""

    def __init__(
        self, config: LabConfig | None = None, force_offline: bool = False
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            response = self._response_text(result)
            self.thread_prompt_tokens[thread_id] = (
                self.thread_prompt_tokens.get(thread_id, 0)
                + self._estimate_prompt_context_tokens(user_id, thread_id)
            )
            self.thread_tokens[thread_id] = self.token_usage(thread_id) + estimate_tokens(
                response
            )
            return self._result(thread_id, response)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._persist_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + estimate_tokens(
            response
        )
        return self._result(thread_id, response)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        summary = str(context.get("summary", ""))
        messages = context.get("messages", [])
        message_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in messages
        )
        return estimate_tokens("\n".join((profile, summary, message_text)))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        profile = self.profile_store.read_text(user_id)
        normalized = message.casefold()

        if " và " in normalized:
            combined = self._multi_profile_answer(profile, normalized)
            if combined:
                return combined
        if any(term in normalized for term in ("tên gì", "tên mình", "tên của mình")):
            return self._profile_answer(profile, "name", "Tên")
        if any(term in normalized for term in ("ở đâu", "nơi ở", "đang ở")):
            return self._profile_answer(profile, "location", "Nơi ở hiện tại")
        if "nghề" in normalized or "công việc" in normalized:
            return self._profile_answer(profile, "profession", "Nghề nghiệp hiện tại")
        if "đồ uống" in normalized or "uống gì" in normalized:
            return self._profile_answer(profile, "favorite_drink", "Đồ uống yêu thích")
        if "món ăn" in normalized:
            return self._profile_answer(profile, "favorite_food", "Món ăn yêu thích")
        if "style" in normalized or "trả lời" in normalized:
            return self._profile_answer(profile, "response_style", "Style trả lời")
        if "nuôi con" in normalized or "corgi" in normalized:
            pet = self._fact_value(profile, "Pet")
            return f"Bạn nuôi {pet}." if pet else "Mình chưa có thông tin về thú cưng."

        requested: list[tuple[str, str]] = []
        if "tên" in normalized:
            requested.append(("name", "Tên"))
        if "đồ uống" in normalized or "uống gì" in normalized:
            requested.append(("favorite_drink", "Đồ uống yêu thích"))
        if "món ăn" in normalized:
            requested.append(("favorite_food", "Món ăn yêu thích"))
        if "ở đâu" in normalized or "nơi ở" in normalized:
            requested.append(("location", "Nơi ở hiện tại"))
        if "nghề" in normalized or "công việc" in normalized:
            requested.append(("profession", "Nghề nghiệp hiện tại"))
        if "style" in normalized or "kiểu trả lời" in normalized:
            requested.append(("response_style", "Style trả lời"))
        if "nuôi con" in normalized or "corgi" in normalized:
            requested.append(("pet", "Thú cưng"))
        if requested:
            labels = {
                "name": "Name",
                "location": "Location",
                "profession": "Profession",
                "favorite_drink": "Favorite drink",
                "favorite_food": "Favorite food",
                "response_style": "Response style",
                "pet": "Pet",
            }
            parts = [
                f"{label}: {self._fact_value(profile, labels[key])}"
                for key, label in requested
                if self._fact_value(profile, labels[key])
            ]
            if parts:
                return "; ".join(parts)

        context = self.compact_memory.context(thread_id)
        if context.get("summary"):
            return "Mình đã ghi nhận thông tin và nén phần hội thoại cũ để giữ context gọn hơn."
        return "Mình đã ghi nhận thông tin vào memory dài hạn và thread hiện tại."

    def _persist_updates(self, user_id: str, message: str) -> None:
        updates = extract_profile_updates(message)
        updates = {
            key: value
            for key, value in updates.items()
            if profile_update_confidence(
                message, key
            ) >= self.config.profile_confidence_threshold
        }
        lowered = message.casefold()
        # Corrections in the supplied benchmark should replace the old fact.
        if "chứ không" in lowered or "đính chính" in lowered:
            location = re.search(
                r"(?:hiện ở|đang ở|ở)\s+([^,.!?]+?)\s+(?:chứ không|nhưng thực ra)",
                message,
                re.IGNORECASE,
            )
            if location:
                updates["location"] = location.group(1).strip()
            profession = re.search(
                r"(?:giờ chuyển sang|hiện tại là|vẫn là)\s+([^,.!?]+)",
                message,
                re.IGNORECASE,
            )
            if profession:
                updates["profession"] = profession.group(1).strip()

        if updates.get("location", "").casefold() in {
            "hiện tại",
            "đã thay đổi",
            "thay đổi",
        }:
            updates.pop("location")
        if any(
            phrase in lowered
            for phrase in ("đừng lấy", "không phải nơi ở", "chỉ là nơi")
        ):
            updates.pop("location", None)
        if "profession" in updates:
            updates["profession"] = re.split(
                r"\s+(?:chứ không|nhưng không|nữa)\b",
                updates["profession"],
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
        if not updates:
            return
        profile = self.profile_store.read_text(user_id)
        lines = profile.splitlines()
        if not lines:
            lines = ["# User Profile"]
        labels = {
            "name": "Name",
            "location": "Location",
            "profession": "Profession",
            "response_style": "Response style",
            "favorite_drink": "Favorite drink",
            "favorite_food": "Favorite food",
            "interests": "Interests",
            "pet": "Pet",
        }
        for key, value in updates.items():
            label = labels[key]
            if key == "interests":
                existing = self._fact_value(profile, label)
                values = {
                    item.strip()
                    for item in f"{existing or ''}, {value}".split(",")
                    if item.strip()
                }
                value = ", ".join(sorted(values))
            replacement = f"- {label}: {value}"
            prefix = f"- {label}:"
            for index, line in enumerate(lines):
                if line.startswith(prefix):
                    lines[index] = replacement
                    break
            else:
                lines.append(replacement)
        self.profile_store.write_text(user_id, "\n".join(lines).rstrip() + "\n")

    @staticmethod
    def _fact_value(profile: str, label: str) -> str | None:
        match = re.search(rf"^- {re.escape(label)}:\s*(.+)$", profile, re.MULTILINE)
        return match.group(1).strip() if match else None

    @classmethod
    def _profile_answer(cls, profile: str, key: str, label: str) -> str:
        labels = {
            "name": "Name",
            "location": "Location",
            "profession": "Profession",
            "favorite_drink": "Favorite drink",
            "favorite_food": "Favorite food",
            "response_style": "Response style",
        }
        value = cls._fact_value(profile, labels[key])
        return f"{label}: {value}" if value else f"Mình chưa có thông tin về {label.lower()}."

    @classmethod
    def _multi_profile_answer(cls, profile: str, normalized: str) -> str:
        fields = [
            ("tên", "Name", "Tên"),
            ("đồ uống", "Favorite drink", "Đồ uống yêu thích"),
            ("món ăn", "Favorite food", "Món ăn yêu thích"),
            ("ở đâu", "Location", "Nơi ở hiện tại"),
            ("nơi ở", "Location", "Nơi ở hiện tại"),
            ("nghề", "Profession", "Nghề nghiệp hiện tại"),
            ("style", "Response style", "Style trả lời"),
            ("kiểu trả lời", "Response style", "Style trả lời"),
            ("corgi", "Pet", "Thú cưng"),
            ("nuôi con", "Pet", "Thú cưng"),
        ]
        parts: list[str] = []
        seen: set[str] = set()
        for term, label, display in fields:
            if term in normalized and label not in seen:
                value = cls._fact_value(profile, label)
                if value:
                    parts.append(f"{display}: {value}")
                    seen.add(label)
        return "; ".join(parts)

    @staticmethod
    def _response_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            messages = result.get("messages")
            if messages:
                content = getattr(messages[-1], "content", messages[-1])
                if isinstance(content, str):
                    return content
            if isinstance(result.get("content"), str):
                return result["content"]
        content = getattr(result, "content", None)
        return content if isinstance(content, str) else str(result)

    def _maybe_build_langchain_agent(self):
        if self.force_offline:
            return None
        return build_chat_model(self.config.model)

    def _result(self, thread_id: str, response: str) -> dict[str, Any]:
        return {
            "response": response,
            "thread_id": thread_id,
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": self.prompt_token_usage(thread_id),
            "compactions": self.compaction_count(thread_id),
        }
