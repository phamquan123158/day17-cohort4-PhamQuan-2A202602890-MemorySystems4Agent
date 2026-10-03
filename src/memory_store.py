from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    cleaned = " ".join(text.split())
    return 0 if not cleaned else max(1, (len(cleaned) + 3) // 4)


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", user_id.strip()).strip("-_")
        if not slug:
            raise ValueError("user_id must contain at least one safe character")
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User Profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0


def extract_profile_updates(message: str) -> dict[str, str]:
    text = " ".join(message.split())
    lowered = text.casefold()
    if not text or text.endswith("?") or lowered.startswith(("bạn có biết", "mình tên gì")):
        return {}

    updates: dict[str, str] = {}
    patterns = {
        "name": r"\bmình tên là\s+([^,.!?]+)",
        "location": r"\b(?:mình\s+(?:đang\s+)?|hiện(?:\s+tại)?\s+|đang\s+)ở\s+([^,.!?]+?)(?=\s+và\s+đang\s+làm\b|[,.!?]|$)",
        "profession": r"\b(?:mình\s+(?:đang\s+)?làm|và\s+đang\s+làm|giờ\s+chuyển\s+sang|hiện\s+tại\s+là)\s+([^,.!?]+)",
        "favorite_drink": r"\b(?:đồ uống yêu thích là|mình thích uống)\s+([^,.!?]+)",
        "favorite_food": r"\b(?:món ăn yêu thích là)\s+([^,.!?]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            updates[key] = match.group(1).strip()
    if "không còn làm" in lowered and not re.search(
        r"(?:giờ chuyển sang|hiện tại là)", text, re.IGNORECASE
    ):
        updates.pop("profession", None)

    if any(term in lowered for term in ("trả lời ngắn gọn", "bullet ngắn", "có ví dụ thực tế")):
        updates["response_style"] = "ngắn gọn, có bullet và ví dụ thực tế"
    if "3 bullet" in lowered or "thành bullet" in lowered:
        updates["response_style"] = "3 bullet ngắn, có ví dụ thực chiến, ưu tiên trade-off"
    drink_match = re.search(
        r"mình thích[^.?!]*(cà phê(?:\s+sữa\s+đá)?|trà|nước ép)",
        text,
        re.IGNORECASE,
    )
    if drink_match:
        updates["favorite_drink"] = drink_match.group(1).strip()
    pet_match = re.search(
        r"(?:nuôi|con)\s+(?:một\s+bé\s+)?(corgi|chó|mèo)(?:\s+tên\s+([^,.!?]+))?",
        text,
        re.IGNORECASE,
    )
    if pet_match:
        pet = pet_match.group(1)
        if pet_match.group(2):
            pet += f" tên {pet_match.group(2).strip()}"
        updates["pet"] = pet
    interests = [
        item for item in ("Python", "AI", "MLOps", "RAG") if item.casefold() in lowered
    ]
    if interests:
        updates["interests"] = ", ".join(interests)
    return updates


def profile_update_confidence(message: str, key: str) -> float:
    """Estimate confidence that a candidate fact is a stable user fact."""

    lowered = " ".join(message.split()).casefold()
    if not lowered or lowered.endswith("?"):
        return 0.0
    if any(marker in lowered for marker in ("hay là", "chỉ là câu đùa", "đùa với")):
        return 0.1
    if key == "profession" and "không còn" in lowered and not any(
        marker in lowered for marker in ("giờ chuyển sang", "hiện tại là")
    ):
        return 0.1
    if key in {"name", "location", "profession"}:
        return 0.95
    if key in {"response_style", "favorite_drink", "favorite_food", "pet"}:
        return 0.9
    return 0.8


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    selected = messages[-max_items:]
    lines = ["Earlier conversation summary:"]
    lines.extend(f"- {item['role']}: {item['content']}" for item in selected)
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if not thread_id:
            raise ValueError("thread_id must not be empty")
        thread = self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )
        messages = thread["messages"]
        if not isinstance(messages, list):
            raise TypeError("thread messages must be a list")
        messages.append({"role": role, "content": content})
        summary = str(thread.get("summary", ""))
        total = estimate_tokens(summary) + sum(
            estimate_tokens(item["content"]) for item in messages
        )
        if total <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        older = messages[:-self.keep_messages]
        thread["summary"] = summarize_messages(older)
        thread["messages"] = messages[-self.keep_messages:]
        thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )
        return {
            "messages": list(thread["messages"]),
            "summary": thread["summary"],
            "compactions": thread["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
