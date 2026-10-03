from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Build an isolated, deterministic configuration for tests."""

    return replace(
        load_config(tmp_path),
        compact_threshold_tokens=80,
        compact_keep_messages=2,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    assert store.read_text("dungct") == "# User Profile\n"
    path = store.write_text("dungct", "# User Profile\n- Name: DungCT\n")

    assert path.name == "User.md"
    assert store.file_size("dungct") > 0
    assert store.edit_text("dungct", "Name: DungCT", "Name: An")
    assert "Name: An" in store.read_text("dungct")
    assert not store.edit_text("dungct", "missing", "replacement")


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    for index in range(8):
        agent.reply(
            "compact-user",
            "long-thread",
            f"Đây là một message dài dùng để kiểm tra compact memory lần {index}.",
        )

    context = agent.compact_memory.context("long-thread")
    assert agent.compaction_count("long-thread") > 0
    assert context["summary"]
    assert len(context["messages"]) <= config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    advanced.reply("user-1", "thread-1", "Mình tên là DũngCT.")
    advanced_answer = advanced.reply("user-1", "thread-2", "Mình tên gì?")
    assert "DũngCT" in advanced_answer["response"]

    baseline.reply("user-1", "thread-1", "Mình tên là DũngCT.")
    baseline_answer = baseline.reply("user-1", "thread-2", "Mình tên gì?")
    assert "DũngCT" not in baseline_answer["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    messages = [
        f"Đây là một đoạn hội thoại dài để benchmark prompt context và compact memory {i}."
        for i in range(20)
    ]

    for message in messages:
        baseline.reply("same-user", "baseline-long", message)
        advanced.reply("same-user", "advanced-long", message)

    assert advanced.compaction_count("advanced-long") > 0
    assert advanced.prompt_token_usage("advanced-long") < baseline.prompt_token_usage(
        "baseline-long"
    )


def test_confidence_threshold_ignores_noise_and_keeps_correction(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    agent.reply("user-1", "facts", "Mình đang ở Huế và đang làm MLOps engineer.")
    agent.reply(
        "user-1",
        "facts",
        "Có lúc mình đùa là hay là chuyển sang product manager.",
    )
    profile = agent.profile_store.read_text("user-1")
    assert "Profession: MLOps engineer" in profile
    assert "product manager" not in profile

    agent.reply(
        "user-1",
        "facts",
        "Mình đính chính: hiện ở Đà Nẵng chứ không còn ở Huế.",
    )
    profile = agent.profile_store.read_text("user-1")
    assert "Location: Đà Nẵng" in profile
    assert "Location: Huế" not in profile
