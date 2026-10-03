from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    matched = sum(item.casefold() in answer.casefold() for item in expected)
    return matched / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    concise_bonus = 0.1 if len(answer.split()) <= 80 else 0.0
    return min(1.0, recall + concise_bonus)


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config
) -> BenchmarkRow:
    initial_size = 0
    total_recall = 0.0
    total_quality = 0.0
    question_count = 0
    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = f"{agent_name}-{conversation['id']}"
        for turn in conversation.get("turns", []):
            agent.reply(user_id, thread_id, turn)
        for index, item in enumerate(conversation.get("recall_questions", [])):
            result = agent.reply(
                user_id,
                f"{thread_id}-recall-{index}",
                item["question"],
            )
            answer = result["response"]
            expected = item.get("expected_contains", [])
            total_recall += recall_points(answer, expected)
            total_quality += heuristic_quality(answer, expected)
            question_count += 1

        if isinstance(agent, AdvancedAgent):
            initial_size += agent.memory_file_size(user_id)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread) for thread in agent.sessions)
        if isinstance(agent, BaselineAgent)
        else sum(agent.thread_tokens.values()),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread) for thread in agent.sessions)
        if isinstance(agent, BaselineAgent)
        else sum(agent.thread_prompt_tokens.values()),
        recall_score=total_recall / question_count if question_count else 0.0,
        response_quality=total_quality / question_count if question_count else 0.0,
        memory_growth_bytes=initial_size,
        compactions=sum(
            agent.compaction_count(thread)
            for thread in (
                agent.sessions if isinstance(agent, BaselineAgent) else agent.compact_memory.state
            )
        ),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        output.append(
            f"| {row.agent_name} | {row.agent_tokens_only} | "
            f"{row.prompt_tokens_processed} | {row.recall_score:.2f} | "
            f"{row.response_quality:.2f} | {row.memory_growth_bytes} | "
            f"{row.compactions} |"
        )
    return "\n".join(output)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    suites = [
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    ]
    for title, path in suites:
        conversations = load_conversations(path)
        print(f"\n## {title}\n")
        baseline = BaselineAgent(config=config, force_offline=True)
        advanced = AdvancedAgent(config=config, force_offline=True)
        print(
            format_rows(
                [
                    run_agent_benchmark("Baseline", baseline, conversations, config),
                    run_agent_benchmark("Advanced", advanced, conversations, config),
                ]
            )
        )


if __name__ == "__main__":
    main()
