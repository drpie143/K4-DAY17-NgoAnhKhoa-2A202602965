from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from tabulate import tabulate


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
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 to 1.0 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    hits = sum(1 for item in expected if item.lower() in ans_lower)
    return round(hits / len(expected), 3)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline responses."""
    if not answer or not answer.strip():
        return 0.0

    score = 0.0
    # Fact coverage
    r_score = recall_points(answer, expected)
    score += r_score * 0.6

    # Structure & bullet formatting
    if any(marker in answer for marker in ["- ", "* ", "1.", "2.", "•"]):
        score += 0.2
    else:
        score += 0.1

    # Conciseness & non-trivial length
    length = len(answer.strip())
    if 50 <= length <= 600:
        score += 0.2
    elif length > 600:
        score += 0.1
    else:
        score += 0.05

    return round(min(1.0, score), 3)


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    """Evaluate one agent over multiple conversations and recall questions."""
    all_threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    users_seen: set[str] = set()

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        users_seen.add(user_id)
        all_threads.append(conv_id)

        # 1. Feed turns to agent in session
        for turn in conv.get("turns", []):
            agent.reply(user_id=user_id, thread_id=conv_id, message=turn)

        # 2. Ask recall questions in a brand new cross-session thread
        for idx, q_item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"recall-{conv_id}-{idx}"
            all_threads.append(recall_thread)
            question = q_item["question"]
            expected = q_item.get("expected_contains", [])

            res = agent.reply(user_id=user_id, thread_id=recall_thread, message=question)
            ans = res["reply"]

            r_pts = recall_points(ans, expected)
            q_pts = heuristic_quality(ans, expected)
            recall_scores.append(r_pts)
            quality_scores.append(q_pts)

    total_agent_tokens = sum(agent.token_usage(t) for t in all_threads)
    total_prompt_tokens = sum(agent.prompt_token_usage(t) for t in all_threads)
    total_compactions = sum(agent.compaction_count(t) for t in all_threads)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # Memory growth: only Advanced has persistent profile files
    memory_growth = 0
    if isinstance(agent, AdvancedAgent):
        memory_growth = sum(agent.memory_file_size(u) for u in users_seen)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append(
            [
                r.agent_name,
                r.agent_tokens_only,
                r.prompt_tokens_processed,
                f"{r.recall_score * 100:.1f}%",
                f"{r.response_quality:.2f}",
                r.memory_growth_bytes,
                r.compactions,
            ]
        )
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both standard and long-context stress benchmarks."""
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    std_conversations = load_conversations(std_data_path)
    stress_conversations = load_conversations(stress_data_path)

    print("=" * 80)
    print("PHASE 2 - TRACK 3 - DAY 17: MEMORY SYSTEMS BENCHMARK")
    print("=" * 80)

    # 1. Standard Benchmark
    print("\n[1] STANDARD BENCHMARK (10 Conversations, Cross-Session Recall)")
    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    row_base_std = run_agent_benchmark("Baseline", baseline_std, std_conversations, config)
    row_adv_std = run_agent_benchmark("Advanced", advanced_std, std_conversations, config)
    print(format_rows([row_base_std, row_adv_std]))

    # 2. Long-Context Stress Benchmark
    print("\n[2] LONG-CONTEXT STRESS BENCHMARK (16-turn Long Context, Heavy Prompt Load)")
    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    row_base_stress = run_agent_benchmark("Baseline", baseline_stress, stress_conversations, config)
    row_adv_stress = run_agent_benchmark("Advanced", advanced_stress, stress_conversations, config)
    print(format_rows([row_base_stress, row_adv_stress]))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
