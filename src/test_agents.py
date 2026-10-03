from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = state_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    dummy_model = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,  # low threshold so compaction triggers easily
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # Initially empty
    assert store.read_text(user_id) == ""

    # Write content
    initial_content = "# User Profile: test_user\n\n- **Name**: Khoa\n- **Location**: Da Nang\n"
    written_path = store.write_text(user_id, initial_content)
    assert written_path.exists()
    assert store.read_text(user_id) == initial_content
    assert store.file_size(user_id) > 0

    # Edit content
    success = store.edit_text(user_id, "Location**: Da Nang", "Location**: Hue")
    assert success is True
    assert "Location**: Hue" in store.read_text(user_id)

    # Failed edit (target text not found)
    failed = store.edit_text(user_id, "Nonexistent text", "Replacement")
    assert failed is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread_id = "th-test"

    assert manager.compaction_count(thread_id) == 0

    # Send messages exceeding the 50-token threshold
    manager.append(thread_id, "user", "Tin nhắn dài thứ nhất có rất nhiều thông tin kỹ thuật về hệ thống phân tán và agent.")
    manager.append(thread_id, "assistant", "Phản hồi thứ nhất với nhiều phân tích chi tiết về kiến trúc microservices.")
    manager.append(thread_id, "user", "Tin nhắn dài thứ ba tiếp tục thảo luận về memory compaction và database.")

    # Should have triggered compaction
    assert manager.compaction_count(thread_id) >= 1
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) <= 2
    assert len(ctx["summary"]) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    user_id = "dungct_test"

    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    # Session 1: User introduces facts in thread-1
    msg1 = "Chào bạn, mình tên là DũngCT, ở Huế và làm MLOps engineer."
    baseline.reply(user_id=user_id, thread_id="thread-1", message=msg1)
    advanced.reply(user_id=user_id, thread_id="thread-1", message=msg1)

    # Session 2: User asks recall question in a completely new thread-2
    recall_q = "Nhắc lại giúp mình: mình tên gì và ở đâu?"
    ans_baseline = baseline.reply(user_id=user_id, thread_id="thread-2", message=recall_q)["reply"]
    ans_advanced = advanced.reply(user_id=user_id, thread_id="thread-2", message=recall_q)["reply"]

    # Baseline should NOT know DũngCT or Huế in thread-2
    assert "DũngCT" not in ans_baseline
    assert "Huế" not in ans_baseline

    # Advanced MUST recall DũngCT and Huế from User.md
    assert "DũngCT" in ans_advanced
    assert "Huế" in ans_advanced


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    user_id = "stress_test_user"
    thread_id = "thread-long"

    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    # Send 10 long turns in the same thread
    long_messages = [
        f"Lượt hội thoại thứ {i}: Đây là một đoạn nội dung kỹ thuật rất dài chứa các thông tin về kiến trúc hệ thống, logging, pipeline MLOps và đánh giá hiệu năng mô hình ngôn ngữ lớn nhằm làm tăng kích thước ngữ cảnh của hội thoại."
        for i in range(1, 11)
    ]

    for msg in long_messages:
        baseline.reply(user_id=user_id, thread_id=thread_id, message=msg)
        advanced.reply(user_id=user_id, thread_id=thread_id, message=msg)

    prompt_baseline = baseline.prompt_token_usage(thread_id)
    prompt_advanced = advanced.prompt_token_usage(thread_id)

    # Advanced with compact memory should process fewer prompt tokens than Baseline
    assert advanced.compaction_count(thread_id) > 0
    assert prompt_advanced < prompt_baseline
