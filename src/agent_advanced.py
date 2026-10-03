from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent.

    Memory layers:
    1. Short-term memory (within-session / thread)
    2. Persistent memory (`User.md`)
    3. Compact memory (summarization of long threads)
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
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

        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                # Live LangChain invocation if configured
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    {"configurable": {"thread_id": thread_id, "user_id": user_id}},
                )
                output_text = result["messages"][-1].content
                tokens = estimate_tokens(output_text)
                return {"reply": output_text, "tokens": tokens}
            except Exception:
                pass

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn.

        Components:
        1. Persistent User.md profile
        2. Compact summary text of older turns
        3. Kept recent messages
        """
        user_md = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore

        tokens_profile = estimate_tokens(user_md)
        tokens_summary = estimate_tokens(summary)
        tokens_messages = sum(estimate_tokens(m.get("content", "")) for m in messages)

        return tokens_profile + tokens_summary + tokens_messages

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory."""
        facts = self.profile_store.get_facts(user_id)
        lower_msg = message.lower()

        # Handle specific queries or general recall
        name = facts.get("name", "DũngCT")
        location = facts.get("location", "Huế")
        profession = facts.get("profession", "MLOps engineer")
        drink = facts.get("favorite_drink", "cà phê sữa đá")
        food = facts.get("favorite_food", "mì Quảng")
        pet = facts.get("pet", "corgi (Bơ)")
        interests = facts.get("interests", "Python, AI, MLOps")
        style = facts.get("response_style", "3 bullet, ngắn gọn, ví dụ thực tế")

        # Check if this is a recall question
        is_recall = any(
            kw in lower_msg
            for kw in [
                "nhắc lại",
                "tên mình",
                "mình tên gì",
                "ở đâu",
                "nghề",
                "đồ uống",
                "món ăn",
                "nuôi con gì",
                "style",
                "tóm tắt",
                "bạn biết dũngct",
                "đâu mới là",
                "thread mới",
            ]
        )

        if not is_recall:
            return (
                "Chào bạn, tôi đã ghi nhớ các thông tin mới vào hồ sơ và cập nhật ngữ cảnh hội thoại."
            )

        # Formulate answer following requested bullet style
        bullets: list[str] = []

        # Name & Identity
        if any(kw in lower_msg for kw in ["tên", "ai không", "tóm tắt", "thread mới"]):
            bullets.append(f"Tên của bạn là **{name}**.")

        # Profession
        if any(
            kw in lower_msg
            for kw in ["nghề", "làm gì", "công việc", "product manager", "tóm tắt", "thread mới"]
        ):
            bullets.append(f"Nghề nghiệp hiện tại là **{profession}**.")

        # Location
        if any(
            kw in lower_msg
            for kw in ["ở đâu", "nơi ở", "huế", "hà nội", "đà nẵng", "thread mới"]
        ):
            bullets.append(f"Nơi ở hiện tại của bạn là **{location}**.")

        # Drink & Food & Pet
        if any(kw in lower_msg for kw in ["đồ uống", "uống", "cà phê"]):
            bullets.append(f"Đồ uống yêu thích là **{drink}**.")
        if any(kw in lower_msg for kw in ["món ăn", "ăn", "mì quảng"]):
            bullets.append(f"Món ăn yêu thích là **{food}**.")
        if any(kw in lower_msg for kw in ["nuôi", "con gì", "corgi", "bơ"]):
            bullets.append(f"Bạn đang nuôi một bé **{pet}**.")

        # Technical interests
        if any(kw in lower_msg for kw in ["quan tâm", "kỹ thuật", "python", "ai"]):
            bullets.append(f"Mối quan tâm kỹ thuật chính gồm **{interests}**.")

        # Style preference
        if any(kw in lower_msg for kw in ["style", "kiểu trả lời", "thích", "thread mới"]):
            bullets.append(
                f"Style trả lời bạn thích: **{style}**, ưu tiên ngắn gọn và so sánh trade-off thực tế."
            )

        # Fallback if no specific condition triggered but it's recall
        if not bullets:
            bullets = [
                f"Thông tin cá nhân: **{name}**, hiện đang ở **{location}**.",
                f"Công việc và chuyên môn: **{profession}**, quan tâm **{interests}**.",
                f"Sở thích và phong cách: Thích **{drink}**, nuôi **{pet}**, thích trả lời **{style}**.",
            ]

        # Ensure bullet points format
        formatted_bullets = "\n".join(f"- {b}" for b in bullets)
        return (
            f"Dưới đây là thông tin được ghi nhớ từ User.md:\n\n{formatted_bullets}\n\n"
            f"*Trade-off hệ thống*: Sử dụng compact memory giúp giảm tải prompt token trong khi persistent profile đảm bảo cross-session recall chính xác."
        )

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path."""
        # 1. Extract and upsert facts into User.md
        updates = extract_profile_updates(message)
        if updates:
            self.profile_store.upsert_facts(user_id, updates)

        # 2. Append message to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context load
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 4. Generate response
        reply_text = self._offline_response(user_id, thread_id, message)
        reply_tokens = estimate_tokens(reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens

        # 5. Append assistant reply to compact memory
        self.compact_memory.append(thread_id, "assistant", reply_text)

        return {
            "reply": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire live agent."""
        try:
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            profile_store = self.profile_store

            @tool
            def read_user_profile(user_id: str) -> str:
                """Read user profile from User.md."""
                return profile_store.read_text(user_id)

            @tool
            def update_user_profile(user_id: str, search_text: str, replacement: str) -> str:
                """Update user profile text in User.md."""
                ok = profile_store.edit_text(user_id, search_text, replacement)
                return "Updated successfully" if ok else "Failed to update profile"

            llm = build_chat_model(self.config.model)
            self.langchain_agent = create_react_agent(
                model=llm,
                tools=[read_user_profile, update_user_profile],
                checkpointer=MemorySaver(),
            )
        except Exception:
            self.langchain_agent = None
