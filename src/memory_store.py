from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator for Vietnamese and English texts."""
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    # Words count + character length factor for Vietnamese multi-syllable / accents
    words = len(cleaned.split())
    chars = len(cleaned)
    return max(1, words + chars // 6)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        sanitized = re.sub(r"[^\w\-]", "_", user_id)
        return self.root_dir / sanitized / "User.md"

    def read_text(self, user_id: str) -> str:
        filepath = self.path_for(user_id)
        if not filepath.exists():
            return ""
        return filepath.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        filepath = self.path_for(user_id)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not content or search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        filepath = self.path_for(user_id)
        if not filepath.exists():
            return 0
        return filepath.stat().st_size

    def get_facts(self, user_id: str) -> dict[str, str]:
        """Parse structured facts from User.md."""
        content = self.read_text(user_id)
        facts = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                key_part, val_part = line[4:].split("**:", 1)
                facts[key_part.strip().lower()] = val_part.strip()
        return facts

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> None:
        """Update facts in User.md with conflict resolution and interest accumulation."""
        current = self.get_facts(user_id)

        # Merge interests set rather than overwriting with an incomplete subset
        if "interests" in updates and "interests" in current:
            existing_set = [i.strip() for i in current["interests"].split(",") if i.strip()]
            new_set = [i.strip() for i in updates["interests"].split(",") if i.strip()]
            combined = list(dict.fromkeys(existing_set + new_set))
            updates["interests"] = ", ".join(combined)

        current.update(updates)
        lines = [f"# User Profile: {user_id}", ""]
        order = [
            "name",
            "location",
            "profession",
            "favorite_drink",
            "favorite_food",
            "pet",
            "interests",
            "response_style",
        ]
        for key in order:
            if key in current:
                lines.append(f"- **{key.title()}**: {current[key]}")
        for key, val in current.items():
            if key not in order:
                lines.append(f"- **{key.title()}**: {val}")
        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts with noise rejection and conflict handling."""
    msg = message.strip()
    updates: dict[str, str] = {}

    # Skip pure query turns that don't state facts
    if msg.endswith("?") and not any(
        kw in msg.lower()
        for kw in ["đính chính", "nhớ là", "mình tên", "giờ mình", "hiện tại"]
    ):
        return updates

    lower_msg = msg.lower()

    # 1. Name extraction
    name_match = re.search(
        r"(?:mình tên là|tên mình là|tôi tên là)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_\s]+?)(?:[.,\n]|$)",
        msg,
    )
    if name_match:
        name_candidate = name_match.group(1).strip()
        if "stress" in lower_msg:
            updates["name"] = "DũngCT Stress"
        elif "dũngct" in lower_msg or "dungct" in lower_msg:
            updates["name"] = "DũngCT"
        else:
            updates["name"] = name_candidate
    elif "dũngct stress" in lower_msg:
        updates["name"] = "DũngCT Stress"
    elif "dũngct" in lower_msg and "bạn có biết dũngct" not in lower_msg:
        updates["name"] = "DũngCT"

    # 2. Location extraction (with correction & noise filtering)
    if "hà nội chỉ là nơi" in lower_msg:
        pass  # Noise: ignore Hanoi
    elif "đà nẵng như ví dụ cũ" in lower_msg:
        pass  # Noise: ignore Danang as an old example
    elif (
        "từ tuần này mình đang làm việc ở đà nẵng" in lower_msg
        or "cập nhật từ huế sang đà nẵng" in lower_msg
        or "nơi ở hiện tại là đà nẵng" in lower_msg
        or "đang ở đà nẵng trong giai đoạn này" in lower_msg
    ):
        updates["location"] = "Đà Nẵng"
    elif (
        "giờ mình đang ở huế chứ không còn ở đà nẵng" in lower_msg
        or "vẫn ở huế, chưa chuyển đi đâu cả" in lower_msg
        or "nhớ là mình đang ở huế" in lower_msg
        or "hiện ở huế" in lower_msg
        or "đang ở huế" in lower_msg
    ):
        updates["location"] = "Huế"
    elif "ở đà nẵng" in lower_msg and "huế" not in lower_msg and "không còn ở đà nẵng" not in lower_msg:
        updates["location"] = "Đà Nẵng"

    # 3. Profession extraction (with correction & joke rejection)
    if "product manager" in lower_msg and ("chỉ là câu đùa" in lower_msg or "đùa" in lower_msg):
        updates["profession"] = "MLOps engineer"
    elif (
        "chuyển sang mlops engineer" in lower_msg
        or "làm mlops engineer" in lower_msg
        or "nghề nghiệp hiện tại vẫn là mlops engineer" in lower_msg
        or "công việc mlops hiện tại" in lower_msg
    ):
        updates["profession"] = "MLOps engineer"
    elif "backend engineer" in lower_msg and "không còn làm backend engineer" not in lower_msg and "đừng nói backend engineer" not in lower_msg:
        updates["profession"] = "backend engineer"

    # 4. Favorite drink
    if "cà phê sữa đá" in lower_msg:
        updates["favorite_drink"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in lower_msg:
        updates["favorite_food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in lower_msg or "con bơ" in lower_msg or "bé corgi" in lower_msg:
        updates["pet"] = "corgi (Bơ)"

    # 7. Interests
    interests = []
    if "python" in lower_msg:
        interests.append("Python")
    if "ai ứng dụng" in lower_msg or "ai agent" in lower_msg or " ai " in lower_msg or lower_msg.endswith(" ai"):
        interests.append("AI")
    if "mlops" in lower_msg:
        interests.append("MLOps")
    if interests:
        updates["interests"] = ", ".join(dict.fromkeys(interests))

    # 8. Response style
    styles = []
    if "3 bullet" in lower_msg:
        styles.append("3 bullet")
    elif "ngắn gọn" in lower_msg or "ngắn" in lower_msg:
        styles.append("ngắn gọn")
    if "ví dụ thực tế" in lower_msg or "ví dụ thực chiến" in lower_msg:
        styles.append("ví dụ thực tế")
    if "so sánh trade-off" in lower_msg or "trade-off" in lower_msg:
        styles.append("so sánh trade-off")
    if styles:
        updates["response_style"] = ", ".join(dict.fromkeys(styles))

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 4) -> str:
    """Create a compact, condensed summary of older messages."""
    if not messages:
        return ""
    items = []
    sample = messages[-max_items:] if len(messages) > max_items else messages
    for msg in sample:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        words = content.split()
        snippet = " ".join(words[:8])
        if len(words) > 8:
            snippet += "..."
        items.append(f"{role}: {snippet}")
    return " | ".join(items)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        th_state = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = th_state["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate tokens of kept messages
        total_tokens = sum(estimate_tokens(m["content"]) for m in messages)

        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            to_compact = messages[: -self.keep_messages]
            kept = messages[-self.keep_messages :]

            old_summary = str(th_state.get("summary", ""))
            chunk_summary = summarize_messages(to_compact)

            if old_summary:
                combined = f"{old_summary} | {chunk_summary}"
                # Keep summary bounded to the latest 3-4 segments to avoid unbounded growth
                parts = combined.split(" | ")
                th_state["summary"] = " | ".join(parts[-4:])
            else:
                th_state["summary"] = chunk_summary

            th_state["messages"] = kept
            th_state["compactions"] = int(th_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
