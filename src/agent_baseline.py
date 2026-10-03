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
    """Agent A: Baseline Agent.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Forgets long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                # Live LangChain invocation if configured
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    {"configurable": {"thread_id": thread_id}},
                )
                output_text = result["messages"][-1].content
                out_tokens = estimate_tokens(output_text)
                return {"reply": output_text, "tokens": out_tokens}
            except Exception:
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic within-thread memory implementation."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]

        # Calculate prompt context: all previous messages in this session + incoming message
        history_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        new_prompt_tokens = history_tokens + estimate_tokens(message)
        session.prompt_tokens_processed += new_prompt_tokens

        # Record user message in thread session
        session.messages.append({"role": "user", "content": message})

        # Generate response:
        # Since baseline only has within-thread memory, if this is a fresh thread (e.g. recall query),
        # it has NO facts from previous threads.
        reply_text = (
            "Chào bạn, tôi đã ghi nhận phản hồi của bạn trong phiên làm việc này."
        )

        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "reply": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": new_prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire LangChain agent."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            llm = build_chat_model(self.config.model)
            self.langchain_agent = create_react_agent(
                model=llm,
                tools=[],
                checkpointer=MemorySaver(),
            )
        except Exception:
            self.langchain_agent = None
