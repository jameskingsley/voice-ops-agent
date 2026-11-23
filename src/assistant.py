import time
import uuid
from typing import Any

from openai import APITimeoutError, BadRequestError, InternalServerError, OpenAI

from src.tools import VoiceOpsTools

VOICE_SYSTEM_PROMPT = """You are VoiceOps Agent, a hands-free operations copilot for Nigerian retail and logistics teams.

Rules for voice responses:
1. Always call check_order_status, check_sku_stock, or lookup_policy before quoting order details, stock counts, or rules.
2. Write responses meant to be spoken aloud: keep answers clear, natural, and under 3 to 4 short sentences.
3. Say currency amounts clearly (for example, '57,000 Naira') and avoid bullet symbols or heavy formatting.
4. Never fabricate order IDs, session references, or stock numbers."""


class VoiceOpsAssistant:
    """Tool-calling LLM agent optimized for low-latency spoken responses and prompt caching."""

    def __init__(
        self,
        client: OpenAI,
        chat_model: str = "gemini-3.8-flash",
        fallback_model: str = "claude-haiku-5-5",
        session_id: str | None = None,
    ) -> None:
        self.client = client
        self.chat_model = chat_model
        self.fallback_model = fallback_model
        self.session_id = session_id or f"voice-{uuid.uuid4().hex[:8]}"
        self.tools = VoiceOpsTools()
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": VOICE_SYSTEM_PROMPT}
        ]

    def _call_llm(self, model_name: str) -> Any:
        try:
            return self.client.chat.completions.create(
                model=model_name,
                messages=self.messages,
                tools=self.tools.schemas(),
                max_tokens=4096,
                reasoning_effort="low",
                prompt_cache_key=self.session_id,
                extra_headers={"X-Session-Id": self.session_id},
            )
        except BadRequestError as exc:
            # Fallback if a routed model rejects reasoning_effort
            if "reasoning_effort" in str(exc).lower():
                return self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    tools=self.tools.schemas(),
                    max_tokens=4096,
                    prompt_cache_key=self.session_id,
                    extra_headers={"X-Session-Id": self.session_id},
                )
            raise

    def respond(self, user_text: str, max_rounds: int = 4) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.messages.append({"role": "user", "content": user_text})

        total_prompt = 0
        total_cached = 0
        total_completion = 0
        tool_events: list[dict[str, Any]] = []
        active_model = self.chat_model
        final_text = ""

        for _ in range(max_rounds):
            try:
                reply = self._call_llm(active_model)
            except (InternalServerError, APITimeoutError):
                active_model = self.fallback_model
                reply = self._call_llm(active_model)

            usage = getattr(reply, "usage", None)
            if usage:
                total_prompt += getattr(usage, "prompt_tokens", 0) or 0
                total_completion += getattr(usage, "completion_tokens", 0) or 0
                details = getattr(usage, "prompt_tokens_details", None)
                if details:
                    total_cached += getattr(details, "cached_tokens", 0) or 0

            msg = reply.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                final_text = (msg.content or "").strip()
                self.messages.append({"role": "assistant", "content": final_text})
                break

            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                output = self.tools.execute(tc.function.name, tc.function.arguments)
                tool_events.append({
                    "tool": tc.function.name,
                    "arguments": tc.function.arguments,
                    "output": output,
                })
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

        return {
            "answer": final_text,
            "model": active_model,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "prompt_tokens": total_prompt,
            "cached_tokens": total_cached,
            "completion_tokens": total_completion,
            "tool_events": tool_events,
        }
