import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FALLBACK_NGN_RATES_PER_1M: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {"input": 450.0, "cached_input": 112.5, "output": 1800.0},
    "claude-haiku-5-5": {"input": 1200.0, "cached_input": 120.0, "output": 6000.0},
    "whisper-1": {"per_minute_ngn": 9.0},
    "tts-1": {"per_1m_chars_ngn": 22500.0},
}


class VoiceTelemetryLogger:
    """Tracks stage-by-stage latency (ms) and per-request NGN spend to JSONL."""

    def __init__(self, log_path: str = "logs/voice_telemetry.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def estimate_turn_cost_ngn(
        self,
        chat_model: str,
        prompt_tokens: int,
        cached_tokens: int,
        completion_tokens: int,
        stt_provider: str,
        audio_duration_sec: float,
        tts_provider: str,
        tts_chars: int,
    ) -> float:
        rates = FALLBACK_NGN_RATES_PER_1M.get(
            chat_model, FALLBACK_NGN_RATES_PER_1M["gemini-3.8-flash"]
        )
        uncached_prompt = max(0, prompt_tokens - cached_tokens)
        llm_cost = (
            (uncached_prompt / 1_000_000) * rates["input"]
            + (cached_tokens / 1_000_000) * rates["cached_input"]
            + (completion_tokens / 1_000_000) * rates["output"]
        )

        stt_cost = 0.0
        if stt_provider == "bridge":
            mins = max(audio_duration_sec, 1.0) / 60.0
            stt_cost = mins * FALLBACK_NGN_RATES_PER_1M["whisper-1"]["per_minute_ngn"]

        tts_cost = 0.0
        if tts_provider == "bridge":
            tts_cost = (tts_chars / 1_000_000) * FALLBACK_NGN_RATES_PER_1M["tts-1"]["per_1m_chars_ngn"]

        return round(llm_cost + stt_cost + tts_cost, 6)

    def log_turn(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload
