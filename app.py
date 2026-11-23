import json
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.assistant import VoiceOpsAssistant
from src.audio_io import VoiceAudioEngine
from src.telemetry import VoiceTelemetryLogger


def _get_config(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            val = str(st.secrets[key]).strip()
            if val:
                return val
    except Exception:
        pass
    return os.getenv(key, default).strip()


def load_runtime():
    load_dotenv(override=True)
    api_key = _get_config("WEKOYA_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError(
            "Missing WEKOYA_API_KEY. Set it in .env locally or in Streamlit Cloud -> Settings -> Secrets."
        )
    base_url = _get_config("WEKOYA_BASE_URL", "https://wekoya-api.wekoya.tech/v1")
    chat_model = _get_config("CHAT_MODEL", "gemini-3.8-flash")
    stt_model = _get_config("STT_MODEL", "whisper-1")
    tts_model = _get_config("TTS_MODEL", "tts-1")
    tts_voice = _get_config("TTS_VOICE", "alloy")
    log_path = _get_config("TELEMETRY_LOG_PATH", "logs/voice_telemetry.jsonl")

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, chat_model, stt_model, tts_model, tts_voice, log_path


def run_voice_turn(user_audio_bytes: bytes | None, text_override: str | None = None) -> None:
    t_total = time.perf_counter()
    audio_engine: VoiceAudioEngine = st.session_state.audio_engine
    assistant: VoiceOpsAssistant = st.session_state.assistant
    telemetry: VoiceTelemetryLogger = st.session_state.telemetry

    # Stage 1: Speech-to-Text (or direct text input)
    if user_audio_bytes is not None:
        with st.spinner("1/3 Transcribing speech..."):
            stt_res = audio_engine.transcribe(user_audio_bytes)
        transcript = stt_res["text"]
    else:
        stt_res = {
            "text": text_override or "",
            "provider": "text-input",
            "model": "none",
            "latency_ms": 0.0,
            "duration_sec": 0.0,
            "fallback_reason": None,
        }
        transcript = stt_res["text"]

    if not transcript:
        st.warning("Could not detect speech in the recording. Please try speaking closer to the mic.")
        return

    # Stage 2: Tool-Calling Agent Reasoning
    with st.spinner("2/3 Running tool-calling operations agent..."):
        llm_res = assistant.respond(transcript)

    # Stage 3: Text-to-Speech Synthesis
    with st.spinner("3/3 Synthesizing voice response..."):
        tts_res = audio_engine.synthesize(llm_res["answer"])

    total_ms = round((time.perf_counter() - t_total) * 1000, 1)
    cost_ngn = telemetry.estimate_turn_cost_ngn(
        chat_model=llm_res["model"],
        prompt_tokens=llm_res["prompt_tokens"],
        cached_tokens=llm_res["cached_tokens"],
        completion_tokens=llm_res["completion_tokens"],
        stt_provider=stt_res["provider"],
        audio_duration_sec=stt_res["duration_sec"],
        tts_provider=tts_res["provider"],
        tts_chars=tts_res["chars"],
    )

    log_entry = telemetry.log_turn({
        "session_id": assistant.session_id,
        "transcript": transcript,
        "stt_provider": stt_res["provider"],
        "stt_ms": stt_res["latency_ms"],
        "llm_model": llm_res["model"],
        "llm_ms": llm_res["latency_ms"],
        "tts_provider": tts_res["provider"],
        "tts_ms": tts_res["latency_ms"],
        "total_ms": total_ms,
        "prompt_tokens": llm_res["prompt_tokens"],
        "cached_tokens": llm_res["cached_tokens"],
        "completion_tokens": llm_res["completion_tokens"],
        "cost_ngn": cost_ngn,
    })

    st.session_state.turns.append({
        "transcript": transcript,
        "answer": llm_res["answer"],
        "audio_bytes": tts_res["audio_bytes"],
        "audio_format": tts_res["format"],
        "stt_res": stt_res,
        "llm_res": llm_res,
        "tts_res": tts_res,
        "total_ms": total_ms,
        "cost_ngn": cost_ngn,
        "log_entry": log_entry,
    })


def main() -> None:
    st.set_page_config(page_title="VoiceOps Agent", page_icon="🎙️", layout="wide")
    st.title("VoiceOps Agent")
    st.caption(
        "Push-to-talk operations assistant: STT (Bridge / Google en-NG / faster-whisper fallback) -> "
        "Tool-Calling LLM (gemini-3.8-flash) -> TTS (Bridge / Edge-TTS fallback) with stage latency & NGN cost telemetry."
    )

    try:
        client, chat_model, stt_model, tts_model, tts_voice, log_path = load_runtime()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    if "session_id" not in st.session_state:
        st.session_state.session_id = "voice-ops-session-01"
    if "assistant" not in st.session_state:
        st.session_state.assistant = VoiceOpsAssistant(
            client=client,
            chat_model=chat_model,
            session_id=st.session_state.session_id,
        )
    if "audio_engine" not in st.session_state:
        st.session_state.audio_engine = VoiceAudioEngine(
            client=client,
            stt_model=stt_model,
            tts_model=tts_model,
            tts_voice=tts_voice,
        )
    if "telemetry" not in st.session_state:
        st.session_state.telemetry = VoiceTelemetryLogger(log_path=log_path)
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "mic_counter" not in st.session_state:
        st.session_state.mic_counter = 0

    with st.sidebar:
        st.header("Session & Telemetry")
        st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
        if st.button("Reset Voice Session"):
            st.session_state.turns = []
            st.session_state.mic_counter += 1
            st.session_state.assistant = VoiceOpsAssistant(
                client=client,
                chat_model=chat_model,
                session_id=st.session_state.session_id,
            )
            st.rerun()

        st.divider()
        log_file = Path(log_path)
        if log_file.exists():
            rows = [
                json.loads(line)
                for line in log_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            total_spend = sum(float(r.get("cost_ngn", 0.0)) for r in rows)
            avg_ms = sum(float(r.get("total_ms", 0.0)) for r in rows) / max(len(rows), 1)
            st.metric("Total Session Spend", f"NGN {total_spend:.4f}")
            st.metric("Avg End-to-End Latency", f"{avg_ms:.0f} ms")
            st.dataframe(rows[-8:], use_container_width=True)
        else:
            st.info("Record a voice prompt or click a sample query to populate telemetry.")

    # Conversation history above the next input slot
    for idx, turn in enumerate(st.session_state.turns):
        is_latest = idx == len(st.session_state.turns) - 1
        with st.chat_message("user"):
            st.write(turn["transcript"])
            st.caption(f"STT: {turn['stt_res']['provider']} ({turn['stt_res']['latency_ms']} ms)")

        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.audio(turn["audio_bytes"], format=turn["audio_format"], autoplay=is_latest)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("STT Latency", f"{turn['stt_res']['latency_ms']} ms")
            m2.metric("LLM + Tools", f"{turn['llm_res']['latency_ms']} ms")
            m3.metric("TTS Latency", f"{turn['tts_res']['latency_ms']} ms")
            m4.metric("Turn Cost", f"NGN {turn['cost_ngn']:.5f}")

            st.caption(
                f"Total: {turn['total_ms']} ms | Model: {turn['llm_res']['model']} | "
                f"Prompt tokens: {turn['llm_res']['prompt_tokens']} (Cached: {turn['llm_res']['cached_tokens']}) | "
                f"TTS: {turn['tts_res']['provider']}"
            )
            if turn["llm_res"].get("tool_events"):
                with st.expander("Tool Calls"):
                    st.json(turn["llm_res"]["tool_events"])

    st.divider()
    st.markdown("**Next Voice Message (Push-to-Talk):**")
    mic_label = (
        "Record your first operations question"
        if len(st.session_state.turns) == 0
        else f"Record follow-up #{len(st.session_state.turns) + 1}"
    )
    audio_val = st.audio_input(mic_label, key=f"voice_mic_{st.session_state.mic_counter}")

    st.markdown("**Or Test with Sample Queries:**")
    b1, b2, b3 = st.columns(3)
    sample_query = None
    if b1.button("Voice check: Order ORD-4022"):
        sample_query = "Check order ORD-4022 and tell me its payment and delivery status."
    if b2.button("Voice check: Stock SKU-PKG-012"):
        sample_query = "Check stock for SKU-PKG-012 and whether we need to reorder."
    if b3.button("Voice check: Same-day dispatch cutoff"):
        sample_query = "What is our same-day dispatch cutoff rule?"

    typed_follow_up = st.chat_input("Or type a follow-up operations question...")

    if audio_val is not None:
        raw_bytes = audio_val.getvalue()
        run_voice_turn(user_audio_bytes=raw_bytes)
        st.session_state.mic_counter += 1
        st.rerun()

    if sample_query:
        run_voice_turn(user_audio_bytes=None, text_override=sample_query)
        st.rerun()

    if typed_follow_up:
        run_voice_turn(user_audio_bytes=None, text_override=typed_follow_up)
        st.rerun()


if __name__ == "__main__":
    main()
