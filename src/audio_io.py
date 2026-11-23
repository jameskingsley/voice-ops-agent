import asyncio
import io
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from openai import OpenAI

DOMAIN_STT_PROMPT = (
    "VoiceOps Agent, Naira, order ORD-4021, ORD-4022, SKU-PWR-100, "
    "SKU-INV-500, SKU-PKG-012, Lagos, Enugu, dispatch cutoff, bank transfer."
)


def estimate_wav_duration_sec(audio_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return round(frames / float(rate), 2)
    except Exception:
        pass
    return round(max(len(audio_bytes) / 32000.0, 1.0), 2)


class VoiceAudioEngine:
    """Handles STT and TTS via Bridge OpenAI-compatible endpoints with fast cloud/local fallbacks."""

    def __init__(
        self,
        client: OpenAI,
        stt_model: str = "whisper-1",
        tts_model: str = "tts-1",
        tts_voice: str = "alloy",
        edge_voice: str = "en-NG-EzinneNeural",
    ) -> None:
        self.client = client
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.edge_voice = edge_voice
        self._local_whisper = None
        self._bridge_stt_available = True
        self._bridge_tts_available = True

    def _get_local_whisper(self):
        if self._local_whisper is None:
            from faster_whisper import WhisperModel
            self._local_whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
        return self._local_whisper

    def transcribe(self, audio_bytes: bytes, filename: str = "voice_input.wav") -> dict[str, Any]:
        duration_sec = estimate_wav_duration_sec(audio_bytes)
        t0 = time.perf_counter()
        bridge_err = None

        # 1. Primary: Bridge OpenAI-compatible audio.transcriptions
        if self._bridge_stt_available:
            try:
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = filename
                resp = self.client.audio.transcriptions.create(
                    model=self.stt_model,
                    file=audio_file,
                    prompt=DOMAIN_STT_PROMPT,
                )
                text = (getattr(resp, "text", None) or str(resp)).strip()
                if text:
                    return {
                        "text": text,
                        "provider": "bridge",
                        "model": self.stt_model,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "duration_sec": duration_sec,
                        "fallback_reason": None,
                    }
            except Exception as exc:
                bridge_err = str(exc)
                if any(code in bridge_err for code in ("404", "405", "501", "Not Found")):
                    self._bridge_stt_available = False

        # 2. Fast Cloud Fallback: SpeechRecognition (en-NG locale)
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-NG").strip()
            if text:
                return {
                    "text": text,
                    "provider": "speech-recognition (en-NG)",
                    "model": "google-web-speech-en-NG",
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "duration_sec": duration_sec,
                    "fallback_reason": bridge_err,
                }
        except Exception:
            pass

        # 3. Offline Local Fallback: faster-whisper CPU model
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            model = self._get_local_whisper()
            segments, _ = model.transcribe(
                str(tmp_path),
                beam_size=1,
                initial_prompt=DOMAIN_STT_PROMPT,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return {
                "text": text,
                "provider": "faster-whisper (offline fallback)",
                "model": "base.en",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "duration_sec": duration_sec,
                "fallback_reason": bridge_err,
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _synthesize_edge_tts(self, text: str) -> bytes:
        import edge_tts
        communicate = edge_tts.Communicate(text=text, voice=self.edge_voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    def synthesize(self, text: str) -> dict[str, Any]:
        clean_text = text.replace("*", "").replace("#", "").strip()
        t0 = time.perf_counter()
        bridge_err = None

        # 1. Primary: Bridge OpenAI-compatible audio.speech
        if self._bridge_tts_available:
            try:
                resp = self.client.audio.speech.create(
                    model=self.tts_model,
                    voice=self.tts_voice,
                    input=clean_text,
                    response_format="mp3",
                )
                audio_bytes = resp.read() if hasattr(resp, "read") else bytes(resp)
                if audio_bytes:
                    return {
                        "audio_bytes": audio_bytes,
                        "format": "audio/mp3",
                        "provider": "bridge",
                        "model": self.tts_model,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "chars": len(clean_text),
                        "fallback_reason": None,
                    }
            except Exception as exc:
                bridge_err = str(exc)
                if any(code in bridge_err for code in ("404", "405", "501", "Not Found")):
                    self._bridge_tts_available = False

        # 2. Fallback: edge-tts neural voice (en-NG-EzinneNeural)
        audio_bytes = asyncio.run(self._synthesize_edge_tts(clean_text))
        return {
            "audio_bytes": audio_bytes,
            "format": "audio/mp3",
            "provider": f"edge-tts ({self.edge_voice})",
            "model": self.edge_voice,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "chars": len(clean_text),
            "fallback_reason": bridge_err,
        }
