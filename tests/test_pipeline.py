import json
from src.tools import VoiceOpsTools
from src.telemetry import VoiceTelemetryLogger


def test_tools_and_telemetry(tmp_path):
    tools = VoiceOpsTools("data/ops_data.json")
    order = json.loads(tools.execute("check_order_status", '{"order_id": "ORD-4022"}'))
    assert order["found"] is True
    assert order["payment_status"] == "pending_reconciliation"

    stock = json.loads(tools.execute("check_sku_stock", '{"sku": "SKU-PKG-012"}'))
    assert stock["found"] is True
    assert stock["needs_reorder"] is True

    logger = VoiceTelemetryLogger(str(tmp_path / "telemetry.jsonl"))
    cost = logger.estimate_turn_cost_ngn(
        chat_model="gemini-3.8-flash",
        prompt_tokens=1500,
        cached_tokens=1000,
        completion_tokens=80,
        stt_provider="faster-whisper (fallback)",
        audio_duration_sec=3.5,
        tts_provider="edge-tts (en-NG-EzinneNeural)",
        tts_chars=180,
    )
    assert cost > 0
