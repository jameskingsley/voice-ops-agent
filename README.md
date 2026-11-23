# VoiceOps Agent

A push-to-talk voice operations assistant built for Nigerian retail, energy, and logistics operators using Streamlit, the OpenAI Python SDK against the **Bridge** API (https://wekoya-api.wekoya.tech/v1), multi-tier Speech-to-Text (STT), tool-calling LLM reasoning, neural Text-to-Speech (TTS), and per-turn stage latency + Naira (NGN) telemetry.

## Architecture & Voice Pipeline

1. **Push-to-Talk Capture & Fresh Multi-Turn Slots (pp.py):** Uses st.audio_input with dynamic session keys so each completed turn stays in the conversation thread while presenting a fresh recording slot for the next turn (alongside typed follow-ups and sample queries).
2. **Multi-Tier Speech-to-Text (src/audio_io.py):**
   - **Primary:** Bridge OpenAI-compatible udio.transcriptions (whisper-1 with domain prompt hints).
   - **Fast Cloud Fallback:** SpeechRecognition (en-NG locale) for low-latency Nigerian English transcription.
   - **Offline Local Fallback:** aster-whisper (ase.en, int8 CPU) with operational domain initial_prompt.
3. **Tool-Calling Operations Agent (src/assistant.py, src/tools.py):**
   - Executes check_order_status, check_sku_stock, and lookup_policy against data/ops_data.json.
   - Normalizes spoken phonetic variations of order IDs and SKUs (for example, "ROD 422" or "ORD 422" -> ORD-4022, "SKU PKG 012" -> SKU-PKG-012).
   - Uses easoning_effort="low", bounded output tokens, prompt_cache_key + X-Session-Id headers, and automatic fallback from gemini-3.8-flash to claude-haiku-5-5 on 504 / timeout errors.
4. **Text-to-Speech Playback (src/audio_io.py):**
   - **Primary:** Bridge OpenAI-compatible udio.speech.
   - **Fallback:** edge-tts (en-NG-EzinneNeural) with auto-play on the latest response.
5. **Latency & Naira Cost Telemetry (src/telemetry.py):**
   - Logs STT latency, LLM + tool latency, TTS latency, total turn duration (ms), token/cache counts, and estimated NGN cost per turn to logs/voice_telemetry.jsonl, displaying live metrics in the UI and sidebar.

## Project Structure

`	ext
voice-ops-agent/
├── app.py                  # Streamlit push-to-talk UI, multi-turn mic reset, and telemetry display
├── data/
│   └── ops_data.json       # Sample orders, SKU inventory/reorder thresholds, and merchant policies
├── src/
│   ├── __init__.py
│   ├── assistant.py        # Tool-calling loop, prompt caching, and model failover
│   ├── audio_io.py         # Multi-tier STT and TTS synthesis with fallback
│   ├── telemetry.py        # Per-turn latency (ms) and NGN cost logger
│   └── tools.py            # Order, SKU stock, and policy tools + fuzzy ID/SKU normalization
├── tests/
│   └── test_pipeline.py    # Unit checks for tools and NGN cost telemetry
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
Local Setup & Running
PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Set WEKOYA_API_KEY in .env
streamlit run app.py
To run tests:

PowerShell
pytest
