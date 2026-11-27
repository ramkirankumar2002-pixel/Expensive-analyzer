# tts.py - "Listen to report": turn the AI report text into speech with Gemini text-to-speech.
# (No Streamlit in here, so it is easy to test.)
#
# Privacy rule: only the report text (summary + tips) is sent. Gemini wrote it from aggregated
# numbers only, so it holds no transaction descriptions or merchant names.

import base64
import io
import re
import sys
import wave

import pandas as pd

import report as rp

TTS_MODEL_NAME = "gemini-3.8-flash-lite-tts"   # model names get retired often: change it here if Gemini returns 404
VOICE = "Kore"                                  # one calm voice for all three languages

# The audio format in Google's docs: 24,000 samples per second, 16-bit (2 bytes per sample), mono
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1

FRIENDLY_ERROR = "Couldn't generate audio right now, try again in a minute"

# At most this many speech requests per browser session (counted with report.take_call)
SPEECH_LIMIT = 3
SPEECH_LIMIT_MESSAGE = (f"You've reached the limit of {SPEECH_LIMIT} audio requests for this session. "
                        "Audio you already made can still be played and downloaded.")

# For each report language: the language code and how the voice should speak
VOICE_SETTINGS = {
    "en": {
        "language": "en-IN",
        "style": "Speak Indian English in a warm, clear voice at a moderate pace. "
                 'Read rupee amounts naturally: "₹4,800" is "four thousand eight hundred rupees".',
    },
    "hi": {
        "language": "hi-IN",
        "style": "Speak natural Hindi in a calm voice at a natural pace. "
                 'Read rupee amounts naturally in Hindi: "₹4,800" is "चार हज़ार आठ सौ रुपये".',
    },
    "hinglish": {
        "language": "en-IN",
        "style": "Speak with an Indian accent in a calm, casual tone. The text is Hinglish: Hindi and English "
                 "mixed, written in Roman script. Pronounce the Hindi words as Hindi, the way people talk. "
                 'Read rupee amounts naturally: "₹4,800" is "chaar hazaar aath sau rupaye".',
    },
}


class TTSError(Exception):
    """A problem we can show to the user as-is (message is already friendly)."""


def audio_cache_key(month, language):
    """The name under which the app saves the audio: one for each month and language."""
    return f"audio_{month}_{language}"


def audio_filename(month, language):
    """month = any date inside it -> 'expense-report-2026-05-hi.wav'."""
    return f"expense-report-{pd.Timestamp(month):%Y-%m}-{language}.wav"


def report_text(report):
    """The words to read out: the summary, then each tip on its own line."""
    return "\n".join([report["summary"], *report["tips"]]).strip()


def build_request(text, language):
    """Everything sent to the TTS model. The style instruction is kept apart from the text, so the voice
    follows it without reading it out."""
    settings = VOICE_SETTINGS[language]
    return {
        "model": TTS_MODEL_NAME,
        "input": [{"type": "user_input", "content": [{
            "type": "text", "text": text,
            "annotations": [{"type": "speech_metadata", "style": settings["style"]}],
        }]}],
        "response_format": {"type": "audio", "mime_type": "audio/l16"},      # l16 = raw 16-bit PCM
        "generation_config": {"speech_config": [{"voice": VOICE, "language": settings["language"]}]},
    }


def pcm_to_wav(pcm, sample_rate=SAMPLE_RATE, channels=CHANNELS, sample_width=SAMPLE_WIDTH):
    """Raw PCM is only the sound samples. A WAV file is the same samples with a header in front that
    tells the player how to read them. Built in memory: nothing is written to disk."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(sample_width)
        f.setframerate(sample_rate)
        f.writeframes(pcm)
    return buffer.getvalue()


def _ask_tts(request, api_key):
    """Send the request to Gemini and return the audio object: .data (base64 text), .mime_type,
    .sample_rate, .channels. Imported here so tests never need the package."""
    from google import genai
    from google.genai import types
    # the TTS models are only found on API version v1beta; speech takes longer than text, so wait up to 90 s
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta", timeout=90_000))
    return client.interactions.create(**request).output_audio


def _read_audio(audio):
    """(pcm bytes, sample rate, channels) from the reply. The reply's own values are used when it has
    them; otherwise the ones written in its type ('audio/l16; rate=24000'), then the documented ones."""
    mime = (audio.mime_type or "").lower()
    if not ("l16" in mime or "pcm" in mime):
        raise ValueError(f"expected raw PCM audio, got '{mime}'")
    pcm = base64.b64decode(audio.data)
    if not pcm:
        raise ValueError("the reply had no audio")
    in_type = re.search(r"rate=(\d+)", mime)
    rate = getattr(audio, "sample_rate", None) or (int(in_type.group(1)) if in_type else SAMPLE_RATE)
    return pcm, rate, getattr(audio, "channels", None) or CHANNELS


def generate_audio(report, language, ask=None):
    """Turn a report ({'summary': str, 'tips': [str]}) into {'wav': bytes, 'seconds': float}.
    ask(request, api_key) -> audio object can be swapped out in tests. Raises TTSError."""
    api_key = rp.get_api_key()
    if not api_key:
        print("[tts] No Gemini API key found.", file=sys.stderr)
        raise TTSError(FRIENDLY_ERROR)
    ask = ask or _ask_tts
    try:
        pcm, rate, channels = _read_audio(ask(build_request(report_text(report), language), api_key))
        wav = pcm_to_wav(pcm, rate, channels)
    except Exception as error:
        # Print the REAL cause in the server console (key removed) so it can be debugged
        print(f"[tts] Gemini speech failed: {type(error).__name__}: {str(error).replace(api_key, '<key>')[:300]}",
              file=sys.stderr)
        raise TTSError(FRIENDLY_ERROR)
    return {"wav": wav, "seconds": len(pcm) / (rate * channels * SAMPLE_WIDTH)}
