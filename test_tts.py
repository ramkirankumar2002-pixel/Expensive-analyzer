# test_tts.py - small tests for tts.py (plain asserts). No network: Gemini is replaced by a fake.
# Run with:  python test_tts.py

import base64
import io
import os
import types
import wave

import pandas as pd

import report as rp
import tts

REPORT = {"summary": "इस महीने आपने Food पर ₹4,800 खर्च किए।", "tips": ["घर पर खाना बनाइए।", "Bills जाँचिए।"]}
PCM = b"\x01\x00\x02\x00" * 12000            # 48,000 bytes = 24,000 samples = exactly 1 second


def use_fake_key():
    os.environ["GEMINI_API_KEY"] = "fake-key"
    rp.load_dotenv = lambda *a, **k: None    # don't let a real .env file change the key


def reply(pcm=PCM, mime="audio/l16", **extra):
    """What the real API returns: base64 text plus a description of the audio."""
    return types.SimpleNamespace(data=base64.b64encode(pcm).decode(), mime_type=mime, **extra)


def test_language_and_voice_go_into_the_request():
    use_fake_key()
    expected = {"en": ("en-IN", "Indian English", "four thousand eight hundred rupees"),
                "hi": ("hi-IN", "natural Hindi", "चार हज़ार आठ सौ रुपये"),
                "hinglish": ("en-IN", "casual tone", "Roman script")}
    assert set(tts.VOICE_SETTINGS) == set(rp.LANGUAGES)            # every report language has a voice
    assert tts.VOICE == "Kore"
    for language, (code, style_words, example) in expected.items():
        seen = []

        def fake(request, key):
            seen.append(request)
            return reply()
        tts.generate_audio(REPORT, language, ask=fake)
        request = seen[0]
        assert request["model"] == tts.TTS_MODEL_NAME
        assert request["generation_config"]["speech_config"] == [{"voice": "Kore", "language": code}]
        assert request["response_format"] == {"type": "audio", "mime_type": "audio/l16"}
        content = request["input"][0]["content"][0]
        style = content["annotations"][0]["style"]
        assert style_words in style and example in style, language


def test_only_the_report_text_is_sent():
    text = tts.build_request(tts.report_text({**REPORT, "language": "hi", "extra": "Swiggy"}), "hi")
    sent = text["input"][0]["content"][0]["text"]
    assert sent == "इस महीने आपने Food पर ₹4,800 खर्च किए।\nघर पर खाना बनाइए।\nBills जाँचिए।"
    assert "Swiggy" not in str(text)
    assert tts.report_text({"summary": "Only a summary.", "tips": []}) == "Only a summary."


def test_the_sdk_accepts_the_request():
    """The installed google-genai must understand the request's fields (checked offline)."""
    from google.genai._gaos.types.interactions import speechannotation, speechconfig
    request = tts.build_request("hello", "en")
    speechconfig.SpeechConfig(**request["generation_config"]["speech_config"][0])
    speechannotation.SpeechAnnotation(**request["input"][0]["content"][0]["annotations"][0])


def test_pcm_to_wav_has_a_valid_header():
    data = tts.pcm_to_wav(PCM)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE" and data[12:16] == b"fmt "
    assert int.from_bytes(data[4:8], "little") == len(data) - 8      # the size written in the header is right
    with wave.open(io.BytesIO(data)) as f:
        assert (f.getframerate(), f.getsampwidth(), f.getnchannels()) == (24000, 2, 1)
        assert f.getnframes() == 24000 and f.readframes(24000) == PCM    # the sound is unchanged


def test_audio_result_and_format_from_the_reply():
    use_fake_key()
    out = tts.generate_audio(REPORT, "en", ask=lambda r, k: reply())
    assert out["seconds"] == 1.0 and out["wav"] == tts.pcm_to_wav(PCM)
    # the reply's own sample rate and channels win over the documented ones
    for audio in [reply(sample_rate=16000, channels=2), reply(mime="audio/L16;codec=pcm;rate=16000", channels=2)]:
        out = tts.generate_audio(REPORT, "en", ask=lambda r, k: audio)
        with wave.open(io.BytesIO(out["wav"])) as f:
            assert (f.getframerate(), f.getnchannels()) == (16000, 2)
        assert out["seconds"] == 0.75                                 # 48,000 bytes / (16000 * 2 * 2)


def test_cache_key_and_file_name():
    may, june = pd.Period("2026-05"), pd.Period("2026-06")
    keys = {tts.audio_cache_key(m, language) for m in (may, june) for language in rp.LANGUAGES}
    assert len(keys) == 6                                             # all different
    assert tts.audio_cache_key(may, "hi") == "audio_2026-05_hi"
    assert tts.audio_cache_key(may, "hi") != rp.report_cache_key(may, "hi")     # never overwrites the report
    assert tts.audio_filename(may.to_timestamp(), "hinglish") == "expense-report-2026-05-hinglish.wav"


def assert_friendly(ask):
    try:
        tts.generate_audio(REPORT, "en", ask=ask)
        assert False, "should have raised"
    except tts.TTSError as error:
        assert str(error) == "Couldn't generate audio right now, try again in a minute"


def test_errors_are_friendly():
    use_fake_key()
    for message in ["429 RESOURCE_EXHAUSTED quota", "404 model not found", "timed out", "secret details"]:
        def broken(request, key, message=message):
            raise RuntimeError(message)
        assert_friendly(broken)
    assert_friendly(lambda r, k: None)                                # no audio in the reply
    assert_friendly(lambda r, k: reply(pcm=b""))                      # empty audio
    assert_friendly(lambda r, k: reply(mime="audio/wav"))             # not the raw PCM we asked for


def test_missing_key_is_friendly_and_makes_no_call():
    use_fake_key()
    real = rp.get_api_key
    rp.get_api_key = lambda: ""
    try:
        calls = []
        assert_friendly(lambda r, k: calls.append(1))
        assert calls == []
    finally:
        rp.get_api_key = real


def test_the_key_is_never_printed():
    import contextlib
    use_fake_key()

    def broken(request, key):
        raise RuntimeError("bad key " + key)
    console = io.StringIO()
    with contextlib.redirect_stderr(console):
        assert_friendly(broken)
    assert "fake-key" not in console.getvalue() and "<key>" in console.getvalue()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all tts tests passed")
