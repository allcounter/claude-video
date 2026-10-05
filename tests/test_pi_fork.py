"""Pi fork behaviour: mlx-whisper first in auto, and Gemini by default for YouTube URLs only."""
from __future__ import annotations

import sys
import types

import pytest

import config
import watch
import whisper
from transcribe import Segments

# Captured at collection, before conftest stubs local_available for every test.
REAL_LOCAL_AVAILABLE = whisper.local_available


# --- mlx-local first -------------------------------------------------------

def _fake_mlx(monkeypatch, segments=None, error=None):
    calls = []
    module = types.ModuleType("mlx_whisper")
    def transcribe(path, path_or_hf_repo=None):
        calls.append((path, path_or_hf_repo))
        if error:
            raise error
        return {"segments": segments if segments is not None else
                [{"start": 0.0, "end": 1.0, "text": " hello "}, {"start": 1.0, "end": 2.0, "text": "  "}]}
    module.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setattr(whisper, "local_available", lambda: True)
    monkeypatch.setattr(whisper, "extract_audio", lambda video, out: out)
    return calls


def _no_cloud(monkeypatch):
    monkeypatch.setattr(whisper, "_transcribe_file", lambda *a, **kw: pytest.fail("cloud must not be called"))


def test_auto_prefers_mlx_local_and_drops_blank_segments(monkeypatch, tmp_path):
    calls = _fake_mlx(monkeypatch)
    _no_cloud(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "dummy")  # A cloud key must not outrank the free local path.
    segments, used = whisper.transcribe_video("video.mp4", tmp_path / "audio.mp3")
    assert used == "mlx-local"
    assert segments == [{"start": 0.0, "end": 1.0, "text": "hello"}]
    assert calls == [(str(tmp_path / "audio.mp3"), whisper.LOCAL_MODEL)]


def test_mlx_empty_result_is_no_speech_not_a_cloud_upload(monkeypatch, tmp_path):
    _fake_mlx(monkeypatch, segments=[])
    _no_cloud(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    segments, used = whisper.transcribe_video("video.mp4", tmp_path / "audio.mp3")
    assert used == "mlx-local" and segments == [] and segments.no_speech


def test_mlx_failure_falls_back_to_cloud(monkeypatch, tmp_path):
    _fake_mlx(monkeypatch, error=RuntimeError("metal unavailable"))
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    monkeypatch.setattr(whisper, "_transcribe_file", lambda backend, key, path: [{"start": 0.0, "end": 1.0, "text": "cloud"}])
    (tmp_path / "audio.mp3").write_bytes(b"x")
    segments, used = whisper.transcribe_video("video.mp4", tmp_path / "audio.mp3")
    assert used == "groq" and segments[0]["text"] == "cloud"


def test_explicit_cloud_backend_skips_mlx(monkeypatch, tmp_path):
    calls = _fake_mlx(monkeypatch)
    monkeypatch.setattr(whisper, "_transcribe_file", lambda backend, key, path: [{"start": 0.0, "end": 1.0, "text": "cloud"}])
    (tmp_path / "audio.mp3").write_bytes(b"x")
    _, used = whisper.transcribe_video("video.mp4", tmp_path / "audio.mp3", backend="groq", api_key="dummy")
    assert used == "groq" and calls == []


def test_local_available_is_false_without_mlx(monkeypatch):
    monkeypatch.delitem(sys.modules, "mlx_whisper", raising=False)
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert REAL_LOCAL_AVAILABLE() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert REAL_LOCAL_AVAILABLE() is True


def _mock_audio(monkeypatch):
    monkeypatch.setattr(watch, "download", lambda *a, **kw: {"video_path": "video.mp4"})
    monkeypatch.setattr(watch, "get_metadata", lambda *a: {"duration_seconds": 30, "has_audio": True, "has_video": False})


def _record_transcribe(monkeypatch, seen, used="mlx-local"):
    def fake(video, audio, backend=None, api_key=None):
        seen.update(backend=backend, api_key=api_key)
        return Segments([{"start": 1, "end": 2, "text": "spoken"}]), used
    monkeypatch.setattr(watch, "transcribe_video", fake)


@pytest.mark.parametrize("cloud_key", [True, False])
def test_watch_auto_routes_to_mlx_first(monkeypatch, capsys, cloud_key):
    _mock_audio(monkeypatch)
    monkeypatch.setattr(watch, "local_whisper_available", lambda: True)
    monkeypatch.setattr(watch, "load_api_key", lambda *a: ("groq", "dummy") if cloud_key else (None, None))
    seen = {}
    _record_transcribe(monkeypatch, seen)
    monkeypatch.setattr(sys, "argv", ["watch", "video.mp4", "--detail", "transcript"])
    assert watch.main() == 0
    assert seen == {"backend": None, "api_key": None}
    assert "via whisper (mlx-local)" in capsys.readouterr().out


def test_watch_explicit_whisper_choice_skips_mlx(monkeypatch):
    _mock_audio(monkeypatch)
    monkeypatch.setattr(watch, "local_whisper_available", lambda: True)
    monkeypatch.setattr(watch, "load_api_key", lambda *a: ("groq", "dummy"))
    seen = {}
    _record_transcribe(monkeypatch, seen, used="groq")
    monkeypatch.setattr(sys, "argv", ["watch", "video.mp4", "--detail", "transcript", "--whisper", "groq"])
    assert watch.main() == 0
    assert seen == {"backend": "groq", "api_key": "dummy"}


def test_watch_no_whisper_skips_mlx(monkeypatch):
    _mock_audio(monkeypatch)
    monkeypatch.setattr(watch, "local_whisper_available", lambda: True)
    monkeypatch.setattr(watch, "transcribe_video", lambda *a, **kw: pytest.fail("--no-whisper disables mlx-local too"))
    monkeypatch.setattr(sys, "argv", ["watch", "video.mp4", "--detail", "transcript", "--no-whisper"])
    assert watch.main() == 0


# --- engine default: auto = Gemini for YouTube URLs only --------------------

def test_engine_default_is_auto_and_local_for_non_youtube(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unit-key")
    assert config.get_config()["engine"] == "auto"
    assert config.resolve_engine("auto", has_key=True) == "local"
    assert config.resolve_engine("auto", has_key=True, youtube=True) == "gemini"
    assert config.resolve_engine("local", has_key=True) == "local"
    assert config.resolve_engine("gemini", has_key=True) == "gemini"


@pytest.mark.parametrize("engine_env", [None, "auto", "local"])
def test_local_file_never_contacts_google_without_explicit_gemini(monkeypatch, static_clip, capsys, engine_env):
    import gemini
    monkeypatch.setenv("GEMINI_API_KEY", "unit-key")
    if engine_env:
        monkeypatch.setenv("WATCH_ENGINE", engine_env)
    for name in ("ask", "upload_file"):
        monkeypatch.setattr(gemini, name, lambda *a, **kw: pytest.fail("a local file must never reach Gemini under auto or local"))
    monkeypatch.setattr(sys, "argv", ["watch", str(static_clip), "--no-whisper"])
    assert watch.main() == 0
    assert "## Frames" in capsys.readouterr().out


@pytest.mark.parametrize("how", ["flag", "env"])
def test_explicit_gemini_still_works(monkeypatch, capsys, how):
    import gemini
    monkeypatch.setenv("GEMINI_API_KEY", "unit-key")
    monkeypatch.setattr(gemini, "ask", lambda video, q, **kw: {"text": "ok", "model": kw["model"], "processing": "agentic", "total_tokens": 1})
    argv = ["watch", "https://youtu.be/abc"]
    if how == "flag":
        argv += ["--engine", "gemini"]
    else:
        monkeypatch.setenv("WATCH_ENGINE", "gemini")
    monkeypatch.setattr(sys, "argv", argv)
    assert watch.main() == 0
    assert "## Answer (from Gemini)" in capsys.readouterr().out


def test_setup_status_reports_mlx_local_in_auto(monkeypatch):
    import setup
    monkeypatch.setattr(whisper, "local_available", lambda: True)
    status = setup._status()
    assert status["configured_backend"] == "auto"
    assert status["whisper_backend"] == "mlx-local" and status["backend_ready"] is True
    assert status["mlx_local_available"] is True
    assert status["engine"] == "local"


@pytest.mark.parametrize("engine_env", [None, "auto"])
@pytest.mark.parametrize("url", ["https://www.youtube.com/watch?v=abc123", "https://youtu.be/abc123"])
def test_youtube_url_goes_to_gemini_by_default(monkeypatch, capsys, engine_env, url):
    import gemini
    monkeypatch.setenv("GEMINI_API_KEY", "unit-key")
    if engine_env:
        monkeypatch.setenv("WATCH_ENGINE", engine_env)
    seen = {}
    def ask(video, q, **kw):
        seen["video"] = video
        return {"text": "ok", "model": kw["model"], "processing": "agentic", "total_tokens": 1}
    monkeypatch.setattr(gemini, "ask", ask)
    monkeypatch.setattr(gemini, "upload_file", lambda *a, **kw: pytest.fail("a YouTube URL is never uploaded"))
    monkeypatch.setattr(sys, "argv", ["watch", url, "--question", "what is shown"])
    assert watch.main() == 0
    assert seen["video"] == {"uri": url}
    assert "## Answer (from Gemini)" in capsys.readouterr().out


@pytest.mark.parametrize("argv_extra", [["--engine", "local"], ["--detail", "transcript"], ["--no-whisper"], ["--timestamps", "0:01"]])
def test_youtube_url_stays_local_when_asked(monkeypatch, argv_extra):
    import gemini
    monkeypatch.setenv("GEMINI_API_KEY", "unit-key")
    for name in ("ask", "upload_file"):
        monkeypatch.setattr(gemini, name, lambda *a, **kw: pytest.fail("local was asked for"))
    monkeypatch.setattr(watch, "resolve_engine", _spy_engine := (lambda *a, **kw: _record(a, kw)))
    monkeypatch.setattr(sys, "argv", ["watch", "https://youtu.be/abc123", *argv_extra])
    with pytest.raises(_Stop):
        watch.main()
    assert _LAST["engine"] == "local"


def test_youtube_url_without_key_stays_local(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(config, "load_gemini_key", lambda: None)
    monkeypatch.setattr(watch, "load_gemini_key", lambda: None)
    monkeypatch.setattr(watch, "resolve_engine", lambda *a, **kw: _record(a, kw))
    monkeypatch.setattr(sys, "argv", ["watch", "https://youtu.be/abc123"])
    with pytest.raises(_Stop):
        watch.main()
    assert _LAST["engine"] == "local"


@pytest.mark.parametrize("url", ["https://x.com/a/status/1", "https://vimeo.com/123", "https://example.com/v.mp4"])
def test_other_urls_stay_local_under_auto(monkeypatch, url):
    monkeypatch.setenv("GEMINI_API_KEY", "unit-key")
    monkeypatch.setattr(watch, "resolve_engine", lambda *a, **kw: _record(a, kw))
    monkeypatch.setattr(sys, "argv", ["watch", url])
    with pytest.raises(_Stop):
        watch.main()
    assert _LAST["engine"] == "local"


class _Stop(Exception):
    pass


_LAST = {}


def _record(a, kw):
    """Run the real resolver, remember its verdict, and stop before any media work."""
    _LAST["engine"] = config.resolve_engine(*a, **kw)
    raise _Stop()
