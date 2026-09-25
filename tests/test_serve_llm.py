"""The serving launcher: readiness, not liveness, and failures that fail fast.

Warmup is the whole reason this script exists. An engine that captures CUDA
graphs and JITs Triton kernels answers `503` on `/ready` for tens of seconds
on a cold cache, and a pipeline that starts measuring immediately records a
first turn seconds slower than the steady state. These tests pin the two ways
that goes wrong: declaring ready too early, and waiting on a process that has
already died.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from scripts.serve_llm import DEFAULT_APP, main, server_command, wait_until_ready


class FakeProcess:
    """Stands in for `Popen`: `poll()` returns None until it is made to exit."""

    def __init__(self, exit_after: int | None = None, returncode: int = 1):
        self.exit_after = exit_after
        self.returncode = returncode
        self.polls = 0

    def poll(self):
        self.polls += 1
        if self.exit_after is not None and self.polls > self.exit_after:
            return self.returncode
        return None


def responses(sequence):
    """Patch `probe` to return each (status, body) in turn, then repeat the last."""
    remaining = list(sequence)

    def probe(url, timeout=2.0):
        probe.urls.append(url)
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    probe.urls = []
    return probe


class TestReadiness:
    def test_it_waits_through_503_and_returns_the_ready_body(self, monkeypatch):
        probe = responses([
            (0, "URLError: connection refused"),      # process not listening yet
            (503, {"status": "not_ready", "loaded": False}),
            (503, {"status": "not_ready", "loaded": True}),   # warming up
            (200, {"status": "ready", "active": 0}),
        ])
        monkeypatch.setattr("scripts.serve_llm.probe", probe)
        monkeypatch.setattr("scripts.serve_llm.time.sleep", lambda _: None)
        body = wait_until_ready("http://127.0.0.1:8000", timeout_s=30, log=lambda *_: None)
        assert body == {"status": "ready", "active": 0}

    def test_it_polls_readiness_not_liveness(self, monkeypatch):
        """`/health` says the worker is alive, which it is throughout warmup."""
        probe = responses([(200, {"status": "ready"})])
        monkeypatch.setattr("scripts.serve_llm.probe", probe)
        wait_until_ready("http://127.0.0.1:8000/", timeout_s=5, log=lambda *_: None)
        assert probe.urls == ["http://127.0.0.1:8000/ready"]

    def test_timeout_reports_the_last_status_and_body(self, monkeypatch):
        probe = responses([(503, {"status": "not_ready", "loaded": False})])
        monkeypatch.setattr("scripts.serve_llm.probe", probe)
        monkeypatch.setattr("scripts.serve_llm.time.sleep", lambda _: None)
        with pytest.raises(RuntimeError, match="not_ready"):
            wait_until_ready("http://127.0.0.1:8000", timeout_s=2, log=lambda *_: None)

    def test_a_server_that_exits_during_warmup_fails_immediately(self, monkeypatch):
        """This engine refuses geometry its paged kernels cannot serve, at load,
        with the reason. Polling a dead process until a 600-second deadline
        would bury that refusal under ten minutes of silence.
        """
        probe = responses([(0, "URLError: connection refused")])
        monkeypatch.setattr("scripts.serve_llm.probe", probe)
        monkeypatch.setattr("scripts.serve_llm.time.sleep", lambda _: None)
        process = FakeProcess(exit_after=2, returncode=3)
        with pytest.raises(RuntimeError, match="exited with code 3"):
            wait_until_ready("http://127.0.0.1:8000", timeout_s=600,
                             process=process, log=lambda *_: None)
        assert process.polls <= 4, "it did not wait out the deadline"

    def test_a_ready_server_is_accepted_even_with_a_live_process(self, monkeypatch):
        probe = responses([(200, {"status": "ready"})])
        monkeypatch.setattr("scripts.serve_llm.probe", probe)
        body = wait_until_ready("http://127.0.0.1:8000", timeout_s=5,
                                process=FakeProcess(), log=lambda *_: None)
        assert body["status"] == "ready"


class TestServerCommand:
    def test_it_uses_the_factory_flag(self):
        """The measured profiles are zero-argument factories on purpose: a
        configuration that was A/B-ed on a device cannot then drift."""
        command = server_command("engine.server.api:create_rtx4060_flash_app",
                                 port=8000, host="127.0.0.1", extra=[])
        assert command[:3] == [sys.executable, "-m", "uvicorn"]
        assert "--factory" in command
        assert "engine.server.api:create_rtx4060_flash_app" in command
        assert command[command.index("--port") + 1] == "8000"

    def test_extra_arguments_are_passed_through_last(self):
        command = server_command("app:factory", port=9000, host="0.0.0.0",
                                 extra=["--log-level", "warning"])
        assert command[-2:] == ["--log-level", "warning"]


def test_the_script_runs_without_the_engine_installed():
    """`--help` must not import the engine: the launcher is checked into this
    repo, the engine is a separate checkout, and CI has neither."""
    result = subprocess.run([sys.executable, "scripts/serve_llm.py", "--help"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "--engine-root" in result.stdout and "--ready-timeout" in result.stdout


class TestFactoryFlagGuard:
    """The engine's own profiles are zero-argument on purpose: a configuration
    A/B-ed on an architecture cannot then drift. So passing --model alongside
    one has to be refused, not ignored -- a recorded configuration that
    disagrees with the served one is the failure this launcher exists to
    prevent."""

    def test_model_with_a_zero_argument_profile_is_refused(self, capsys):
        with pytest.raises(SystemExit):
            main(["--app", "engine.server.api:create_rtx4060_flash_app",
                  "--model", "Qwen/Qwen3-4B"])
        assert "cannot receive" in capsys.readouterr().err

    def test_the_refusal_names_the_factory_that_would_work(self, capsys):
        with pytest.raises(SystemExit):
            main(["--app", "engine.server.api:create_app", "--num-blocks", "512"])
        assert DEFAULT_APP in capsys.readouterr().err

    def test_a_zero_argument_profile_alone_is_accepted(self, monkeypatch):
        """No pool flags, so nothing to disagree about. It should get as far as
        trying to start a server, which is where this test stops it."""
        started = {}

        def fake_popen(command, env=None):
            started["command"] = command
            raise KeyboardInterrupt        # stop before a real uvicorn runs

        monkeypatch.setattr("scripts.serve_llm.subprocess.Popen", fake_popen)
        with pytest.raises(KeyboardInterrupt):
            main(["--app", "engine.server.api:create_rtx4060_flash_app"])
        assert "engine.server.api:create_rtx4060_flash_app" in started["command"]

    def test_the_default_factory_accepts_the_pool_flags(self, monkeypatch):
        captured = {}

        def fake_popen(command, env=None):
            captured["env"] = env
            raise KeyboardInterrupt

        monkeypatch.setattr("scripts.serve_llm.subprocess.Popen", fake_popen)
        with pytest.raises(KeyboardInterrupt):
            main(["--model", "Qwen/Qwen3-4B", "--num-blocks", "512",
                  "--graph-buckets", "1,2"])
        assert captured["env"]["LLM_SERVER_MODEL"] == "Qwen/Qwen3-4B"
        assert captured["env"]["LLM_SERVER_NUM_BLOCKS"] == "512"
        assert captured["env"]["LLM_SERVER_GRAPH_BUCKETS"] == "1,2"
