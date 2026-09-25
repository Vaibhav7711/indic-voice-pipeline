"""The engine patcher: it must match the real file, and be safe to re-run.

Two failure modes matter. A patcher that does not match leaves the corruption
in place while reporting success. A patcher that is not idempotent corrupts
the file it is fixing on the second Colab run — and a Colab notebook always
gets re-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.patch_engine_utf8 import (
    MARKER,
    ORIGINAL,
    REPLACEMENT,
    check_compiles,
    main,
    patch_text,
)

#: The surrounding lines from engine/server/openai.py, at the real
#: indentation: `_stream` is nested inside `install_openai_routes`, so its
#: loop body is at sixteen spaces. The patch matches literally, so a fixture
#: at the wrong depth would test nothing that exists.
ENGINE_SOURCE = '''import json


def install_openai_routes(app, *, require_service, model_name,
                          max_prompt_tokens, request_timeout_s):

    async def _stream(service, handle, stops, *, chunk, request,
                      prompt_tokens, include_usage):
        import asyncio
        tokenizer = service.engine.tokenizer
        loop = asyncio.get_running_loop()
        started = loop.time()
        sent_text = ""
        finish = None
        try:
            while finish is None:
                if await request.is_disconnected():
                    service.cancel(handle)
                    return
                done = handle.completed.is_set()
                text = tokenizer.decode(handle.request.output_token_ids, skip_special_tokens=True)
''' + ORIGINAL + '''                if hit:
                    service.cancel(handle, reason="STOP_SEQUENCE")
                    finish = "stop"
                    break
                if done:
                    finish = "stop"
                    break
                await asyncio.sleep(0.005)
            yield f"data: {json.dumps(chunk('', finish))}\\n\\n"
        finally:
            pass

    return _stream
'''


class TestPatching:
    def test_it_patches_the_engine_source(self):
        patched, status = patch_text(ENGINE_SOURCE)
        assert status == "patched"
        assert MARKER in patched
        assert "delta = stable[len(sent_text):]" in patched
        assert "delta = trimmed[len(sent_text):]" not in patched

    def test_the_result_is_valid_python(self):
        """Caught here rather than at uvicorn startup, where the traceback
        would be about a syntax error in someone else's repository."""
        patched, _ = patch_text(ENGINE_SOURCE)
        check_compiles(patched, "openai.py")

    def test_the_stop_string_branch_is_preserved(self):
        """`hit` comes from the same line the patch replaces; losing it would
        break stop-sequence handling while fixing the encoding."""
        patched, _ = patch_text(ENGINE_SOURCE)
        assert "trimmed, hit = truncate_at_stop(" in patched
        assert "if hit:" in patched

    def test_the_guard_and_the_holdback_are_both_present(self):
        patched, _ = patch_text(ENGINE_SOURCE)
        assert "stable.startswith(sent_text)" in patched
        assert 'trimmed.endswith("\\ufffd")' in patched


class TestIdempotence:
    def test_patching_twice_changes_nothing(self):
        once, first = patch_text(ENGINE_SOURCE)
        twice, second = patch_text(once)
        assert first == "patched"
        assert second == "already-patched"
        assert twice == once

    def test_the_marker_is_checked_before_the_original(self):
        """The original text is contained inside the replacement, so testing
        for the original first would re-apply the patch on every run and nest
        the hold-back inside itself."""
        assert ORIGINAL.strip().splitlines()[0] in REPLACEMENT
        _, status = patch_text(REPLACEMENT)
        assert status == "already-patched"

    def test_an_already_patched_file_still_compiles(self):
        once, _ = patch_text(ENGINE_SOURCE)
        twice, _ = patch_text(once)
        check_compiles(twice, "openai.py")


class TestRefusal:
    def test_unrelated_source_is_not_found_rather_than_mangled(self):
        source = "def stream():\n    return 1\n"
        result, status = patch_text(source)
        assert status == "not-found"
        assert result == source

    def test_a_changed_delta_computation_is_not_found(self):
        """What the engine fixing this itself looks like. The right response is
        to delete the patcher, not to force a patch into changed code."""
        upstream_fixed = ENGINE_SOURCE.replace(
            "if len(trimmed) > len(sent_text):",
            "if trimmed.startswith(sent_text) and len(trimmed) > len(sent_text):")
        _, status = patch_text(upstream_fixed)
        assert status == "not-found"


class TestCli:
    def _engine(self, tmp_path: Path, source: str = ENGINE_SOURCE) -> Path:
        target = tmp_path / "engine" / "server"
        target.mkdir(parents=True)
        (target / "openai.py").write_text(source, encoding="utf-8")
        return tmp_path

    def test_it_patches_a_checkout_and_reports_zero(self, tmp_path, capsys):
        root = self._engine(tmp_path)
        assert main(["--engine-root", str(root)]) == 0
        assert "patched:" in capsys.readouterr().out
        patched = (root / "engine/server/openai.py").read_text(encoding="utf-8")
        assert MARKER in patched

    def test_a_second_run_is_a_no_op(self, tmp_path, capsys):
        root = self._engine(tmp_path)
        main(["--engine-root", str(root)])
        before = (root / "engine/server/openai.py").read_text(encoding="utf-8")
        assert main(["--engine-root", str(root)]) == 0
        assert "already-patched" in capsys.readouterr().out
        assert (root / "engine/server/openai.py").read_text(encoding="utf-8") == before

    def test_check_reports_unpatched_without_writing(self, tmp_path, capsys):
        root = self._engine(tmp_path)
        assert main(["--engine-root", str(root), "--check"]) == 1
        assert "unpatched" in capsys.readouterr().out
        assert MARKER not in (root / "engine/server/openai.py").read_text(
            encoding="utf-8")

    def test_check_reports_zero_once_patched(self, tmp_path):
        root = self._engine(tmp_path)
        main(["--engine-root", str(root)])
        assert main(["--engine-root", str(root), "--check"]) == 0

    def test_a_missing_checkout_is_an_error_not_a_crash(self, tmp_path, capsys):
        assert main(["--engine-root", str(tmp_path / "nowhere")]) == 2
        assert "does not exist" in capsys.readouterr().err

    def test_unrecognised_source_exits_nonzero(self, tmp_path, capsys):
        root = self._engine(tmp_path, source="print('hello')\n")
        assert main(["--engine-root", str(root)]) == 1
        assert "not-found" in capsys.readouterr().err


def test_the_target_text_is_what_the_engine_actually_contains():
    """A reminder in test form: `ORIGINAL` is a literal copy of upstream at
    v0.1.0-beta. If the engine is checked out beside this repo, verify against
    it; otherwise this documents where the text came from."""
    beside = Path(__file__).resolve().parents[2] / "full-inference-engine"
    source = beside / "engine/server/openai.py"
    if not source.is_file():
        pytest.skip("full-inference-engine is not checked out beside this repo")
    text = source.read_text(encoding="utf-8")
    assert ORIGINAL in text or MARKER in text, (
        "neither the original nor the patched text is present; the engine's "
        "_stream has changed and scripts/patch_engine_utf8.py needs review"
    )
