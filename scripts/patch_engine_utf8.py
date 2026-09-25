"""Patch the serving engine's streaming loop so Indic text survives it.

The defect and its evidence are in `docs/ENGINE_BUG_UTF8_STREAMING.md`. In
short: Qwen's byte-level BPE splits a 3-byte Devanagari character across two
tokens, so `engine/server/openai.py:_stream` decodes a half-arrived character
as U+FFFD, sends it, and can never retract it -- the corrected text is no
longer an extension of what was already sent, and its length-based delta then
skips the real character. The output is pronounced by TTS, so this is not
cosmetic.

This belongs upstream in `full-inference-engine`. Until it lands there, a
Colab session clones the engine fresh every run and needs the fix applied
before serving, which is what this does:

    python scripts/patch_engine_utf8.py --engine-root ../full-inference-engine
    python scripts/patch_engine_utf8.py --engine-root ... --check   # report only

It is **idempotent** -- running it twice reports `already-patched` and changes
nothing -- because a Colab notebook gets re-run, and a patcher that appended
its change every time would corrupt the file it is fixing. It refuses rather
than guessing if the target text is not found, which is what happens when the
engine fixes this itself: the right response then is to delete this script,
not to force a patch into changed code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("engine/server/openai.py")

#: The exact text as of full-inference-engine v0.1.0-beta. Matched literally,
#: indentation included: a fuzzy match on code this delicate is how a patcher
#: silently applies a fix to the wrong place.
ORIGINAL = """                trimmed, hit = truncate_at_stop(text, stops) if stops else (text, False)
                if len(trimmed) > len(sent_text):
                    delta = trimmed[len(sent_text):]
                    sent_text = trimmed
                    yield f"data: {json.dumps(chunk(delta, None))}\\n\\n"
"""

REPLACEMENT = '''                trimmed, hit = truncate_at_stop(text, stops) if stops else (text, False)
                # A trailing U+FFFD is a multi-byte character that has only
                # partly arrived: byte-level BPE splits a 3-byte Devanagari
                # character across two tokens. Emitting it is unrecoverable --
                # the next decode replaces it in place, so the text already
                # sent stops being a prefix of the corrected text and the real
                # character is skipped by the slice below. Holding it back
                # costs one poll of latency and keeps the stream a pure append.
                stable = trimmed[:-1] if trimmed.endswith("\\ufffd") else trimmed
                # startswith is a guard, not an assertion: with the hold-back
                # above it should always hold, and if it ever does not,
                # skipping a poll beats sending a delta cut at a wrong offset.
                if stable.startswith(sent_text) and len(stable) > len(sent_text):
                    delta = stable[len(sent_text):]
                    sent_text = stable
                    yield f"data: {json.dumps(chunk(delta, None))}\\n\\n"
'''

#: Present only after patching, so detection does not depend on the diff.
MARKER = 'stable = trimmed[:-1] if trimmed.endswith("\\ufffd") else trimmed'


def patch_text(source: str) -> tuple[str, str]:
    """Return (new_source, status).

    Status is `patched`, `already-patched`, or `not-found`. `already-patched`
    is checked first: the original text is *contained* in the replacement, so
    testing for the original first would re-apply the patch forever.
    """
    if MARKER in source:
        return source, "already-patched"
    if ORIGINAL not in source:
        return source, "not-found"
    return source.replace(ORIGINAL, REPLACEMENT, 1), "patched"


def check_compiles(source: str, filename: str) -> None:
    """Fail loudly here rather than at uvicorn startup."""
    compile(source, filename, "exec")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine-root", required=True,
                        help="the full-inference-engine checkout to patch")
    parser.add_argument("--check", action="store_true",
                        help="report the status and change nothing")
    args = parser.parse_args(argv)

    path = Path(args.engine_root).expanduser().resolve() / TARGET
    if not path.is_file():
        print(f"error: {path} does not exist. Is --engine-root a "
              f"full-inference-engine checkout?", file=sys.stderr)
        return 2

    source = path.read_text(encoding="utf-8")
    new_source, status = patch_text(source)

    if status == "already-patched":
        print(f"already-patched: {path}")
        print("The hold-back is present; streamed Devanagari will survive.")
        return 0

    if status == "not-found":
        print(f"not-found: the target text is not in {path}", file=sys.stderr)
        print("The engine's _stream has changed. Check whether it fixed this "
              "itself -- if the delta is no longer computed as a slice by "
              "length, delete this script rather than forcing the patch.",
              file=sys.stderr)
        return 1

    if args.check:
        print(f"unpatched: {path}")
        print("Streamed Devanagari will be corrupted. Run without --check to "
              "apply the fix.")
        return 1

    check_compiles(new_source, str(path))
    path.write_text(new_source, encoding="utf-8")
    print(f"patched: {path}")
    print("Restart the server for it to take effect -- uvicorn imported the "
          "old module.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
