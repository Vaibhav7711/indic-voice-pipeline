"""Explicit Whisper ASR runtime.

Owns the full loop: mel → encoder → autoregressive decoder with dual KV cache.
``model.generate()`` and ``pipeline("asr")`` are never in the execution path.

Exports are resolved lazily (PEP 562). Importing a pure-numpy submodule such as
``asr.explicit.chunking`` must not drag in torch: eager re-exports here made
``from asr.explicit.chunking import chunk_audio`` fail at collection time on any
machine without CUDA/torch installed, which took ``tests/test_chunking.py`` with
it. The public names below still resolve exactly as before.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from asr.explicit.loader import LoadedWhisper, load_whisper
    from asr.explicit.runner import (
        ASRMetrics,
        ASRResult,
        ASRRunner,
        LongFormASRResult,
        LongFormMetrics,
    )

_LAZY = {
    "LoadedWhisper": "asr.explicit.loader",
    "load_whisper": "asr.explicit.loader",
    "ASRRunner": "asr.explicit.runner",
    "ASRResult": "asr.explicit.runner",
    "ASRMetrics": "asr.explicit.runner",
    "LongFormASRResult": "asr.explicit.runner",
    "LongFormMetrics": "asr.explicit.runner",
}

__all__ = [
    "LoadedWhisper", "load_whisper", "ASRRunner", "ASRResult", "ASRMetrics",
    "LongFormASRResult", "LongFormMetrics",
]


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
