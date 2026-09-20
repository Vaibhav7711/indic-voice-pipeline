"""Verify GPU, CUDA, and all imports before running anything."""

import sys


def main():
    errors = []

    # GPU check.
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available — select a GPU runtime.")
        else:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {name} ({vram:.1f} GiB)")
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA: {torch.version.cuda}")
    except ImportError:
        errors.append("PyTorch not found.")

    # Key imports.
    for mod, desc in [
        ("transformers", "HuggingFace Transformers"),
        ("peft", "PEFT (LoRA)"),
        ("datasets", "HuggingFace Datasets"),
        ("librosa", "librosa (audio)"),
        ("soundfile", "soundfile (WAV I/O)"),
        ("evaluate", "evaluate (WER)"),
    ]:
        try:
            __import__(mod)
            print(f"  {desc}: OK")
        except ImportError:
            errors.append(f"{desc} ({mod}) not installed.")

    # Project imports.
    try:
        from asr.explicit import ASRRunner, load_whisper  # noqa: F401
        from llm import LLMRunner, load_llm  # noqa: F401
        from pipeline import VoicePipeline  # noqa: F401
        from tts import TTSSynthesizer  # noqa: F401
        print("  Project modules: OK")
    except ImportError as e:
        errors.append(f"Project import failed: {e}")

    if errors:
        print(f"\n{'='*50}")
        print("PREFLIGHT FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"\n{'='*50}")
        print("All preflight checks passed.")


if __name__ == "__main__":
    main()
