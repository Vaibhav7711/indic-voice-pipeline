"""Choose an ASR and LLM backend from configuration, in one place.

Every entry point used to construct `ASRRunner` and `LLMRunner` directly, so
the CTranslate2 tier the project measured at 1.31× and formally adopted was
reachable only from the validation script — it never served a turn. These
factories give the demo, the notebook and the live agent one seam each.

Selection is deliberately explicit rather than "use the fast one if present":
a serving path that silently changes engine is a serving path whose numbers
cannot be compared across runs.
"""

from __future__ import annotations

__all__ = ["build_asr", "build_llm", "ASR_ENGINES", "LLM_ENGINES"]

ASR_ENGINES = ("explicit", "ct2")
LLM_ENGINES = ("explicit", "http")


def build_asr(
    engine: str = "explicit",
    *,
    model: str = "openai/whisper-large-v3-turbo",
    adapter: str | None = None,
    device: str | None = None,
    dtype=None,
    ct2_model: str | None = None,
    ct2_compute_type: str = "int8_float16",
    language_candidates: list[str] | None = None,
):
    """Return an object satisfying ``asr.streaming.StreamingTranscriber``.

    ``explicit`` is the reference implementation and stays the default: it is
    what the correctness checks compare against. ``ct2`` is the faster serving
    tier and needs a converted model directory from
    ``scripts/convert_ct2.py`` — it is not built on demand, because converting
    takes minutes and silently doing it mid-startup would hide the cost.
    """
    if engine not in ASR_ENGINES:
        raise ValueError(f"asr engine must be one of {ASR_ENGINES}, got {engine!r}")

    if engine == "ct2":
        if not ct2_model:
            raise ValueError(
                "asr engine 'ct2' needs ct2_model: a directory produced by "
                "scripts/convert_ct2.py --model <base> --adapter <adapter>",
            )
        from asr.engines.ct2 import CT2Transcriber

        resolved = device or "auto"
        return CT2Transcriber(ct2_model, device=resolved,
                              compute_type=ct2_compute_type,
                              language_candidates=language_candidates)

    from asr.explicit import ASRRunner, load_whisper

    loaded = load_whisper(model, adapter_path=adapter, device=device, dtype=dtype)
    runner = ASRRunner(loaded.model, loaded.processor, loaded.device, loaded.dtype,
                       language_candidates=language_candidates)
    runner.loaded = loaded            # keep the handle for callers that report it
    return runner


def build_llm(
    engine: str = "explicit",
    *,
    model: str = "Qwen/Qwen3-1.7B",
    device: str | None = None,
    dtype=None,
    quantization: str | None = None,
    static_cache: bool = False,
    compile_decode: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    chat: bool = False,
    extra_body: dict | None = None,
    tokenizer=None,
):
    """Return ``(generator, tokenizer, info)`` for the agent turn.

    The turn needs ``stream``/``generate`` and, for chat templating, a
    tokenizer. An HTTP engine has no local tokenizer, so one can be supplied
    (loading just the tokenizer is cheap and keeps the prompt identical to
    what the explicit path would send — which is the only way the two are
    comparable).

    ``chat`` defaults to False for the HTTP engine on purpose: the turn has
    already rendered the model's chat template, and letting the server apply
    its own on top would double-wrap the prompt. With a Qwen3 tokenizer that
    is not merely redundant -- a server template rendered without
    ``enable_thinking=False`` re-enables thinking, so the model fills its
    budget with ``<think>`` and there is nothing to speak. That failure is
    what :mod:`llm.prompting` exists to prevent, and it would reappear
    silently, as a quality problem rather than a serving bug, so the
    combination is refused here.
    """
    if engine not in LLM_ENGINES:
        raise ValueError(f"llm engine must be one of {LLM_ENGINES}, got {engine!r}")

    if engine == "http":
        from llm.engines import HttpLLMEngine

        if chat and tokenizer is not False:
            raise ValueError(
                "chat=True double-wraps the prompt: agent.turn already renders the "
                "model's chat template via llm.prompting (with enable_thinking=False "
                "for Qwen3), so the server would template an already-templated "
                "string. Use the /v1/completions route (chat=False). Pass "
                "tokenizer=False only if you intend the server to own templating.",
            )
        if tokenizer is False:
            tokenizer = None
        elif tokenizer is None:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model)
            except Exception:  # noqa: BLE001 - a plain prompt still works
                tokenizer = None
        generator = HttpLLMEngine(
            model, base_url=base_url or "http://127.0.0.1:8000/v1",
            api_key=api_key, chat=chat, tokenizer=tokenizer, extra_body=extra_body,
        )
        return generator, tokenizer, {"engine": "http", "model": model,
                                      "endpoint": generator.endpoint, "chat": chat,
                                      "templated_by": "server" if chat else "client"}

    from llm import LLMRunner, load_llm

    # dtype is explicit here rather than left to `pick_dtype` whenever two
    # engines are being compared: a bf16 reference against an fp16 server is
    # two different sets of numerics, and two greedy decoders over different
    # numerics diverge for reasons that say nothing about either engine.
    loaded = load_llm(model, device=device, dtype=dtype, quantization=quantization)
    generator = LLMRunner(loaded.model, loaded.tokenizer, loaded.device,
                          static_cache=static_cache, compile_decode=compile_decode)
    return generator, loaded.tokenizer, {
        "engine": "explicit", "model": model, "device": str(loaded.device),
        "dtype": str(loaded.dtype), "quantization": loaded.quantization,
        "static_cache": static_cache, "compile_decode": compile_decode,
    }
