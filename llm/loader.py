from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#: Weight bytes per parameter, for the VRAM arithmetic that decides whether a
#: model can sit alongside Whisper. 4-bit NF4 stores ~0.55 bytes/param once
#: the un-quantized layers (embeddings, norms, lm_head) are counted.
BYTES_PER_PARAM = {None: 2.0, "8bit": 1.0, "4bit": 0.55}


@dataclass(frozen=True)
class LoadedLLM:
    model_name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    dtype: torch.dtype
    device: torch.device
    quantization: str | None = None


def estimate_vram_gib(param_millions: float, quantization: str | None = None) -> float:
    """Resident weight footprint in GiB, for planning what fits together."""
    return param_millions * 1e6 * BYTES_PER_PARAM[quantization] / 1024 ** 3


def pick_dtype(requested: torch.dtype | None = None,
               device: torch.device | None = None) -> torch.dtype:
    """bf16 where the GPU has it natively, fp16 on other GPUs, fp32 on CPU.

    T4 (sm_75) has no bf16 tensor cores: PyTorch runs bf16 matmuls through
    a slow path, and a 0.6B model decoded at ~41 ms/token in the first
    sweeps for that reason. Ampere and later take bf16 as requested.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return torch.float32
    if requested is not None and requested != torch.bfloat16:
        return requested
    # Compute capability, not `torch.cuda.is_bf16_supported()`. That call
    # counts emulated support and answers True on a T4, so this function used
    # to hand back exactly the slow path its own docstring warns about -- a
    # measured sweep loaded bf16 on a T4 and its explicit arm carried a prefill
    # penalty that was read as the engine being fast. bf16 tensor cores arrive
    # with Ampere (sm_80).
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def load_llm(
    model_name: str = "Qwen/Qwen3-1.7B",
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    quantization: str | None = None,
) -> LoadedLLM:
    """Load a causal LM. ``dtype=None`` picks per device (see pick_dtype).

    ``quantization`` is ``None``, ``"8bit"`` or ``"4bit"`` (bitsandbytes NF4
    with the compute dtype above). It exists for the case where the LLM has
    to share a small card with Whisper: on 6 GB, Qwen3-1.7B in fp16 (3.4 GB)
    cannot sit alongside whisper-large-v3-turbo (1.6 GB) once KV cache and
    activations are counted, while at 4-bit (~1.2 GB) it can. Quantization
    costs quality and adds dequantization work per token, so measure both —
    ``benchmarks.llm_bakeoff`` accepts a ``model:4bit`` spec for exactly that
    comparison.
    """
    device = torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(dtype, device)
    # Value before environment: a typo should be reported as a typo, not as
    # a hardware problem.
    if quantization not in (None, "4bit", "8bit"):
        raise ValueError(f"quantization must be None, '4bit' or '8bit', got {quantization!r}")
    if quantization and device.type != "cuda":
        raise ValueError("quantization requires a CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantization:
        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        # device_map pins it where bitsandbytes expects; no .to() afterwards.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=config, device_map={"": device},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype,
        ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return LoadedLLM(model_name, model, tokenizer, dtype, device, quantization)
