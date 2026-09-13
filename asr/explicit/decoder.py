"""Explicit Whisper decoder with autoregressive KV-cached decoding.

Whisper's decoder has two KV caches per layer:
  - Self-attention: grows with each decoded token (same as GPT-style LLMs).
  - Cross-attention: computed once from encoder outputs, fixed thereafter.

Decoder prompt tokens control behavior:
  <|startoftranscript|> <|language|> <|task|> [<|notimestamps|>]

For Hindi transcription: [startoftranscript, hi, transcribe, notimestamps]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class DecoderState:
    past_key_values: object  # (self_attn_kv, cross_attn_kv) per layer
    encoder_outputs: BaseModelOutput
    next_token: torch.Tensor
    decoded_tokens: list[int]


class WhisperDecoder:
    """Explicit autoregressive Whisper decoder with per-step CUDA timing.

    Usage:
        state, ms = decoder.prefill(encoder_outputs, language="hi")
        while not done:
            state, ms = decoder.decode_one(state)
    """

    def __init__(self, model: WhisperForConditionalGeneration, device: torch.device):
        self.model = model
        self.device = device
        self.gen_config = model.generation_config

    def _build_prompt_ids(
        self,
        language: str | None = None,
        task: str = "transcribe",
        timestamps: bool = False,
    ) -> list[int]:
        """Build the 3-4 token decoder prompt sequence."""
        ids: list[int] = [self.gen_config.decoder_start_token_id]

        # Language token.
        if language is not None:
            lang_to_id = getattr(self.gen_config, "lang_to_id", {}) or {}
            # Try both "<|hi|>" and "hi" formats.
            lang_id = lang_to_id.get(f"<|{language}|>") or lang_to_id.get(language)
            if lang_id is not None:
                ids.append(lang_id)

        # Task token.
        task_to_id = getattr(self.gen_config, "task_to_id", {}) or {}
        task_id = task_to_id.get(task)
        if task_id is not None:
            ids.append(task_id)

        # No-timestamps token.
        if not timestamps:
            no_ts = getattr(self.gen_config, "no_timestamps_token_id", None)
            if no_ts is not None:
                ids.append(no_ts)

        return ids

    @torch.inference_mode()
    def prefill(
        self,
        encoder_outputs: BaseModelOutput,
        *,
        language: str | None = None,
        task: str = "transcribe",
        timestamps: bool = False,
    ) -> tuple[DecoderState, float]:
        """Process decoder prompt tokens. Populates both KV caches.

        Returns the initial DecoderState and CUDA-timed prefill latency.
        """
        prompt_ids = self._build_prompt_ids(language, task, timestamps)
        decoder_input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self.device,
        )

        torch.cuda.synchronize(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        outputs = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        end.record()
        end.synchronize()

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        state = DecoderState(
            past_key_values=outputs.past_key_values,
            encoder_outputs=encoder_outputs,
            next_token=next_token,
            decoded_tokens=[],
        )
        return state, start.elapsed_time(end)

    @torch.inference_mode()
    def decode_one(self, state: DecoderState) -> tuple[DecoderState, float]:
        """Decode one token. Self-attn cache grows; cross-attn cache reused."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        outputs = self.model(
            encoder_outputs=state.encoder_outputs,
            decoder_input_ids=state.next_token,
            past_key_values=state.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        end.record()
        end.synchronize()

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        new_decoded = state.decoded_tokens + [int(state.next_token.item())]

        return DecoderState(
            past_key_values=outputs.past_key_values,
            encoder_outputs=state.encoder_outputs,
            next_token=next_token,
            decoded_tokens=new_decoded,
        ), start.elapsed_time(end)
