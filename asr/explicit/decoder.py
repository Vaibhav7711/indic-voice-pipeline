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
from time import perf_counter_ns

import torch
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from asr.explicit.timing import Timer


@dataclass
class DecoderState:
    past_key_values: object  # (self_attn_kv, cross_attn_kv) per layer
    encoder_outputs: BaseModelOutput
    next_token: torch.Tensor
    decoded_tokens: list[int]
    #: P(<|nospeech|>) at the first decoding position; None if unavailable.
    no_speech_prob: float | None = None


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
        # generate() masks these before every argmax (SuppressTokensLogitsProcessor)
        # and these only at the first generated position (…AtBegin). On clean
        # speech the argmax never lands on them; on silence or noise it does,
        # and the explicit loop must take the same path as the reference.
        self._suppress = self._ids_tensor(getattr(self.gen_config, "suppress_tokens", None))
        self._begin_suppress = self._ids_tensor(
            getattr(self.gen_config, "begin_suppress_tokens", None),
        )
        # <|nospeech|>: Whisper's own estimate that the window contains no
        # speech. Read at the first decoding position, exactly where
        # generate() reads it. Without this the greedy loop transcribes
        # silence or noise into a hallucinated phrase and then repeats it.
        self.no_speech_token_id = self._resolve_no_speech_token()

    def _resolve_no_speech_token(self) -> int | None:
        for attr in ("no_speech_token_id", "no_speech_token"):
            value = getattr(self.gen_config, attr, None)
            if isinstance(value, int):
                return value
        # Not in the generation config on every checkpoint; the tokenizer
        # knows it, but the decoder is not given one, so fall back to the
        # id that has been stable across Whisper releases.
        return 50363

    def _ids_tensor(self, ids) -> torch.Tensor | None:
        ids = [int(i) for i in (ids or [])]
        return torch.tensor(ids, dtype=torch.long, device=self.device) if ids else None

    def _pick(self, logits: torch.Tensor, *, at_begin: bool) -> torch.Tensor:
        """Greedy token after the same suppression generate() applies. ``logits``
        is (batch, vocab) for the last position."""
        if self._suppress is not None or (at_begin and self._begin_suppress is not None):
            logits = logits.clone()
            if self._suppress is not None:
                logits[:, self._suppress] = float("-inf")
            if at_begin and self._begin_suppress is not None:
                logits[:, self._begin_suppress] = float("-inf")
        return logits.argmax(dim=-1, keepdim=True)

    def _language_token_ids(self) -> dict[str, int]:
        """``{"hi": 50276, ...}`` from the generation config, accepting either
        ``<|hi|>`` or ``hi`` key spellings."""
        lang_to_id = getattr(self.gen_config, "lang_to_id", {}) or {}
        table: dict[str, int] = {}
        for key, token_id in lang_to_id.items():
            code = key[2:-2] if key.startswith("<|") and key.endswith("|>") else key
            table[code] = int(token_id)
        return table

    def _build_prompt_ids(
        self,
        language: str,
        task: str = "transcribe",
        timestamps: bool = False,
    ) -> list[int]:
        """Build the decoder prompt ``<|sot|> <|lang|> <|task|> [<|notimestamps|>]``.

        Whisper's prompt grammar puts the language token immediately after
        ``<|startoftranscript|>``. Skipping it and going straight to the task
        token is a sequence the model never saw in training, so ``language``
        is required here; callers that want auto-detection run
        :meth:`detect_language` first.
        """
        if not language:
            raise ValueError(
                "language is required to build a Whisper prompt; call "
                "detect_language() first for automatic detection",
            )
        ids: list[int] = [self.gen_config.decoder_start_token_id]

        lang_id = self._language_token_ids().get(language)
        if lang_id is None:
            raise ValueError(f"unknown Whisper language code {language!r}")
        ids.append(lang_id)

        task_to_id = getattr(self.gen_config, "task_to_id", {}) or {}
        task_id = task_to_id.get(task)
        if task_id is not None:
            ids.append(task_id)

        if not timestamps:
            no_ts = getattr(self.gen_config, "no_timestamps_token_id", None)
            if no_ts is not None:
                ids.append(no_ts)

        return ids

    def strip_generate_output(self, ids: list[int]) -> list[int]:
        """Reduce a ``model.generate()`` sequence to the tokens the explicit
        loop would have produced, for like-for-like comparison.

        transformers < 5 returns the decoder prompt
        (``<|sot|><|lang|><|task|><|notimestamps|>``) at the front; 5.x
        strips it. Both shapes are handled by looking at the first token, and
        a trailing EOS is dropped so the comparison is over content only.
        """
        out = list(ids)
        if out and out[0] == self.gen_config.decoder_start_token_id:
            no_ts = getattr(self.gen_config, "no_timestamps_token_id", None)
            if no_ts in out:
                out = out[out.index(no_ts) + 1:]
            else:
                out = out[4:]
        eos = self.gen_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        while out and out[-1] in eos_ids:
            out.pop()
        return out

    @torch.inference_mode()
    def detect_language(
        self,
        encoder_outputs: BaseModelOutput,
        *,
        candidates: list[str] | None = None,
    ) -> tuple[str, float, float]:
        """Predict the spoken language from a single decoder step.

        Runs the decoder on ``<|startoftranscript|>`` alone and takes the
        argmax over language tokens only — the same procedure Whisper's
        reference implementation uses. Returns ``(code, probability, ms)``.

        ``candidates`` restricts the choice to a set of language codes. Use
        it when the deployment only serves a few languages: a fine-tuned
        adapter whose training labels lacked the language token (the v1
        Hindi adapter is one) skews this position's distribution, and
        restricting candidates is what keeps detection usable on it.
        """
        table = self._language_token_ids()
        if candidates:
            unknown = [c for c in candidates if c not in table]
            if unknown:
                raise ValueError(f"unknown Whisper language codes {unknown!r}")
            table = {c: table[c] for c in candidates}
        if not table:
            raise RuntimeError("generation config has no lang_to_id table")

        decoder_input_ids = torch.tensor(
            [[self.gen_config.decoder_start_token_id]], dtype=torch.long,
            device=self.device,
        )
        start = perf_counter_ns()
        logits = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        ).logits[0, -1].float()
        codes = list(table)
        ids = torch.tensor([table[c] for c in codes], device=logits.device)
        probs = torch.softmax(logits[ids], dim=-1)
        best = int(probs.argmax().item())
        ms = (perf_counter_ns() - start) / 1_000_000
        return codes[best], float(probs[best].item()), ms

    @torch.inference_mode()
    def prefill(
        self,
        encoder_outputs: BaseModelOutput,
        *,
        language: str,
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

        with Timer(self.device) as timer:
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )

        last_logits = outputs.logits[:, -1, :]
        no_speech_prob = None
        if self.no_speech_token_id is not None:
            probs = torch.softmax(last_logits.float(), dim=-1)
            no_speech_prob = float(probs[0, self.no_speech_token_id].item())

        next_token = self._pick(last_logits, at_begin=True)
        state = DecoderState(
            past_key_values=outputs.past_key_values,
            encoder_outputs=encoder_outputs,
            next_token=next_token,
            decoded_tokens=[],
            no_speech_prob=no_speech_prob,
        )
        return state, timer.ms

    @torch.inference_mode()
    def decode_one(self, state: DecoderState) -> tuple[DecoderState, float]:
        """Decode one token. Self-attn cache grows; cross-attn cache reused."""
        with Timer(self.device) as timer:
            outputs = self.model(
                encoder_outputs=state.encoder_outputs,
                decoder_input_ids=state.next_token,
                past_key_values=state.past_key_values,
                use_cache=True,
                return_dict=True,
            )

        next_token = self._pick(outputs.logits[:, -1, :], at_begin=False)
        new_decoded = list(state.decoded_tokens)

        return DecoderState(
            past_key_values=outputs.past_key_values,
            encoder_outputs=state.encoder_outputs,
            next_token=next_token,
            decoded_tokens=new_decoded,
            no_speech_prob=state.no_speech_prob,
        ), timer.ms
