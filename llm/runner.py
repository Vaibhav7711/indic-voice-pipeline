"""Explicit LLM prefill/decode runner with CUDA timing.

Lightweight version for the voice pipeline. Owns the autoregressive loop —
does not call model.generate().
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from time import perf_counter_ns

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMMetrics:
    prefill_ms: float = 0.0
    decode_ms: list[float] = field(default_factory=list)
    total_ms: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    peak_allocated_bytes: int = 0
    #: Set when the n-gram loop guard cut generation short. A stop that came
    #: from EOS or the token budget leaves this False.
    stopped_on_repetition: bool = False
    #: Set when a streaming consumer's should_stop() ended decoding (barge-in).
    stopped_by_caller: bool = False

    @property
    def mean_decode_ms(self) -> float:
        return sum(self.decode_ms) / len(self.decode_ms) if self.decode_ms else 0.0

    def as_dict(self) -> dict:
        return {
            "prefill_ms": self.prefill_ms,
            "mean_decode_ms": self.mean_decode_ms,
            "total_decode_ms": sum(self.decode_ms),
            "total_ms": self.total_ms,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "stopped_on_repetition": self.stopped_on_repetition,
            "stopped_by_caller": self.stopped_by_caller,
        }


class _Timer:
    """CUDA-event timing on a GPU, perf_counter elsewhere. Same call shape, so
    the decode loop is identical on both and unit-testable on CPU."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.device = device

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = perf_counter_ns()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self._end.record()
            self._end.synchronize()
            self.ms = self._start.elapsed_time(self._end)
        else:
            self.ms = (perf_counter_ns() - self._t0) / 1_000_000
        return False


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_bytes(device: torch.device) -> int:
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0


@dataclass
class LLMResult:
    text: str
    token_ids: list[int]
    metrics: LLMMetrics


class LLMRunner:
    """Explicit prefill/decode LLM runner."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        *,
        repetition_penalty: float = 1.1,
        loop_guard_ngram: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Greedy decoding on a small model loops easily. The penalty (CTRL-style:
        # divide positive / multiply negative logits of already-seen tokens)
        # makes loops unlikely; the n-gram guard is the backstop that ends one
        # if it still happens, and the metrics say when it fired.
        self.repetition_penalty = repetition_penalty
        self.loop_guard_ngram = loop_guard_ngram
        #: Metrics of the most recent (or in-progress) generate()/stream() call.
        self.last_metrics: LLMMetrics | None = None

        eos = tokenizer.eos_token_id
        self.eos_ids: set[int] = {eos} if isinstance(eos, int) else set(eos or [])

    def _select(self, logits: torch.Tensor, seen: list[int]) -> torch.Tensor:
        """Greedy pick after repetition penalty. ``logits`` is (1, vocab)."""
        if self.repetition_penalty != 1.0 and seen:
            logits = logits.float().clone()
            ids = torch.tensor(sorted(set(seen)), device=logits.device)
            scores = logits[0, ids]
            logits[0, ids] = torch.where(
                scores > 0, scores / self.repetition_penalty, scores * self.repetition_penalty,
            )
        return logits.argmax(dim=-1, keepdim=True)

    def _decode_tokens(
        self,
        prompt: str,
        metrics: LLMMetrics,
        *,
        max_new_tokens: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> Generator[int, None, list[int]]:
        """The one decode loop. Yields token ids as they are chosen and
        *returns* the final token list (the loop guard can retract tokens
        that were already yielded).

        ``generate()`` uses the returned list; ``stream()`` turns the yielded
        ids into text deltas. ``metrics`` is filled in as decoding proceeds so
        a consumer that stops early still gets timings for what ran.
        """
        total_start = perf_counter_ns()
        _reset_peak(self.device)
        # Visible while streaming, so a consumer can read live counts.
        self.last_metrics = metrics

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        metrics.prompt_tokens = input_ids.shape[1]

        # Prefill.
        with _Timer(self.device) as timer, torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        metrics.prefill_ms = timer.ms

        seen = input_ids[0].tolist()
        next_token = self._select(outputs.logits[:, -1, :], seen)
        past_kv = outputs.past_key_values
        generated: list[int] = []
        n = self.loop_guard_ngram

        try:
            for _step in range(max_new_tokens):
                token_id = int(next_token.item())
                if token_id in self.eos_ids:
                    break
                generated.append(token_id)
                seen.append(token_id)
                if n and len(generated) >= 2 * n and generated[-n:] == generated[-2 * n:-n]:
                    # Loop: retract the repeated n-gram. It has not been
                    # yielded yet if the consumer keeps pace, but a streaming
                    # consumer may already have shown it — see stream().
                    del generated[-n:]
                    metrics.stopped_on_repetition = True
                    break
                yield token_id
                if should_stop is not None and should_stop():
                    metrics.stopped_by_caller = True
                    break

                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device),
                ], dim=1)

                with _Timer(self.device) as timer, torch.inference_mode():
                    outputs = self.model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                metrics.decode_ms.append(timer.ms)

                next_token = self._select(outputs.logits[:, -1, :], seen)
                past_kv = outputs.past_key_values
        finally:
            metrics.generated_tokens = len(generated)
            metrics.total_ms = (perf_counter_ns() - total_start) / 1_000_000
            metrics.peak_allocated_bytes = _peak_bytes(self.device)
            self.last_metrics = metrics
        return generated

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> LLMResult:
        """Generate text from a prompt with explicit prefill/decode and timing."""
        metrics = LLMMetrics()
        gen = self._decode_tokens(prompt, metrics, max_new_tokens=max_new_tokens)
        while True:
            try:
                next(gen)
            except StopIteration as done:
                generated = done.value
                break
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return LLMResult(text.strip(), generated, metrics)

    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        """Yield text as it is generated. Timings land in ``self.last_metrics``.

        Deltas are cut from the running decode of *all* generated tokens, not
        from decoding tokens one at a time: a BPE token can be a fragment of
        a multi-byte character, and Devanagari matras are separate code points
        that combine with the consonant before them. Decoding the whole
        sequence and emitting only the new suffix — held back while it ends
        in a replacement character — keeps every yielded piece valid text.

        The n-gram loop guard retracts tokens that were already yielded; the
        consumer cannot un-say them, so the guard is a backstop and the
        repetition penalty is the real defence.
        """
        metrics = LLMMetrics()
        generated: list[int] = []
        emitted = ""
        for token_id in self._decode_tokens(
            prompt, metrics, max_new_tokens=max_new_tokens, should_stop=should_stop,
        ):
            generated.append(token_id)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if text.endswith("\ufffd"):
                continue
            if len(text) > len(emitted) and text.startswith(emitted):
                delta = text[len(emitted):]
                emitted = text
                yield delta
            elif text != emitted:
                # Decoder re-spelled an earlier piece (rare); resync by
                # emitting from the common prefix.
                common = 0
                for a, b in zip(emitted, text, strict=False):
                    if a != b:
                        break
                    common += 1
                emitted = text
                yield text[common:]
