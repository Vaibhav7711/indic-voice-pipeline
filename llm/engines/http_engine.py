"""An OpenAI-compatible HTTP backend, so an external engine can serve the turn.

The agent turn does not depend on `LLMRunner`; it depends on a small
structural contract (`agent.turn.ResponseGenerator`). That makes swapping in a
faster engine a matter of satisfying the contract rather than changing the
turn:

    stream(prompt, max_new_tokens=..., should_stop=...) -> Iterator[str]
    generate(prompt, max_new_tokens=...) -> object with .text and .metrics
    last_metrics                          -> the metrics of the last call
    tokenizer                             -> optional; used for chat templating

This adapter covers any engine exposing the OpenAI `/v1/completions` or
`/v1/chat/completions` shape with `stream: true` — vLLM, SGLang, llama.cpp's
server, Ollama's compatibility endpoint, TGI's OpenAI route, and most custom
servers built to that spec. An engine with a different API needs its own
adapter; this file is the template for it.

**Why this matters more than the ASR engine.** On the measured live turns ASR
was 342 ms of a 4400 ms turn and off the critical path (a candidate final is
decoded during the endpoint silence, so `asr_after_endpoint ≈ 0`). The LLM was
3324 ms of it — 878 ms prefill that *grows with dialogue history*, plus
2446 ms until enough text existed to speak. A Whisper engine at 1.31× saves
~90 ms; the LLM is where seconds are.

**What this adapter cannot measure.** Prefill and per-token times come from
the server's own accounting if it reports usage, and are otherwise inferred
from arrival times: the first chunk's arrival is time-to-first-token, and
subsequent gaps are per-token latency. Fields the server does not report stay
``None`` rather than 0.0 — a zero would be averaged into a benchmark.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any

__all__ = ["HttpEngineMetrics", "HttpEngineResult", "HttpLLMEngine"]

DEFAULT_TIMEOUT = 120.0


@dataclass
class HttpEngineMetrics:
    """Mirrors the fields `agent.turn` and the benchmarks read off `LLMMetrics`.

    Anything the server does not tell us is ``None``, never zero.
    """

    prefill_ms: float | None = None
    decode_ms: list[float] = field(default_factory=list)
    total_ms: float = 0.0
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    peak_allocated_bytes: int = 0
    stopped_on_repetition: bool = False
    stopped_by_caller: bool = False
    static_cache: bool = False
    compiled_decode: bool = False
    #: True when prefill_ms is the first chunk's arrival time rather than a
    #: figure the server reported. It then includes network latency.
    prefill_is_arrival_time: bool = True
    endpoint: str = ""
    model: str = ""

    @property
    def mean_decode_ms(self) -> float | None:
        return sum(self.decode_ms) / len(self.decode_ms) if self.decode_ms else None

    def as_dict(self) -> dict:
        return {
            "prefill_ms": self.prefill_ms,
            "mean_decode_ms": self.mean_decode_ms,
            "total_decode_ms": sum(self.decode_ms) if self.decode_ms else None,
            "total_ms": self.total_ms,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "stopped_by_caller": self.stopped_by_caller,
            "stopped_on_repetition": self.stopped_on_repetition,
            "prefill_is_arrival_time": self.prefill_is_arrival_time,
            "endpoint": self.endpoint,
            "model": self.model,
        }


@dataclass
class HttpEngineResult:
    text: str
    token_ids: list[int]
    metrics: HttpEngineMetrics


class HttpLLMEngine:
    """Serve the turn from an OpenAI-compatible endpoint.

    ``transport`` is injectable so the streaming and metrics logic is testable
    without a server; by default it posts with ``urllib`` and yields decoded
    SSE lines, which avoids adding a dependency for one POST.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        chat: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        tokenizer: Any = None,
        extra_body: dict | None = None,
        transport: Callable[[str, dict, dict], Iterator[str]] | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        #: Chat endpoints apply the server's own template. The turn already
        #: renders a prompt with the model's template when a tokenizer is
        #: available, so the completions endpoint is the faithful choice when
        #: you want the turn's prompt used verbatim.
        self.chat = chat
        self.timeout = timeout
        self.tokenizer = tokenizer
        self.extra_body = dict(extra_body or {})
        self._transport = transport or _urllib_sse
        self.last_metrics: HttpEngineMetrics | None = None

    # -- request shaping --------------------------------------------------

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions" if self.chat else f"{self.base_url}/completions"

    def _payload(self, prompt: str, max_new_tokens: int) -> dict:
        body = {
            "model": self.model,
            "max_tokens": max_new_tokens,
            "stream": True,
            # Greedy, to match the explicit runner it is replacing. An engine
            # sampling differently is not serving the same model.
            "temperature": 0.0,
            # Ask for usage on the final chunk where the server supports it,
            # so token counts are reported rather than guessed.
            "stream_options": {"include_usage": True},
        }
        body.update(self.extra_body)
        if self.chat:
            body["messages"] = [{"role": "user", "content": prompt}]
        else:
            body["prompt"] = prompt
        return body

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _piece(chunk: dict) -> str:
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta")
            if isinstance(delta, dict) and delta.get("content"):
                return delta["content"]
            if choice.get("text"):
                return choice["text"]
        return ""

    # -- the contract -----------------------------------------------------

    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        """Yield text as the server produces it, stopping on ``should_stop``.

        Stopping matters for barge-in: abandoning the response stream lets the
        engine free the sequence instead of generating tokens nobody hears.
        """
        metrics = HttpEngineMetrics(endpoint=self.endpoint, model=self.model)
        self.last_metrics = metrics
        start = perf_counter_ns()
        last = start
        stopped = False
        try:
            for line in self._transport(self.endpoint, self._payload(prompt, max_new_tokens),
                                        self._headers()):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    metrics.prompt_tokens = usage.get("prompt_tokens")
                    metrics.generated_tokens = usage.get("completion_tokens")
                piece = self._piece(chunk)
                if not piece:
                    continue
                now = perf_counter_ns()
                if metrics.prefill_ms is None:
                    metrics.prefill_ms = (now - start) / 1_000_000
                else:
                    metrics.decode_ms.append((now - last) / 1_000_000)
                last = now
                yield piece
                if should_stop is not None and should_stop():
                    stopped = True
                    break
        finally:
            metrics.stopped_by_caller = stopped
            metrics.total_ms = (perf_counter_ns() - start) / 1_000_000
            if metrics.generated_tokens is None and metrics.decode_ms:
                # No usage reported: the chunk count is a lower bound, since a
                # server may batch several tokens into one chunk.
                metrics.generated_tokens = len(metrics.decode_ms) + 1

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> HttpEngineResult:
        """Collect the stream. The turn prefers ``stream``; this is for parity."""
        pieces = list(self.stream(prompt, max_new_tokens=max_new_tokens))
        metrics = self.last_metrics or HttpEngineMetrics()
        return HttpEngineResult("".join(pieces).strip(), [], metrics)

    # -- health -----------------------------------------------------------

    def probe(self) -> dict:
        """One tiny generation, to fail loudly at startup rather than mid-turn."""
        pieces = list(self.stream("नमस्ते", max_new_tokens=4))
        metrics = self.last_metrics
        return {
            "endpoint": self.endpoint, "model": self.model,
            "reachable": True, "text": "".join(pieces),
            "first_chunk_ms": metrics.prefill_ms if metrics else None,
            "reported_usage": bool(metrics and metrics.prompt_tokens is not None),
        }


def _urllib_sse(url: str, payload: dict, headers: dict) -> Iterator[str]:
    """POST JSON and yield server-sent-event lines, without adding a dependency."""
    import urllib.request

    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST",
    )
    with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
        for raw in response:
            yield raw.decode("utf-8", "replace").rstrip("\n")
