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

**Streamed Indic text can arrive corrupted, and this adapter counts it.**
Qwen's tokenizer is byte-level BPE, so a 3-byte Devanagari character is
routinely split across two tokens. A server that streams by decoding the
tokens it has so far and sending the string diff will decode a partial
character as U+FFFD, send it, and then have no way to retract it -- the
corrected text is no longer an extension of what it already sent. The real
character is dropped and the replacement character is spoken. English never
shows this, because ASCII is one byte per character.

``replacement_chars`` counts U+FFFD in the streamed text for exactly this
reason. It is not cosmetic: corrupted text goes straight to TTS and is
pronounced, so a voice agent mangling every fourth character would otherwise
pass every test this pipeline has. Nothing here can repair it -- the bytes are
gone by the time they arrive -- so the count exists to make it loud.

**What this adapter cannot measure.** Token counts come from the server's own
accounting when it reports usage; times are inferred from arrival. The first
chunk's arrival is time-to-first-token *including network latency*, flagged by
``prefill_is_arrival_time``. Later gaps are gaps between *chunks*, and a chunk
is not a token: a server that polls its own generator (full-inference-engine
polls at 5 ms and sends the cumulative diff) batches however many tokens
arrived in that window, so ``decode_ms`` is chunk latency, flagged by
``decode_is_per_chunk``. Fields the server does not report stay ``None``
rather than 0.0 — a zero would be averaged into a benchmark.

**Verified against** `Vaibhav7711/full-inference-engine`
(`engine/server/openai.py`): paged KV, continuous batching, CUDA-graphed
decode, and Qwen3-0.6B — the model this pipeline already serves. Its
`temperature = 0` is greedy and takes precedence over every other knob, so
token-identity against the explicit runner is a gate that can actually be
run. Use its `/v1/completions` route, never `/v1/chat/completions`: see
``chat`` below.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter_ns
from typing import Any

__all__ = ["HttpEngineMetrics", "HttpEngineResult", "HttpLLMEngine"]

DEFAULT_TIMEOUT = 120.0

#: U+FFFD, written as a code point so it is legible in source. See the
#: module docstring: its presence in streamed text means the server split a
#: multi-byte character across chunks.
REPLACEMENT_CHAR = chr(0xFFFD)


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
    #: Chunks received. Not the token count: see decode_is_per_chunk.
    chunks: int = 0
    #: True when decode_ms entries are gaps between chunks rather than between
    #: tokens. A polling server batches whatever tokens landed in its window,
    #: so these are an upper bound on per-token latency and the count of them
    #: is a lower bound on the token count.
    decode_is_per_chunk: bool = True
    #: U+FFFD characters in the streamed text. Non-zero means the server's
    #: incremental decode is splitting multi-byte characters; see the module
    #: docstring. The text cannot be repaired here, only reported.
    replacement_chars: int = 0
    #: Status code and body of a failed request, so the engine's own refusal
    #: ("prompt exceeds the 4096-token server limit") reaches the caller
    #: instead of urllib's bare "HTTP Error 413".
    error: str | None = None
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
            "replacement_chars": self.replacement_chars,
            "prefill_is_arrival_time": self.prefill_is_arrival_time,
            "decode_is_per_chunk": self.decode_is_per_chunk,
            "chunks": self.chunks or None,
            "error": self.error,
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
        chat: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        tokenizer: Any = None,
        stop: list[str] | None = None,
        loop_guard_chars: int = 48,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        presence_penalty: float = 0.0,
        seed: int | None = None,
        extra_body: dict | None = None,
        transport: Callable[[str, dict, dict], Iterator[str]] | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        #: Chat endpoints apply the server's own template, and the turn has
        #: already applied the model's. Sending a rendered prompt as a chat
        #: message wraps `<|im_start|>system ...` inside another user turn.
        #: Worse, on Qwen3 a server template rendered without
        #: `enable_thinking=False` re-enables thinking mode, and the model
        #: spends its whole budget inside `<think>` with nothing to speak --
        #: the exact bug `llm.prompting` exists to prevent. Completions is
        #: therefore the default, and `build_llm` refuses the combination that
        #: would double-wrap.
        self.chat = chat
        self.timeout = timeout
        self.tokenizer = tokenizer
        self.stop = list(stop or [])
        #: Tail length in characters for the repetition guard; 0 disables it.
        #: 48 covers loop periods up to 16, which includes the 11-character
        #: period measured in a served run, and is short enough that ordinary
        #: prose does not trip it -- the measured first spoken unit was 41
        #: characters.
        self.loop_guard_chars = loop_guard_chars
        #: Sampling, sent to the server. 0.0 is greedy, which is what makes a
        #: parity comparison against the explicit runner meaningful -- and what
        #: the engine's own SamplingParams documents as taking precedence over
        #: every other knob. Raise it only when the thing being measured is
        #: answer quality rather than engine equivalence.
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.extra_body = dict(extra_body or {})
        # The timeout is bound here rather than added to the transport
        # signature: a caller injecting a transport for a test should not have
        # to accept a parameter that only urllib uses.
        self._transport = transport or partial(_urllib_sse, timeout=timeout)
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
            # 0.0 is greedy, matching the explicit runner it replaces: an
            # engine sampling differently is not serving the same model, and
            # every parity gate here depends on that.
            "temperature": self.temperature,
            # Ask for usage on the final chunk where the server supports it,
            # so token counts are reported rather than guessed.
            "stream_options": {"include_usage": True},
        }
        # Only sent when they differ from the server's own defaults, so a
        # greedy request stays a minimal greedy request.
        if self.top_p < 1.0:
            body["top_p"] = self.top_p
        if self.top_k > 0:
            body["top_k"] = self.top_k
        if self.presence_penalty:
            body["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            body["seed"] = self.seed
        if self.stop:
            body["stop"] = self.stop
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

    @staticmethod
    def _looping(text: str, size: int, *, repeats: int = 3,
                 min_period: int = 4) -> bool:
        """True when the tail is one block repeated `repeats` times over.

        `LLMRunner` guards against degenerate greedy output with an n-gram
        check on token ids and reports `stopped_on_repetition`. Serving over
        HTTP dropped that guard silently: the server owns the decode, so
        nothing downstream could stop it. A measured run caught the cost -- the
        served arm produced `एक बर्फ के` thirteen times and ran to its
        128-token cap while the explicit arm answered the same prompt
        coherently.

        The period is searched rather than assumed. Comparing two fixed halves
        of the window only fires when the window happens to be a multiple of
        the loop's period, and the observed loop has a period of 11 characters,
        which a 24-character window misses entirely.

        `repeats=3` and `min_period=4` are deliberately conservative: two
        repeats of a short phrase happens in ordinary prose, and cutting a
        legitimate answer mid-sentence is worse than speaking a loop. The check
        is on characters, because that is all this side receives, so it is
        weaker than a token-level guard and no substitute for one in the
        engine.
        """
        if size <= 0 or repeats < 2:
            return False
        tail = text[-size:]
        if len(tail) < size:
            return False
        for period in range(min_period, size // repeats + 1):
            block = tail[-period:]
            if tail[-period * repeats:] == block * repeats:
                return True
        return False

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
        completed = False
        #: Everything yielded so far, for the repetition guard. Text only, so
        #: it stays small next to the audio the turn is already buffering.
        seen: list[str] = []
        # Held so it can be closed explicitly: closing the response is what the
        # server sees as a disconnect, and this engine cancels the request on
        # disconnect. Leaving it to garbage collection would keep generating
        # tokens nobody hears for as long as the reference survives.
        source = self._transport(self.endpoint, self._payload(prompt, max_new_tokens),
                                 self._headers())
        try:
            for line in source:
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    completed = True
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
                metrics.chunks += 1
                metrics.replacement_chars += piece.count(REPLACEMENT_CHAR)
                seen.append(piece)
                if metrics.prefill_ms is None:
                    metrics.prefill_ms = (now - start) / 1_000_000
                else:
                    metrics.decode_ms.append((now - last) / 1_000_000)
                last = now
                yield piece
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                if self._looping("".join(seen), self.loop_guard_chars):
                    # Abandoning the stream closes the socket, which this
                    # engine reads as a disconnect and cancels the request, so
                    # the loop stops costing GPU time as well as going unspoken.
                    metrics.stopped_on_repetition = True
                    stopped = True
                    break
            else:
                # The transport ran out without a [DONE]. Still a complete
                # response as far as this side is concerned.
                completed = True
        except OSError as error:                        # includes urllib's HTTPError
            metrics.error = _error_detail(error)
            raise
        finally:
            closer = getattr(source, "close", None)
            if closer is not None:
                closer()
            # A caller stops this stream in two ways, and only one of them
            # trips `should_stop`. `agent.turn` cancels playback and abandons
            # the generator, which arrives here as GeneratorExit -- no break,
            # no error, just a stream that ended early. Recording only the
            # explicit case reported `stopped_by_caller: False` on a real
            # barge-in, which is the metric claiming the response finished
            # when the user had cut it off mid-sentence.
            metrics.stopped_by_caller = stopped or (
                not completed and metrics.error is None)
            metrics.total_ms = (perf_counter_ns() - start) / 1_000_000
            if metrics.generated_tokens is None and metrics.chunks:
                # No usage reported: the chunk count is a lower bound, since a
                # server may batch several tokens into one chunk. A server that
                # reports usage overwrites this with the real count.
                metrics.generated_tokens = metrics.chunks

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> HttpEngineResult:
        """Collect the stream. The turn prefers ``stream``; this is for parity."""
        pieces = list(self.stream(prompt, max_new_tokens=max_new_tokens))
        metrics = self.last_metrics or HttpEngineMetrics()
        return HttpEngineResult("".join(pieces).strip(), [], metrics)

    # -- health -----------------------------------------------------------

    def probe(self) -> dict:
        """One tiny generation, to fail loudly at startup rather than mid-turn.

        This does not double as a readiness wait. An engine that captures CUDA
        graphs and JITs Triton kernels at startup answers 503 until warmup
        finishes, so a probe that retried would hide a cold server behind a
        first turn that looks pathologically slow. Wait for readiness before
        constructing the agent -- `scripts/serve_llm.py` does.
        """
        pieces = list(self.stream("नमस्ते", max_new_tokens=4))
        metrics = self.last_metrics
        return {
            "endpoint": self.endpoint, "model": self.model,
            "reachable": True, "text": "".join(pieces),
            "first_chunk_ms": metrics.prefill_ms if metrics else None,
            "reported_usage": bool(metrics and metrics.prompt_tokens is not None),
            "chunks": metrics.chunks if metrics else None,
            # The probe prompt is Devanagari on purpose: this is where a server
            # that splits multi-byte characters across stream chunks shows
            # itself, at startup, rather than in a transcript days later.
            "replacement_chars": metrics.replacement_chars if metrics else None,
        }


def _error_detail(error: Exception) -> str:
    """Read a failed response's body, which is where the engine says why.

    A serving engine that refuses a request explains itself -- "prompt exceeds
    the 4096-token server limit", "server is not accepting requests". urllib
    renders that as "HTTP Error 413: Request Entity Too Large" and drops the
    body on the floor, which turns a fixable configuration error into a
    mystery.
    """
    code = getattr(error, "code", None)
    body = ""
    reader = getattr(error, "read", None)
    if reader is not None:
        try:
            body = reader().decode("utf-8", "replace")[:500]
        except Exception:  # noqa: BLE001 - the body is a diagnostic, not the error
            body = ""
    if body:
        try:
            parsed = json.loads(body)
        except ValueError:
            pass
        else:
            if isinstance(parsed, dict) and "detail" in parsed:
                body = str(parsed["detail"])
    return f"HTTP {code}: {body}" if code is not None else f"{type(error).__name__}: {error}"


def _urllib_sse(url: str, payload: dict, headers: dict,
                timeout: float = DEFAULT_TIMEOUT) -> Iterator[str]:
    """POST JSON and yield server-sent-event lines, without adding a dependency."""
    import urllib.request

    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            yield raw.decode("utf-8", "replace").rstrip("\n")
