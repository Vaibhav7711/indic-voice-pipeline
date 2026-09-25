"""The HTTP engine adapter: contract, streaming, metrics honesty, barge-in.

No server involved — the transport is injected, so the streaming and metrics
logic is tested directly.
"""

from __future__ import annotations

import json

from llm.engines import HttpLLMEngine


def sse(chunks, usage=None, done=True):
    """Build an OpenAI-style SSE transport from content pieces."""
    lines = []
    for piece in chunks:
        lines.append("data: " + json.dumps(
            {"choices": [{"delta": {"content": piece}}]}))
    if usage is not None:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    if done:
        lines.append("data: [DONE]")

    def transport(url, payload, headers):
        transport.calls.append({"url": url, "payload": payload, "headers": headers})
        yield from lines

    transport.calls = []
    return transport


class TestContract:
    def test_stream_yields_the_pieces(self):
        engine = HttpLLMEngine("qwen", transport=sse(["नम", "स्ते", " दुनिया"]))
        assert list(engine.stream("p")) == ["नम", "स्ते", " दुनिया"]

    def test_generate_collects_the_stream(self):
        engine = HttpLLMEngine("qwen", transport=sse(["एक ", "दो"]))
        result = engine.generate("p")
        assert result.text == "एक दो"
        assert result.metrics is engine.last_metrics

    def test_it_satisfies_what_the_turn_looks_for(self):
        engine = HttpLLMEngine("qwen", transport=sse(["x"]))
        assert callable(getattr(engine, "stream", None))
        assert callable(getattr(engine, "generate", None))
        assert hasattr(engine, "last_metrics") and hasattr(engine, "tokenizer")

    def test_completions_endpoint_sends_the_prompt_verbatim(self):
        """The turn already renders the model's chat template, so the
        completions route is the faithful one."""
        transport = sse(["x"])
        engine = HttpLLMEngine("qwen", chat=False, transport=transport)
        list(engine.stream("RENDERED PROMPT", max_new_tokens=7))
        payload = transport.calls[0]["payload"]
        assert payload["prompt"] == "RENDERED PROMPT"
        assert "messages" not in payload
        assert payload["max_tokens"] == 7
        assert payload["stream"] is True
        assert payload["temperature"] == 0.0, "greedy, to match the explicit runner"
        assert transport.calls[0]["url"].endswith("/v1/completions")

    def test_chat_endpoint_wraps_the_prompt(self):
        transport = sse(["x"])
        engine = HttpLLMEngine("qwen", chat=True, transport=transport)
        list(engine.stream("hello"))
        payload = transport.calls[0]["payload"]
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert transport.calls[0]["url"].endswith("/v1/chat/completions")

    def test_api_key_and_extra_body_are_passed(self):
        transport = sse(["x"])
        engine = HttpLLMEngine("qwen", api_key="secret", transport=transport,
                               extra_body={"top_k": 1, "repetition_penalty": 1.1})
        list(engine.stream("p"))
        call = transport.calls[0]
        assert call["headers"]["Authorization"] == "Bearer secret"
        assert call["payload"]["top_k"] == 1
        assert call["payload"]["repetition_penalty"] == 1.1


class TestMetrics:
    def test_first_chunk_is_time_to_first_token_and_is_flagged_as_arrival(self):
        engine = HttpLLMEngine("qwen", transport=sse(["a", "b", "c"]))
        list(engine.stream("p"))
        m = engine.last_metrics
        assert m.prefill_ms is not None and m.prefill_ms >= 0
        assert m.prefill_is_arrival_time is True, "includes network latency; say so"
        assert len(m.decode_ms) == 2, "gaps between chunks, not counting the first"
        assert m.total_ms >= m.prefill_ms

    def test_reported_usage_is_preferred_over_counting_chunks(self):
        engine = HttpLLMEngine("qwen", transport=sse(
            ["a", "b"], usage={"prompt_tokens": 31, "completion_tokens": 44}))
        list(engine.stream("p"))
        m = engine.last_metrics
        assert m.prompt_tokens == 31
        assert m.generated_tokens == 44, "not the 2 chunks we saw"

    def test_without_usage_the_token_count_is_a_chunk_lower_bound(self):
        engine = HttpLLMEngine("qwen", transport=sse(["a", "b", "c"]))
        list(engine.stream("p"))
        assert engine.last_metrics.prompt_tokens is None, "unknown, not zero"
        assert engine.last_metrics.generated_tokens == 3

    def test_unmeasured_fields_are_none_not_zero(self):
        engine = HttpLLMEngine("qwen", transport=sse([]))
        list(engine.stream("p"))
        m = engine.last_metrics
        assert m.prefill_ms is None
        assert m.mean_decode_ms is None
        assert m.generated_tokens is None
        assert m.as_dict()["total_decode_ms"] is None

    def test_malformed_and_comment_lines_are_skipped(self):
        def transport(url, payload, headers):
            yield ": keep-alive"
            yield ""
            yield "data: {not json"
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]})
            yield "data: [DONE]"

        engine = HttpLLMEngine("qwen", transport=transport)
        assert list(engine.stream("p")) == ["ok"]


class TestBargeIn:
    def test_should_stop_ends_the_stream_and_is_recorded(self):
        """Abandoning the stream lets the engine free the sequence instead of
        generating tokens nobody will hear."""
        seen = []
        engine = HttpLLMEngine("qwen", transport=sse(["a", "b", "c", "d"]))
        for piece in engine.stream("p", should_stop=lambda: len(seen) >= 2):
            seen.append(piece)
        assert seen == ["a", "b"]
        assert engine.last_metrics.stopped_by_caller is True

    def test_a_completed_stream_is_not_marked_stopped(self):
        engine = HttpLLMEngine("qwen", transport=sse(["a", "b"]))
        list(engine.stream("p", should_stop=lambda: False))
        assert engine.last_metrics.stopped_by_caller is False

    def test_metrics_survive_a_transport_failure(self):
        def transport(url, payload, headers):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "a"}}]})
            raise ConnectionError("engine died")

        engine = HttpLLMEngine("qwen", transport=transport)
        try:
            list(engine.stream("p"))
        except ConnectionError:
            pass
        assert engine.last_metrics is not None
        assert engine.last_metrics.total_ms > 0


def test_probe_reports_reachability_and_whether_usage_is_available():
    engine = HttpLLMEngine("qwen", transport=sse(
        ["ठीक"], usage={"prompt_tokens": 3, "completion_tokens": 1}))
    info = engine.probe()
    assert info["reachable"] is True and info["text"] == "ठीक"
    assert info["reported_usage"] is True
    assert info["endpoint"].endswith("/v1/chat/completions")
