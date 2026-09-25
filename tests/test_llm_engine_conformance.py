"""Conformance against the wire shapes `full-inference-engine` actually emits.

`tests/test_llm_http_engine.py` tests the adapter against the OpenAI format as
specified. This module tests it against one *implementation* of that format --
`Vaibhav7711/full-inference-engine`, the engine this pipeline serves Qwen3-0.6B
with -- because the places a format leaves room for choices are exactly the
places an adapter breaks:

  * completions stream deltas arrive as `choices[0].text`, not `delta.content`
  * chat streams put `role: assistant` in the *first* delta, with no content
  * the terminal chunk carries `finish_reason` and an empty delta
  * `stream_options.include_usage` appends a chunk with `choices: []`
  * the server polls its own generator at 5 ms and sends the cumulative diff,
    so one chunk is however many tokens landed in that window

The fixtures below are transcribed from `engine/server/openai.py` rather than
imagined. None of this needs a server, a GPU, or the engine installed; if the
engine changes its wire format, these fail and say which shape moved.
"""

from __future__ import annotations

import json

import pytest

from llm.engines import HttpLLMEngine, build_llm

MODEL = "Qwen/Qwen3-0.6B"


def engine_sse(lines):
    """A transport yielding pre-built SSE lines, recording what was sent."""

    def transport(url, payload, headers):
        transport.calls.append({"url": url, "payload": payload, "headers": headers})
        yield from lines
        transport.exhausted = True

    transport.calls = []
    transport.exhausted = False
    return transport


def completion_chunks(deltas, *, usage=None, finish="stop"):
    """The exact shape `/v1/completions` streams: a `text` delta per chunk.

    Transcribed from the `chunk()` closure in `openai.py:completions`, which
    emits `{"choices": [{"index": 0, "text": delta, "finish_reason": None,
    "logprobs": None}]}`, then a terminal chunk with an empty delta and the
    finish reason, then usage if asked, then `[DONE]`.
    """
    lines = []
    for delta in deltas:
        lines.append("data: " + json.dumps({
            "id": "cmpl-abc", "object": "text_completion", "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "text": delta, "finish_reason": None,
                         "logprobs": None}],
        }))
    lines.append("data: " + json.dumps({
        "id": "cmpl-abc", "object": "text_completion", "created": 1, "model": MODEL,
        "choices": [{"index": 0, "text": "", "finish_reason": finish, "logprobs": None}],
    }))
    if usage is not None:
        lines.append("data: " + json.dumps({
            "id": "cmpl-abc", "object": "text_completion", "created": 1,
            "model": MODEL, "choices": [], "usage": usage,
        }))
    lines.append("data: [DONE]")
    return lines


def chat_chunks(deltas, *, usage=None, finish="stop"):
    """The exact shape `/v1/chat/completions` streams, role-first.

    From the `chunk()` closure in `openai.py:chat_completions`: the first
    content chunk also carries `role: assistant`, and the terminal chunk's
    delta is `{}` -- an empty dict, not a dict with an empty string.
    """
    lines = []
    for index, delta in enumerate(deltas):
        body = {"content": delta}
        if index == 0:
            body["role"] = "assistant"
        lines.append("data: " + json.dumps({
            "id": "chatcmpl-abc", "object": "chat.completion.chunk", "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": body, "finish_reason": None}],
        }))
    lines.append("data: " + json.dumps({
        "id": "chatcmpl-abc", "object": "chat.completion.chunk", "created": 1,
        "model": MODEL, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }))
    if usage is not None:
        lines.append("data: " + json.dumps({
            "id": "chatcmpl-abc", "object": "chat.completion.chunk", "created": 1,
            "model": MODEL, "choices": [], "usage": usage,
        }))
    lines.append("data: [DONE]")
    return lines


class TestCompletionsWireFormat:
    """The route the pipeline uses."""

    def test_text_deltas_are_read(self):
        engine = HttpLLMEngine(
            MODEL, transport=engine_sse(completion_chunks(["नम", "स्ते", " दुनिया"])))
        assert list(engine.stream("p")) == ["नम", "स्ते", " दुनिया"]

    def test_the_terminal_empty_delta_is_not_yielded(self):
        """An empty piece would flush a sentence boundary into the TTS buffer."""
        engine = HttpLLMEngine(MODEL, transport=engine_sse(completion_chunks(["ठीक है।"])))
        assert list(engine.stream("p")) == ["ठीक है।"]

    def test_reported_usage_wins_over_counting_chunks(self):
        """The engine reports real token counts; chunks are a lower bound."""
        engine = HttpLLMEngine(MODEL, transport=engine_sse(completion_chunks(
            ["एक ", "दो "], usage={"prompt_tokens": 656, "completion_tokens": 24,
                                   "total_tokens": 680})))
        list(engine.stream("p"))
        metrics = engine.last_metrics
        assert metrics.prompt_tokens == 656
        assert metrics.generated_tokens == 24, "not 2, which is the chunk count"

    def test_chunk_granularity_is_flagged_not_hidden(self):
        """The server polls at 5 ms, so a chunk is not a token. Say so."""
        engine = HttpLLMEngine(MODEL, transport=engine_sse(completion_chunks(["a", "b"])))
        list(engine.stream("p"))
        reported = engine.last_metrics.as_dict()
        assert reported["decode_is_per_chunk"] is True
        assert reported["prefill_is_arrival_time"] is True
        assert reported["chunks"] == 2

    def test_a_chunk_carrying_several_tokens_is_one_gap(self):
        """Honesty check: with no usage, the token count is the chunk count and
        is documented as a lower bound rather than reported as exact."""
        engine = HttpLLMEngine(MODEL, transport=engine_sse(
            completion_chunks(["यह पूरा वाक्य एक ही chunk में आया"])))
        list(engine.stream("p"))
        assert engine.last_metrics.generated_tokens == 1
        assert engine.last_metrics.decode_is_per_chunk is True


class TestChatWireFormat:
    """Not the route the pipeline uses, but the adapter must still parse it."""

    def test_role_first_delta_does_not_break_parsing(self):
        engine = HttpLLMEngine(MODEL, chat=True,
                               transport=engine_sse(chat_chunks(["नम", "स्ते"])))
        assert list(engine.stream("p")) == ["नम", "स्ते"]

    def test_empty_terminal_delta_dict_is_skipped(self):
        engine = HttpLLMEngine(MODEL, chat=True, transport=engine_sse(chat_chunks(["ठीक"])))
        assert list(engine.stream("p")) == ["ठीक"]


class TestDoubleTemplatingIsRefused:
    """The failure this pipeline has already had once, from the other side.

    `llm.prompting` exists because one path rendered Qwen3's template without
    `enable_thinking=False` and the model spent its whole budget inside
    `<think>` with nothing to speak. The chat route would reintroduce exactly
    that: `engine/server/openai.py:_chat_prompt` calls `apply_chat_template`
    with no `enable_thinking`, and the engine's pydantic model ignores unknown
    fields, so there is no way to pass it and no error if you try.
    """

    def test_build_llm_refuses_chat_with_a_client_tokenizer(self):
        with pytest.raises(ValueError, match="double-wraps"):
            build_llm("http", model=MODEL, chat=True, tokenizer=object())

    def test_the_refusal_names_the_route_to_use_instead(self):
        with pytest.raises(ValueError, match="completions"):
            build_llm("http", model=MODEL, chat=True, tokenizer=object())

    def test_server_side_templating_is_possible_but_must_be_explicit(self):
        generator, tokenizer, info = build_llm("http", model=MODEL, chat=True,
                                               tokenizer=False)
        assert tokenizer is None
        assert info["templated_by"] == "server"
        assert generator.chat is True

    def test_the_default_keeps_templating_on_the_client(self):
        generator, _, info = build_llm("http", model=MODEL, tokenizer=False)
        assert generator.chat is False
        assert info["templated_by"] == "client"
        assert generator.endpoint.endswith("/v1/completions")


class TestBargeInReleasesTheSequence:
    def test_stopping_closes_the_transport(self):
        """Closing the response is what the server sees as a disconnect, and
        this engine cancels the request on disconnect -- freeing the paged KV
        blocks instead of decoding tokens nobody will hear. Relying on garbage
        collection to do it would leave that to refcount timing.
        """
        closed = []

        def transport(url, payload, headers):
            try:
                yield from completion_chunks(["एक", "दो", "तीन", "चार"])
            finally:
                closed.append(True)

        engine = HttpLLMEngine(MODEL, transport=transport)
        pieces = []
        for piece in engine.stream("p", should_stop=lambda: len(pieces) >= 2):
            pieces.append(piece)
        assert pieces == ["एक", "दो"]
        assert closed == [True], "the transport was closed, not left to the collector"
        assert engine.last_metrics.stopped_by_caller is True


class TestServerRefusalsReachTheCaller:
    """This engine refuses rather than silently ignoring, and explains itself.

    urllib renders a refusal as `HTTP Error 413: Request Entity Too Large` and
    drops the body, turning a one-line configuration fix into a mystery.
    """

    def test_the_engines_detail_is_captured_in_the_metrics(self):
        class Refusal(OSError):
            code = 413

            def read(self):
                return json.dumps(
                    {"detail": "prompt exceeds the 4096-token server limit"}).encode()

        def transport(url, payload, headers):
            raise Refusal
            yield  # pragma: no cover - generator marker

        engine = HttpLLMEngine(MODEL, transport=transport)
        with pytest.raises(OSError):
            list(engine.stream("p"))
        assert "4096-token server limit" in engine.last_metrics.error
        assert "413" in engine.last_metrics.error

    def test_a_body_less_failure_still_reports_something_useful(self):
        def transport(url, payload, headers):
            raise ConnectionRefusedError("[Errno 61] Connection refused")
            yield  # pragma: no cover - generator marker

        engine = HttpLLMEngine(MODEL, transport=transport)
        with pytest.raises(OSError):
            list(engine.stream("p"))
        assert "Connection refused" in engine.last_metrics.error


class TestStopStrings:
    def test_stop_strings_are_sent_when_configured(self):
        transport = engine_sse(completion_chunks(["x"]))
        engine = HttpLLMEngine(MODEL, stop=["<|im_end|>"], transport=transport)
        list(engine.stream("p"))
        assert transport.calls[0]["payload"]["stop"] == ["<|im_end|>"]

    def test_no_stop_field_is_sent_by_default(self):
        """The engine stops on EOS on its own; an empty `stop` would be noise."""
        transport = engine_sse(completion_chunks(["x"]))
        list(HttpLLMEngine(MODEL, transport=transport).stream("p"))
        assert "stop" not in transport.calls[0]["payload"]
