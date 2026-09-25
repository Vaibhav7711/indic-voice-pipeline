"""End to end over a real socket, with no model and no GPU.

Every other test of the serving seam injects a transport. That leaves the one
piece that will actually be running in production untested: `_urllib_sse` --
the POST, the SSE line framing, the connection close that the engine reads as
a disconnect, and the error body that carries the engine's own explanation.

So this stands up a real HTTP server on loopback, speaking the exact wire
format `engine/server/openai.py` emits, and drives a real `VoiceTurn` through
a real `HttpLLMEngine` and a real `SentenceBuffer` against it. The only fakes
are the model behind the server and the synthesizer, neither of which this is
testing.

What it covers that nothing else does:

  * a turn producing audio with the LLM served over HTTP
  * barge-in mid-response closing the socket, which is how the engine learns
    to cancel the request and free its paged KV blocks
  * dialogue history surviving across HTTP-served turns
  * a server refusal reaching the caller with the engine's `detail` intact
  * the prompt arriving byte-identical, so `enable_thinking=False` is really
    in what the server receives

Runs in about a second on a laptop, which is the point: the wiring should not
need a T4 to be proven wrong.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent import Conversation, VoiceTurn
from agent.playback import BufferSink
from llm.engines import HttpLLMEngine


def completion_chunk(delta: str, finish: str | None = None) -> str:
    """One chunk in the shape `/v1/completions` streams: a `choices[0].text`
    delta, with the terminal chunk carrying an empty delta and a finish
    reason. A usage chunk (`choices: []`) and `[DONE]` follow."""
    return "data: " + json.dumps({
        "id": "cmpl-test", "object": "text_completion", "created": 1,
        "model": "Qwen/Qwen3-0.6B",
        "choices": [{"index": 0, "text": delta, "finish_reason": finish,
                     "logprobs": None}],
    }) + "\n\n"


class EngineHandler(BaseHTTPRequestHandler):
    """A stand-in for the serving engine, deliberately unclever.

    `server.script` is the text to stream, `server.refuse` an optional
    (status, detail) to answer with instead. Every request's payload is
    recorded on the server so a test can assert what was actually sent over
    the wire rather than what the adapter meant to send.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):          # silence the default stderr spam
        pass

    def do_POST(self):
        import time

        length = int(self.headers.get("content-length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.payloads.append(payload)

        if self.server.refuse is not None:
            status, detail = self.server.refuse
            body = json.dumps({"detail": detail}).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()

        try:
            for piece in self.server.script:
                if self.server.delay:
                    # A real engine's inter-token latency. Without it the whole
                    # response lands in a few milliseconds and a barge-in test
                    # races the turn instead of interrupting it.
                    time.sleep(self.server.delay)
                self._chunk(completion_chunk(piece))
            self._chunk(completion_chunk("", "stop"))
            if self.server.usage:
                self._chunk("data: " + json.dumps(
                    {"choices": [], "usage": self.server.usage}) + "\n\n")
            self._chunk("data: [DONE]\n\n")
            self._chunk("")
        except (BrokenPipeError, ConnectionResetError):
            # What the real engine sees on barge-in, and where it cancels the
            # request instead of decoding into a closed socket.
            self.server.disconnected.set()

    def _chunk(self, text: str) -> None:
        data = text.encode("utf-8")
        self.wfile.write(f"{len(data):X}\r\n".encode())
        self.wfile.write(data + b"\r\n")
        self.wfile.flush()


@pytest.fixture
def engine_server():
    """A live server on an ephemeral loopback port, torn down after the test."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), EngineHandler)
    server.script = ["नमस्ते", "। ", "मैं ठीक ", "हूँ।"]
    server.usage = {"prompt_tokens": 128, "completion_tokens": 9, "total_tokens": 137}
    server.refuse = None
    server.delay = 0.0
    server.payloads = []
    server.disconnected = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def base_url(server) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/v1"


def interrupt_when_speaking(turn, timeout_s: float = 5.0) -> bool:
    """Interrupt as soon as playback exists, which is what the microphone
    callback does on a speech onset while the agent is talking."""
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if turn.playback is not None:
            turn.interrupt()
            return True
        time.sleep(0.001)
    return False


class FakeTTS:
    """Shaped like tts.EdgeStreamingSynthesizer, minus the network."""

    streaming = True
    format = "mp3"

    def __init__(self):
        self.sentences: list[str] = []

    def stream(self, text: str):
        self.sentences.append(text)
        yield f"audio:{text[:6]}".encode()


class TestRealSocket:
    def test_the_real_transport_streams_over_a_real_connection(self, engine_server):
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        pieces = list(engine.stream("RENDERED PROMPT", max_new_tokens=32))
        assert "".join(pieces) == "नमस्ते। मैं ठीक हूँ।"

    def test_the_prompt_arrives_byte_identical(self, engine_server):
        """The turn renders Qwen3's template with enable_thinking=False. If
        anything re-encoded or re-wrapped it in transit, the served model is
        not answering the prompt this pipeline built."""
        prompt = "<|im_start|>system\nनिर्देश<|im_end|>\n<|im_start|>user\nनमस्ते<|im_end|>\n"
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        list(engine.stream(prompt))
        sent = engine_server.payloads[0]
        assert sent["prompt"] == prompt
        assert sent["temperature"] == 0.0, "greedy, or it is not the same model"
        assert sent["stream"] is True
        assert "messages" not in sent, "completions route, so no server templating"

    def test_reported_usage_is_read_off_the_wire(self, engine_server):
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        list(engine.stream("p"))
        metrics = engine.last_metrics
        assert metrics.prompt_tokens == 128
        assert metrics.generated_tokens == 9, "the server's count, not the chunk count"
        assert metrics.chunks == 4
        assert metrics.prefill_ms is not None and metrics.prefill_ms > 0

    def test_a_refusal_carries_the_engines_own_detail(self, engine_server):
        """urllib renders this as "HTTP Error 413: Request Entity Too Large"
        and drops the body, which is where the fix is described."""
        engine_server.refuse = (413, "prompt exceeds the 4096-token server limit")
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        with pytest.raises(OSError):
            list(engine.stream("p" * 100))
        assert "4096-token server limit" in engine.last_metrics.error

    def test_an_unreachable_endpoint_fails_loudly(self, engine_server):
        """Nothing listening on the port: the metrics still exist and say so,
        rather than a turn silently producing no audio."""
        host, port = engine_server.server_address[:2]
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B",
                               base_url=f"http://{host}:{port + 1}/v1")
        with pytest.raises(OSError):
            list(engine.stream("p"))
        assert engine.last_metrics.error
        assert engine.last_metrics.total_ms > 0

    def test_probe_round_trips(self, engine_server):
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        info = engine.probe()
        assert info["reachable"] is True
        assert info["reported_usage"] is True
        assert info["endpoint"].endswith("/v1/completions")


class TestTurnOverHttp:
    """The whole turn, with the LLM on the far side of a socket."""

    def test_a_turn_produces_audio_and_records_where_the_time_went(self, engine_server):
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        synth = FakeTTS()
        turn = VoiceTurn(engine, synth, sink_factory=BufferSink)
        result = turn.run("आप कैसे हैं?")

        assert result.error is None
        assert result.response == "नमस्ते। मैं ठीक हूँ।"
        assert synth.sentences, "nothing reached the synthesizer"
        metrics = result.metrics.as_dict()
        assert metrics["llm_streaming"] is True
        assert metrics["total_turn_ms"] > 0
        assert metrics["final_transcript_to_first_llm_token_ms"] > 0
        assert metrics["first_llm_token_to_playback_start_ms"] > 0
        # There was no speech in this turn, so the segment from speech end to
        # a committed transcript does not exist -- and the perceived latency
        # that sums it stays None rather than reporting the remainder as if it
        # were the whole. `scripts/latency_ab.py` reports transcript-to-first-
        # audio for exactly this reason.
        assert metrics["response_latency_ms"] is None
        assert metrics["speech_end_to_final_transcript_ms"] is None

    def test_supplying_the_asr_segment_completes_the_perceived_latency(self,
                                                                       engine_server):
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        turn = VoiceTurn(engine, FakeTTS(), sink_factory=BufferSink)
        result = turn.run("आप कैसे हैं?", speech_end_to_transcript_ms=660.8)
        metrics = result.metrics.as_dict()
        assert metrics["response_latency_ms"] > 660.8

    def test_sentences_are_synthesized_as_they_complete(self, engine_server):
        """The first danda must reach TTS before the rest of the response has
        arrived; otherwise the sentence-level pipelining is decorative."""
        engine_server.script = ["पहला वाक्य", "। ", "दूसरा वाक्य", "।"]
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        synth = FakeTTS()
        VoiceTurn(engine, synth, sink_factory=BufferSink).run("बोलो")
        assert len(synth.sentences) == 2
        assert synth.sentences[0].startswith("पहला")

    def test_a_short_unit_cap_cuts_at_a_clause(self, engine_server):
        """The knob the A/B sweeps, exercised over a real connection.

        The cap only takes effect where there is a clause boundary inside the
        window to cut at -- it will not split mid-word. A long sentence with
        no comma and no clause word stays whole at any cap, which is the
        buffer behaving correctly and is worth knowing before reading an A/B
        arm that shows no change.
        """
        engine_server.script = ["दिल्ली बड़ी है, ",
                                "और वहाँ लोग रहते हैं, ",
                                "जो भाषाएँ बोलते हैं।"]
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        long_units = FakeTTS()
        VoiceTurn(engine, long_units, sink_factory=BufferSink,
                  max_unit_chars=200).run("बोलो")
        short_units = FakeTTS()
        VoiceTurn(engine, short_units, sink_factory=BufferSink,
                  max_unit_chars=30).run("बोलो")
        assert len(short_units.sentences) > len(long_units.sentences), (
            f"long={long_units.sentences} short={short_units.sentences}"
        )
        assert len(short_units.sentences[0]) < len(long_units.sentences[0])

    def test_a_sentence_with_no_clause_boundary_is_not_split_mid_word(self,
                                                                     engine_server):
        engine_server.script = ["एकबहुतलंबाशब्दजिसमेंकोईविरामनहींहैबिलकुल।"]
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        synth = FakeTTS()
        VoiceTurn(engine, synth, sink_factory=BufferSink,
                  max_unit_chars=10).run("बोलो")
        assert len(synth.sentences) == 1

    def test_barge_in_closes_the_socket_so_the_engine_can_cancel(self, engine_server):
        """The whole point of closing rather than abandoning the response: a
        disconnect is how the engine learns to stop decoding and free the
        sequence's KV blocks."""
        engine_server.script = ["एक "] * 200
        engine_server.delay = 0.01      # ~2 s of response to interrupt into
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        synth = FakeTTS()
        turn = VoiceTurn(engine, synth, sink_factory=BufferSink)

        # Interrupt as soon as the turn starts speaking, from another thread,
        # which is what the microphone callback does on a speech onset.
        watcher = threading.Thread(target=interrupt_when_speaking, args=(turn,),
                                   daemon=True)
        watcher.start()
        result = turn.run("रुको")
        watcher.join(timeout=10)

        assert result.metrics.barge_in is True
        assert engine.last_metrics.stopped_by_caller is True
        assert engine_server.disconnected.wait(timeout=5), (
            "the server never saw a disconnect, so a real engine would have "
            "kept decoding tokens nobody hears"
        )

    def test_history_accumulates_across_http_served_turns(self, engine_server):
        """Dialogue memory is what makes prefill grow, which is what the
        history-budget arm is about. It has to actually reach the wire."""
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        turn = VoiceTurn(engine, FakeTTS(), sink_factory=BufferSink,
                         conversation=Conversation(max_turns=6))
        turn.run("पहला सवाल")
        turn.run("दूसरा सवाल")
        first, second = (p["prompt"] for p in engine_server.payloads[:2])
        assert "पहला सवाल" in first
        assert "पहला सवाल" in second, "the first exchange did not reach the second prompt"
        assert "दूसरा सवाल" in second
        assert len(second) > len(first), "this growth is the prefill cost being measured"

    def test_only_what_was_heard_is_remembered_after_a_barge_in(self, engine_server):
        """A barged-in turn must not leave the model believing it delivered
        sentences the user never heard."""
        engine_server.script = ["एक "] * 200
        engine_server.delay = 0.01      # ~2 s of response to interrupt into
        engine = HttpLLMEngine("Qwen/Qwen3-0.6B", base_url=base_url(engine_server))
        conversation = Conversation(max_turns=6)
        turn = VoiceTurn(engine, FakeTTS(), sink_factory=BufferSink,
                         conversation=conversation)

        watcher = threading.Thread(target=interrupt_when_speaking, args=(turn,),
                                   daemon=True)
        watcher.start()
        result = turn.run("रुको")
        watcher.join(timeout=10)

        assert result.metrics.barge_in is True
        remembered = "".join(message["content"] for message in conversation.messages("x")
                             if message["role"] == "assistant")
        assert len(remembered) <= len(result.response) + 4, (
            "history claims more was spoken than was generated"
        )
