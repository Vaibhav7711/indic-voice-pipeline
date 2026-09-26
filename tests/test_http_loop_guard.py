"""The repetition guard that serving over HTTP had dropped.

`LLMRunner` stops degenerate greedy output with an n-gram check on token ids
and reports `stopped_on_repetition`. Routing the turn through an HTTP engine
lost that silently: the server owns the decode, so nothing downstream could
stop it.

A measured Colab sweep caught the cost. Given "मुझे एक छोटी कहानी सुनाओ।" the
served arm produced `बर्फ के` thirteen times and ran to its 128-token cap,
while the explicit arm answered the same prompt coherently — same checkpoint,
same prompt, both greedy. The parity gate saw the same thing on the same
prompt.

The guard here works on characters, because that is all this side receives. It
is weaker than a token-level guard and is not a substitute for one in the
engine; it stops the pathological case from being spoken and from burning GPU
time, since abandoning the stream closes the socket and the engine cancels the
request.
"""

from __future__ import annotations

from llm.engines import HttpLLMEngine
from tests.test_serving_end_to_end import base_url

#: Verbatim from the served arm of the measured run.
OBSERVED_LOOP = "एक छोटी कहानी आपके साथ सुनूंगी। " + "एक बर्फ के " * 13

#: The explicit arm's answer to the same prompt, which must not be stopped.
OBSERVED_COHERENT = (
    "एक छोटी कहानी: एक दिन एक बर्फ लेकर जाने वाला बच्चा अपने घर से बाहर गया। "
    "उसके पास एक बर्फ थी, और उसने उसे खींचकर बर्फ के ऊपर रख दिया।"
)


WINDOW = 48


class TestDetection:
    def test_the_observed_loop_is_detected(self):
        assert HttpLLMEngine._looping(OBSERVED_LOOP, WINDOW) is True

    def test_the_observed_coherent_answer_is_not(self):
        """It repeats the word बर्फ three times without looping. A guard that
        fired here would cut a legitimate answer mid-sentence."""
        assert HttpLLMEngine._looping(OBSERVED_COHERENT, WINDOW) is False

    def test_the_period_is_searched_not_assumed(self):
        """The observed loop has an 11-character period. Comparing two fixed
        halves of the window only fires when the window is a multiple of the
        period, so a 24-character window misses it entirely -- which the first
        version of this guard did."""
        block = "एक बर्फ के "
        assert len(block) == 11
        assert HttpLLMEngine._looping(block * 6, WINDOW) is True

    def test_text_shorter_than_the_window_is_never_a_loop(self):
        assert HttpLLMEngine._looping("नमस्ते", WINDOW) is False
        assert HttpLLMEngine._looping("x" * (WINDOW - 1), WINDOW) is False

    def test_two_repeats_are_not_enough(self):
        """Ordinary prose repeats a short phrase twice; cutting a legitimate
        answer mid-sentence is worse than speaking a loop.

        The filler is real prose, not a run of one character: 30 identical
        characters is itself degenerate output and the guard rightly fires on
        it, which an earlier version of this test mistook for a false positive.
        """
        text = "hello world hello world and then the story moved on elsewhere"
        assert HttpLLMEngine._looping(text, WINDOW) is False

    def test_three_repeats_are(self):
        assert HttpLLMEngine._looping("hello world " * 4, WINDOW) is True

    def test_a_short_period_below_the_minimum_is_ignored(self):
        """Period 1 would fire on any run of one character."""
        assert HttpLLMEngine._looping("aaaa" * 20, WINDOW, min_period=4) is True
        assert HttpLLMEngine._looping("ab" * 40, WINDOW, min_period=4) is True
        assert HttpLLMEngine._looping("x" * 100, WINDOW, min_period=60) is False

    def test_zero_disables_it(self):
        assert HttpLLMEngine._looping(OBSERVED_LOOP, 0) is False

    def test_fewer_than_two_repeats_is_refused(self):
        assert HttpLLMEngine._looping(OBSERVED_LOOP, WINDOW, repeats=1) is False

    def test_it_looks_only_at_the_tail(self):
        """So a phrase repeated early does not stop a response that has moved
        on, and the check costs the same per chunk however long the response
        gets."""
        moved_on = ("abab" * 4
                    + " और उसके बाद कहानी आगे बढ़ी और बच्चा घर वापस चला गया।")
        assert HttpLLMEngine._looping(moved_on, WINDOW) is False

    def test_a_long_run_of_one_character_is_a_loop(self):
        """Not a false positive: 48 identical characters is not speech, and
        handing it to TTS would be worse than cutting it."""
        assert HttpLLMEngine._looping("x" * 60, WINDOW) is True


class TestItStopsARealStream:
    def test_a_looping_stream_is_cut_and_recorded(self, engine_server):
        engine_server.script = ["एक छोटी कहानी: "] + ["एक बर्फ के "] * 40
        engine_server.usage = None
        engine = HttpLLMEngine("Qwen/Qwen3-1.7B", base_url=base_url(engine_server),
                               loop_guard_chars=48)
        text = "".join(engine.stream("कहानी सुनाओ"))
        assert engine.last_metrics.stopped_on_repetition is True
        assert engine.last_metrics.stopped_by_caller is True
        assert len(text) < len("".join(engine_server.script)), "it was cut short"

    def test_the_server_sees_the_disconnect(self, engine_server):
        """Which is what makes the engine cancel the request, so the loop stops
        costing GPU time rather than only going unspoken."""
        engine_server.script = ["एक बर्फ के "] * 60
        engine_server.usage = None
        engine = HttpLLMEngine("Qwen/Qwen3-1.7B", base_url=base_url(engine_server),
                               loop_guard_chars=48)
        list(engine.stream("p"))
        assert engine_server.disconnected.wait(timeout=5)

    def test_a_coherent_stream_runs_to_completion(self, engine_server):
        engine_server.script = ["नमस्ते।", " मैं ठीक हूँ।", " आप कैसे हैं?"]
        engine = HttpLLMEngine("Qwen/Qwen3-1.7B", base_url=base_url(engine_server),
                               loop_guard_chars=48)
        text = "".join(engine.stream("p"))
        assert text == "नमस्ते। मैं ठीक हूँ। आप कैसे हैं?"
        assert engine.last_metrics.stopped_on_repetition is False
        assert engine.last_metrics.stopped_by_caller is False

    def test_the_guard_can_be_turned_off(self, engine_server):
        engine_server.script = ["एक बर्फ के "] * 10
        engine_server.usage = None
        engine = HttpLLMEngine("Qwen/Qwen3-1.7B", base_url=base_url(engine_server),
                               loop_guard_chars=0)
        text = "".join(engine.stream("p"))
        assert engine.last_metrics.stopped_on_repetition is False
        assert text == "एक बर्फ के " * 10

    def test_barge_in_still_takes_precedence(self, engine_server):
        """A caller stopping is not a repetition stop, and the metrics must not
        claim it was."""
        engine_server.script = ["एक बर्फ के "] * 40
        engine_server.usage = None
        engine = HttpLLMEngine("Qwen/Qwen3-1.7B", base_url=base_url(engine_server),
                               loop_guard_chars=48)
        pieces: list[str] = []
        for piece in engine.stream("p", should_stop=lambda: len(pieces) >= 1):
            pieces.append(piece)
        assert len(pieces) == 1
        assert engine.last_metrics.stopped_by_caller is True
        assert engine.last_metrics.stopped_on_repetition is False
