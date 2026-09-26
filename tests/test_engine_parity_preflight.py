"""The VRAM preflight: the parity gate loads a second copy of the weights.

A real Colab run died at 87% of loading the reference copy of Qwen3-4B while
the server already held one. Two 7.5 GiB copies do not fit a 15 GiB T4, and the
serving VRAM budget never accounted for it — it counted the server's copy and
Whisper, not a reference copy in a third process.

The failure gave no explanation: a killed process partway through a progress
bar. The remedy — a smaller checkpoint, or generating the reference where the
server is not running — is not guessable from that, so it is checked before the
load and stated.
"""

from __future__ import annotations

from scripts.engine_parity import free_vram_gib, vram_preflight

T4_GIB = 15.0


class TestPreflight:
    def test_two_4b_copies_do_not_fit_a_t4(self):
        """The measured failure. The server holds ~8.6 GiB (weights plus pool),
        leaving about 6 GiB, and a reference copy needs ~8.1."""
        verdict = vram_preflight("Qwen/Qwen3-4B", free_gib=T4_GIB - 8.6)
        assert verdict["fits"] is False
        assert verdict["needed_gib"] == 8.1

    def test_1_7b_fits_twice_over(self):
        """The server holds ~4.7 GiB, leaving ~10.3 against a ~4.4 GiB need."""
        verdict = vram_preflight("Qwen/Qwen3-1.7B", free_gib=T4_GIB - 4.7)
        assert verdict["fits"] is True
        assert verdict["needed_gib"] == 4.4

    def test_0_6b_fits_easily(self):
        assert vram_preflight("Qwen/Qwen3-0.6B", free_gib=T4_GIB - 2.1)["fits"] is True

    def test_the_need_includes_headroom_beyond_the_weights(self):
        """Weights alone would pass at exactly their own size and then fail on
        activations and the CUDA context."""
        verdict = vram_preflight("Qwen/Qwen3-1.7B", free_gib=3.9)
        assert verdict["reference_copy_gib"] == 3.8
        assert verdict["needed_gib"] > 3.9
        assert verdict["fits"] is False

    def test_unreadable_free_vram_is_unknown_not_a_pass(self):
        """Off-GPU or on error. `None` means undecided; treating it as a pass
        would put the check back where it was, and as a failure would block a
        machine it knows nothing about."""
        verdict = vram_preflight("Qwen/Qwen3-1.7B", free_gib=None)
        assert verdict["fits"] is None
        assert verdict["free_gib"] is None

    def test_an_unknown_checkpoint_is_unknown_not_a_guess(self):
        verdict = vram_preflight("some/unlisted-model", free_gib=10.0)
        assert verdict["fits"] is None
        assert verdict["reference_copy_gib"] is None
        assert verdict["needed_gib"] is None

    def test_the_verdict_carries_the_numbers_it_judged_on(self):
        """So a refusal can be argued with rather than only obeyed."""
        verdict = vram_preflight("Qwen/Qwen3-4B", free_gib=6.4)
        assert verdict["model"] == "Qwen/Qwen3-4B"
        assert verdict["free_gib"] == 6.4
        assert verdict["reference_copy_gib"] == 7.5


def test_free_vram_is_none_without_cuda():
    """This suite runs on a laptop, so the CPU path is the one exercised here.
    It must return None rather than raising or reporting zero."""
    value = free_vram_gib()
    assert value is None or value > 0
