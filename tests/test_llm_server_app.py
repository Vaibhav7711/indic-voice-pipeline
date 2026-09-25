"""The serving factory's configuration and its VRAM arithmetic.

The engine is a separate checkout and needs a GPU, so `create()` cannot run
here. Everything that decides *what* it will allocate can, and that is the
part worth pinning: Qwen3-4B's KV is 144 KiB per cached token, so a pool size
chosen carelessly is the difference between fitting beside Whisper on a T4 and
dying at startup.
"""

from __future__ import annotations

import pytest

from scripts.llm_server_app import (
    DEFAULTS,
    MODEL_GEOMETRY,
    describe,
    kv_bytes_per_token,
    pool_bytes,
    resolve_config,
)


class TestKvArithmetic:
    def test_qwen3_4b_costs_144_kib_per_token(self):
        """36 layers x 2 x 8 KV heads x 128 head dim x 2 bytes. This matches
        the figure in the engine's own T4 planning document; if they disagree,
        one of them is wrong about the checkpoint."""
        assert kv_bytes_per_token("Qwen/Qwen3-4B") == 147_456
        assert kv_bytes_per_token("Qwen/Qwen3-4B") / 1024 == 144.0

    def test_the_engines_default_pool_costs_2_25_gib_at_4b(self):
        """Which is why this repo's factory does not use it for one stream."""
        total = pool_bytes("Qwen/Qwen3-4B", num_blocks=1024, block_size=16)
        assert round(total / 1024**3, 2) == 2.25

    def test_the_default_pool_is_cheaper_and_still_ample(self):
        total = pool_bytes("Qwen/Qwen3-4B", num_blocks=DEFAULTS["num_blocks"],
                           block_size=DEFAULTS["block_size"])
        assert round(total / 1024**3, 3) == 1.125
        # 8192 KV tokens against an <=800-token dialogue prompt plus its reply.
        assert DEFAULTS["num_blocks"] * DEFAULTS["block_size"] == 8192

    def test_0_6b_is_cheaper_per_token_than_4b(self):
        assert kv_bytes_per_token("Qwen/Qwen3-0.6B") < kv_bytes_per_token("Qwen/Qwen3-4B")

    def test_an_unknown_checkpoint_reports_none_not_a_guess(self):
        """A made-up VRAM figure is worse than no figure: this one is used to
        decide whether a model fits beside Whisper, so it gets acted on."""
        assert kv_bytes_per_token("some/unknown-model") is None
        assert pool_bytes("some/unknown-model", num_blocks=512, block_size=16) is None

    def test_bf16_and_fp16_cost_the_same(self):
        assert (kv_bytes_per_token("Qwen/Qwen3-4B", dtype_bytes=2)
                == kv_bytes_per_token("Qwen/Qwen3-4B", dtype_bytes=2))

    def test_geometry_is_recorded_for_every_served_checkpoint(self):
        for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"):
            assert model in MODEL_GEOMETRY


class TestDescribe:
    def test_it_reports_the_resident_estimate(self):
        report = describe({"model_name": "Qwen/Qwen3-4B", "num_blocks": 512,
                           "block_size": 16})
        assert report["kv_tokens"] == 8192
        assert report["kv_pool_gib"] == 1.125
        assert report["weights_gib_fp16"] == 7.5
        assert report["resident_gib_estimate"] == 8.625

    def test_an_unknown_model_reports_none_throughout(self):
        report = describe({"model_name": "x/y", "num_blocks": 512, "block_size": 16})
        assert report["kv_pool_gib"] is None
        assert report["resident_gib_estimate"] is None
        assert report["kv_tokens"] == 8192, "block arithmetic needs no geometry"

    def test_the_note_says_what_is_not_counted(self):
        report = describe({"model_name": "Qwen/Qwen3-4B", "num_blocks": 512,
                           "block_size": 16})
        assert "CUDA context" in report["note"]


class TestConfigResolution:
    def test_the_default_is_qwen3_4b(self):
        assert resolve_config({})["model_name"] == "Qwen/Qwen3-4B"

    def test_the_environment_overrides_every_knob(self):
        config = resolve_config({
            "LLM_SERVER_MODEL": "Qwen/Qwen3-0.6B",
            "LLM_SERVER_NUM_BLOCKS": "256",
            "LLM_SERVER_BLOCK_SIZE": "256",
            "LLM_SERVER_MAX_ACTIVE": "4",
            "LLM_SERVER_GRAPH_BUCKETS": "1,2,4",
            "LLM_SERVER_DTYPE": "bfloat16",
            "LLM_SERVER_MAX_PROMPT_TOKENS": "2048",
        })
        assert config == {
            "model_name": "Qwen/Qwen3-0.6B", "num_blocks": 256, "block_size": 256,
            "max_active": 4, "graph_buckets": (1, 2, 4), "dtype": "bfloat16",
            "max_prompt_tokens": 2048,
        }

    def test_an_empty_value_falls_back_rather_than_failing(self):
        """Shell expansion of an unset variable is the empty string, and that
        should mean "not set", not "invalid"."""
        config = resolve_config({"LLM_SERVER_MODEL": "", "LLM_SERVER_NUM_BLOCKS": ""})
        assert config["model_name"] == DEFAULTS["model"]
        assert config["num_blocks"] == DEFAULTS["num_blocks"]

    def test_a_non_integer_is_refused(self):
        """Silently falling back would serve a different configuration than
        the one recorded beside the numbers."""
        with pytest.raises(ValueError, match="not an integer"):
            resolve_config({"LLM_SERVER_NUM_BLOCKS": "lots"})

    def test_a_non_positive_size_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            resolve_config({"LLM_SERVER_BLOCK_SIZE": "0"})

    def test_a_malformed_bucket_list_is_refused(self):
        with pytest.raises(ValueError, match="comma-separated"):
            resolve_config({"LLM_SERVER_GRAPH_BUCKETS": "1;2"})

    def test_a_bucket_list_with_a_zero_is_refused(self):
        with pytest.raises(ValueError, match="positive integers"):
            resolve_config({"LLM_SERVER_GRAPH_BUCKETS": "0,2"})

    def test_attention_backends_are_omitted_unless_named(self):
        """The engine picks by its own measured per-architecture policy. Passing
        None would override that with nothing; omitting the key lets it
        choose."""
        assert "decode_attention" not in resolve_config({})
        config = resolve_config({"LLM_SERVER_DECODE_ATTENTION": "per_head"})
        assert config["decode_attention"] == "per_head"

    def test_graph_buckets_default_to_a_single_stream(self):
        """Each captured bucket costs memory, and 4B has little to spare."""
        assert resolve_config({})["graph_buckets"] == (1, 2)
        assert resolve_config({})["max_active"] == 2
