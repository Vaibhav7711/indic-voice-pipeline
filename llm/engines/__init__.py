"""Alternative LLM backends that satisfy the turn's generator contract.

The turn depends on a structural contract, not on ``LLMRunner``, so a faster
engine drops in without touching the agent. Whatever serves must still match
the explicit runner's output before it is trusted — see the correctness policy
in the README.
"""

from llm.engines.factory import ASR_ENGINES, LLM_ENGINES, build_asr, build_llm
from llm.engines.http_engine import HttpEngineMetrics, HttpEngineResult, HttpLLMEngine

__all__ = ["HttpLLMEngine", "HttpEngineMetrics", "HttpEngineResult",
           "build_asr", "build_llm", "ASR_ENGINES", "LLM_ENGINES"]
