"""Explicit Whisper encoder forward pass with CUDA-event timing.

Non-autoregressive: processes full mel input in one pass, producing
(batch, 1500, 768) hidden states for the decoder to cross-attend to.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from asr.explicit.timing import Timer


@dataclass
class EncoderResult:
    encoder_outputs: BaseModelOutput
    encoder_ms: float
    hidden_size: int
    sequence_length: int


class WhisperEncoder:
    def __init__(self, model: WhisperForConditionalGeneration, device: torch.device):
        self.encoder = model.get_encoder()
        self.device = device

    @torch.inference_mode()
    def forward(self, input_features: torch.Tensor) -> EncoderResult:
        """Run encoder on mel features (batch, 80, 3000) → hidden states."""
        with Timer(self.device) as timer:
            encoder_outputs = self.encoder(input_features, return_dict=True)

        h = encoder_outputs.last_hidden_state
        return EncoderResult(
            encoder_outputs=encoder_outputs,
            encoder_ms=timer.ms,
            hidden_size=h.shape[-1],
            sequence_length=h.shape[1],
        )
