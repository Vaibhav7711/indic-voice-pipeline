"""Gradio demo: record audio → transcript → answer → speech output."""

from __future__ import annotations

import numpy as np
import gradio as gr

_pipe = None
_tts = None


def _load():
    global _pipe, _tts
    if _pipe:
        return _pipe, _tts

    from asr.explicit import load_whisper
    from llm import load_llm
    from pipeline import VoicePipeline
    from tts import TTSSynthesizer

    w = load_whisper("openai/whisper-small")
    l = load_llm("Qwen/Qwen3-0.6B")
    _pipe = VoicePipeline(w, l)
    _tts = TTSSynthesizer(language="hi")
    return _pipe, _tts


def process(audio, language, max_tokens, do_tts):
    if audio is None:
        return "", "", "", None

    pipe, tts = _load()
    sr, wav = audio
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    lang = language if language != "auto" else None
    result = pipe.run_array(wav, sr, language=lang, llm_max_tokens=max_tokens)

    m = result.metrics
    table = (
        f"| Stage | ms |\n|---|---:|\n"
        f"| Mel | {m.asr.mel_extraction_ms:.0f} |\n"
        f"| Encoder | {m.asr.encoder_ms:.0f} |\n"
        f"| ASR decode | {m.asr.total_decode_ms:.0f} |\n"
        f"| LLM prefill | {m.llm_prefill_ms:.0f} |\n"
        f"| LLM decode | {sum(m.llm_decode_ms):.0f} |\n"
        f"| **Total** | **{m.total_pipeline_ms:.0f}** |\n\n"
        f"Peak VRAM: {m.peak_allocated_bytes/(1024**3):.2f} GiB"
    )

    audio_out = None
    if do_tts and result.answer.strip():
        import tempfile
        path = tempfile.mktemp(suffix=".mp3")
        tts.language = lang or "hi"
        r = tts.synthesize(result.answer, path)
        audio_out = path
        table += f"\nTTS: {r.synthesis_ms:.0f} ms"

    return result.transcript, result.answer, table, audio_out


demo = gr.Blocks(title="Indic Voice Pipeline")
with demo:
    gr.Markdown("# Indic Voice Pipeline\nAudio → Whisper → LLM → TTS")
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy")
            lang = gr.Dropdown(["auto", "hi", "en", "te"], value="hi", label="Language")
            tokens = gr.Slider(16, 256, 64, step=16, label="Max LLM tokens")
            do_tts = gr.Checkbox(True, label="TTS output")
            btn = gr.Button("Run", variant="primary")
        with gr.Column():
            txt_out = gr.Textbox(label="Transcript")
            ans_out = gr.Textbox(label="Answer")
            met_out = gr.Markdown(label="Latency")
            aud_out = gr.Audio(label="Speech", type="filepath")
    btn.click(process, [audio_in, lang, tokens, do_tts], [txt_out, ans_out, met_out, aud_out])

if __name__ == "__main__":
    demo.launch(share=True)
