| Check | Status | Key numbers |
| --- | --- | --- |
| environment | pass | gpu=Tesla T4, vram_gib=14.6, torch=2.11.0+cu128, transformers=5.16.1, git_commit=21ebce752fa78f8c6dc20cb58b115d21aba449be |
| gpu_unit_tests | pass | summary=........                                                                 [100%] |
| fetch_fleurs_clips | pass | hi_clips=5, en_clips=2, hi_seconds=55.9 |
| load_whisper_with_hub_adapter | pass | model=openai/whisper-medium, adapter_dir=/root/.cache/huggingface/hub/models--Hugme6969--whisper-medium-hindi-lora/snapshots/953eb14d6de983ad159491e76742dbfba9f13acb |
| served_model_matches_generate | pass | clips=3, worst_prefix_match=1.0, all_exact=True |
| asr_quality_on_clips | pass | mean_wer=24.6, mean_rtf=0.3 |
| language_detection_base_model | pass | clips=4, mean_detection_ms=48.8, min_probability=0.9 |
| language_detection_adapter | warn | unrestricted_wrong=2, restricted_wrong=0 |
| long_form_chunking | pass | audio_seconds=58.4, chunks=3, wer_vs_concatenated_refs=27.0, rtf=0.2, text=इसे कैमिकल का पीएज कहा जाता है आप लाल गोभी के जूस को इस्तेमाल करके एक संकेतक बना सकते हैं महिलाये या अनुश्रणशा की जाती है कि कोई भी महिलायात रिवास्तविक व्यवाई क्तिति के बावजुग कहती है कि वह व्यवाइत है सड़कों पर ज्यादा धुरक� वजह बहुत ज्यादा कार ओनरशिप है जिससे शरीर में हुई टूटफुट को सुधारने के लिए हेल्थ कैर में नई तक्नीकों के अविश्कार को बढ़ावा मिला है परिषर के किनारें ज्यादातर इमारतें फिर से बनाई गई हैं ताकि परिटो को बेहतर धंग से पता चल पता चले कि वे मूल रूप में कैसी दिखाई देती थी सावधान रहे कि कपड़े को बहुत गर्म न होने दे जो सिकूबनी का कारण बन सकता है या बहुत गर्म होने पर जल सकता है |
| streaming_session | pass | utterances_expected=2, finals=2, partials=11, wer_vs_refs=23.8, wer_vs_offline=12.2 |
| load_llm | pass | llm=Qwen/Qwen3-0.6B, strategy=concurrent |
| llm_prompt_and_decoding | pass | prompt_has_empty_think=True, generated_tokens=65, prefill_ms=348.7, mean_decode_ms=45.5, response=जूस के इस्तेमान के द्वारा के�मिकल प्रणाली के अध्ययन में लाल गोभी बनाना। |
| pipeline_waterfall | pass | strategy=concurrent, mel_ms=14.4, encoder_ms=80.4, asr_decode_ms=1900.2, llm_prefill_ms=105.0 |
| voice_turn_real_tts | pass | sentences=1, chunks=66, audio_bytes=47376, tts_first_chunk_ms=619.7, final_transcript_to_first_llm_token_ms=100.0 |
| barge_in_real_tts | pass | cancelled_after_chunk=1, sentences_planned=1, chunks_synthesised=1, playback_state=cancelled, barge_in_flag=True |
