| Check | Status | Key numbers |
| --- | --- | --- |
| environment | pass | gpu=Tesla T4, vram_gib=14.6, torch=2.11.0+cu128, transformers=5.16.1, git_commit=31a842f9ab890e3fd747e3149d147de24b1f9be3 |
| gpu_unit_tests | skip | --skip-unit-tests |
| fetch_fleurs_clips | pass | hi_clips=5, en_clips=2, hi_seconds=55.9 |
| load_whisper_with_hub_adapter | pass | model=openai/whisper-large-v3-turbo, adapter_dir=/content/v2-final/best |
| served_model_matches_generate | pass | clips=3, worst_prefix_match=1.0, all_exact=True |
| asr_quality_on_clips | pass | mean_wer=22.1, mean_rtf=0.1 |
| language_detection_base_model | warn | clips=4, mean_detection_ms=11.5, min_probability=0.9 |
| language_detection_adapter | pass | unrestricted_wrong=0, restricted_wrong=0 |
| long_form_chunking | pass | audio_seconds=58.4, chunks=3, wer_vs_concatenated_refs=29.4, rtf=0.1, text=इसे केमिकल का पी-एच कहा जाता है आप लाल गोभी के जूस को इस्तेमाल करके एक संकेतक बना सकते हैं महिलाय या अनुशनशा की जाती है कि कोई भी महिलायात्री वास्तविक विवाहित स्थिति के बावजुत कहती है कि वह विवाहित है सड़कों पर ज्यादा धुरुकटना की वज़ा बहुत ज्यादाकार ऑनर्शिप है जिससे शरीर में वजह बहुत ज्यादाकार ऑनर्शिप है जिससे शरीर में हुई टूटफुट को सुधारने के लिए हेल्थकेर में नई तकनीकों के अविश्कार को बढ़ावा मिला है परिसर के किनारे ज्यादातर इमारतें फिर से बनाई गई हैं ताकि परिटों को बहतर ध्रंक से पता चले कि वे मूल रूप में कैसी दिखाई देती थी सावधान रहे कि कपड़े को बहुत गर्म न होने दे जो सिकूरने का कारण बन सकता है या बहुत गर्म होने पर जल सकता है |
| streaming_session | pass | utterances_expected=2, finals=2, partials=11, wer_vs_refs=23.8, wer_vs_offline=7.1 |
| ct2_matches_explicit | pass | clips=3, tokens_equal=1, mean_wer_vs_explicit=1.6, speedup=1.3, compute_type=int8_float16 |
| load_llm | pass | llm=Qwen/Qwen3-0.6B, strategy=concurrent |
| llm_prompt_and_decoding | pass | prompt_has_empty_think=True, generated_tokens=58, prefill_ms=367.6, mean_decode_ms=43.5, response=इसे अमार गोभी के जूस के इस्तेमाल के साथ एक संकेतक बना सकते हैं। |
| llm_compiled_matches_eager | pass | tokens_match=True, eager_ms_per_token=42.4, compiled_ms_per_token=49.7, speedup=0.9, eager_prefill_ms=119.1 |
| pipeline_waterfall | pass | strategy=concurrent, mel_ms=28.2, encoder_ms=167.1, asr_decode_ms=419.1, llm_prefill_ms=112.1 |
| voice_turn_real_tts | skip | --skip-network |
| barge_in_real_tts | skip | --skip-network |
