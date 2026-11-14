# Lila Voice Research Roadmap

This document records future work, not implemented runtime capabilities.
Evaluation and adaptation should be staged and measured independently from the
current Parakeet → transcript → LilaCore path.

## Automatic speech recognition (ASR)

1. **Recognition accuracy and model evaluation** — establish a repeatable
   baseline, word/error rates, latency, resource use, and representative
   utterance coverage before changing models.
2. **Nigerian English and regional accent adaptation** — evaluate performance
   across speakers and regions with consented data; avoid treating one accent
   as the sole reference.
3. **Dialects, languages, and code-switching** — record language transitions
   and evaluate multilingual recognition without silently normalizing the
   speaker's phrasing.
4. **Speaker-specific pronunciation and linguistic patterns** — investigate
   opt-in adaptation for names and pronunciation while keeping the original
   recording and original ASR output.
5. **Technical vocabulary and proper nouns** — measure domain vocabulary,
   product names, acronyms, and local proper-noun recognition.
6. **Background music and noisy-environment robustness** — benchmark noise,
   distance, echo, and music separately; do not assume denoising improves
   every recording.
7. **Accurate labelled speech datasets** — develop a consented, quality-checked
   data collection and manual-label review process.
8. **Speaker adaptation and fine-tuning** — evaluate only after establishing
   dataset sufficiency, privacy controls, and held-out speaker benchmarks.
9. **Streaming and low-latency recognition** — compare latency and accuracy
   against the current completed-utterance baseline.
10. **Voice activity detection and interruption handling** — evaluate endpoint
    detection, barge-in, false starts, and cancellation behavior.
11. **Natural speech generation and expressive TTS** — investigate as a
    separate speech-generation subsystem, not as an ASR feature.
12. **Consistent voice identity and conversational delivery** — assess
    consent, disclosure, identity consistency, and user control for any future
    generated voice.
13. **Native audio understanding and future voice-native models** — evaluate
    only as a distinct model architecture; do not route around the current
    permission-aware tool executor.

## Dataset principles

- Preserve original recordings as immutable source material.
- Keep manually verified transcripts as ground truth; model-generated
  transcripts are hypotheses, never automatically promoted to truth.
- Preserve the original ASR transcript separately from any human correction.
- Record relevant language, accent, recording conditions, device, model
  version, and other metadata when available and consented.
- Use only recordings with appropriate permission and a clear retention
  policy.
- Track dataset versions, train/test splits, model versions, and benchmark
  results.
- Keep personally identifying details and private speech access-controlled;
  do not copy voice datasets into the Git repository.

## Separation of research areas

ASR research concerns input audio and recognition. TTS research concerns
generated output audio. Their datasets, evaluation measures, consent needs,
and runtime interfaces are separate. A future voice-native model would be
evaluated as a separate proposal and must continue to submit any requested
tools through `HarnessExecutor`.

No roadmap item authorizes model downloads, training, data collection,
always-on capture, or a new runtime capability by itself.
