# Notes for explaining the project

## The honest one-liner
- "I took OpenAI's **pretrained** Whisper model (not trained by me) and built an app that re-implements each
  stage of it, proves each stage matches the library, and tests how well the model works."
- Our own work: instrumentation, verification, experiments (WER breakdown, noise, model size, beam search).

## Stage by stage (what / why / numbers to remember)

**1. Audio**
- Decoded and resampled to **16 kHz** mono. Speech information sits below 8 kHz (Nyquist for 16 kHz).
- Fixed **30 s window**: shorter audio is zero-padded, so the encoder always sees the same size.

**2. Log-Mel features**
- STFT: **25 ms** Hann window (400 samples), **10 ms** hop (160) -> 201 frequency bins, 3000 frames per 30 s.
- Power = |STFT|^2. Mel filterbank: **80** triangular filters, dense at low frequency like human hearing.
- Log compresses loudness range; floor at (max - 8) in log10, then (x + 4) / 4.
- Check in the app: manual result equals `whisper.log_mel_spectrogram` (difference ~0).

**3. Encoder**
- conv1 (stride 1) + conv2 (stride 2) + GELU: 3000 frames -> **1500 positions**, one per 20 ms.
- + sinusoidal positional embedding (self-attention has no sense of order by itself).
- N Transformer blocks: self-attention + MLP, residual connections, LayerNorm before each sub-layer.
- Output: 1500 x d_model matrix ("audio features"). `base`: d_model 512, 6 layers, 8 heads.
- Why the stride? Attention cost grows with the square of sequence length; halving it is ~4x cheaper.

**4. Language identification**
- Decoder gets only `<|startoftranscript|>`; the probabilities of the language tokens are the answer.
- One model does ASR, translation and language ID: the task is selected by special tokens.

**5. Decoder**
- Autoregressive: tokens so far -> causal self-attention -> cross-attention to the audio features ->
  probabilities over ~51 865 tokens -> pick best -> append -> repeat until `<|endoftext|>`.
- Prompt: `<|startoftranscript|>` `<|language|>` `<|transcribe|>` `<|notimestamps|>`.
- Greedy = best token each step. Beam search = keep several candidate sentences.
- The key/value cache stores earlier attention keys/values so each new step only processes one token.
- Probability and entropy per token show the model's own uncertainty; errors cluster at low-probability tokens.

**6. Cross-attention**
- Heat-map of which audio position each output token reads. A bright diagonal path = the model follows the
  audio in order. Whisper uses the same signal (alignment heads + dynamic time warping) for word timestamps.

**7. Evaluation**
- WER = (S + D + I) / N. Normalising text first (numbers, spelling, punctuation) avoids counting formatting
  differences as recognition errors: our numbers clip went from 43.8 % to 0 %.
- Noise experiment: white noise at chosen SNR. Model-size experiment: same architecture, more parameters.

## Likely questions

- **Did you train the model?** No. It is pretrained on about 680 000 hours of labelled web audio (weak
  supervision, Radford et al. 2022). We apply it and analyse it.
- **Is Whisper HMM- or CTC-based?** No. It is an attention encoder-decoder (sequence-to-sequence) Transformer trained
  with next-token cross-entropy. HMM systems need separate acoustic model, lexicon and language model; CTC
  predicts per frame with blanks. Whisper's decoder is also an implicit language model.
- **Why log-Mel instead of raw audio?** Compact, mimics hearing, easier for the network to learn from.
- **What does the attention map prove?** That token generation is tied to specific moments of the audio, in
  order, rather than being independent of it.
- **Why did "log-mel" come out as "log-mell"?** Rare technical term; look at the decoder tab: that token has
  low probability and high entropy.
- **Why can WER be misleading?** Formatting differences (digits vs words, spelling variants) count as errors unless
  the text is normalised.
- **What is the difference between model sizes?** Same architecture; more layers, wider vectors, more heads. Larger
  = more accurate but slower.
- **What are the weaknesses?** Hallucination on silence/noise, 30 s window, weaker on low-resource languages and
  accents, no speaker identification.
- **How is this different from just calling `model.transcribe()`?** `transcribe()` hides these stages; here each is
  re-implemented and checked against the library.

## Suggested demo order (about 6 minutes)
1. Overview: pipeline diagram and the "matches official" check.
2. Features: explain window, hop, mel, log.
3. Encoder: shapes table (80x3000 -> 1500x512), positional embedding.
4. Decoder: confidence-shaded transcript, runner-ups at the least confident token.
5. Attention: diagonal path.
6. Evaluation: WER breakdown, then the noise experiment (run it live or show saved numbers).
