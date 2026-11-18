# SpeechLens: lecture to notes

Pretrained **OpenAI Whisper** (not trained by us) turns a lecture recording into a raw transcript.
Our own pipeline then cleans it, corrects ASR errors with an n-gram language model, and produces
a summary, keywords, entities, relations, a question-answering box and downloadable notes.
A second page, **Inside the model**, opens Whisper up (log-Mel, encoder, decoder, attention, WER experiments).

## System Pipeline

| Stage | Method | File / function |
|---|---|---|
| ASR | pretrained Whisper (encoder-decoder Transformer) | `app.py`, `whisper_internals.py` |
| Clean-up | regular expressions | `clean_transcript` |
| Error correction | trigram Kneser-Ney LM, beam rescoring, WER | `LectureLM.correct`, `wer_score` |
| Keywords, summary | TF-IDF (noun phrases; length-adjusted sentence scores) | `keywords`, `summarize` |
| Entities, relations | spaCy NER, dependency parse | `entities`, `relations` |
| Q&A | TF-IDF cosine retrieval | `QAIndex` |
| Spoken answer | gTTS | `speak` |

## Setup

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org) (macOS: `brew install ffmpeg`).

```bash
git clone https://github.com/tani2306998/speechlens.git
cd speechlens
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

The Whisper `base` model (about 140 MB) downloads on first use.

## Using it
1. Choose input (upload, record, sample, or paste a transcript to skip ASR) -> **Generate notes**.
2. Type the true text in "Reference transcript" to see WER for Whisper / after clean-up / after LM correction.
3. Replace `domain_corpus.txt` (or upload a .txt in Settings) with your own lecture slides or notes:
   the language model can only correct towards words it has seen.

## Tests
`python test_notes.py` (modules) and `python test_notes_app.py` (headless app, paste mode).

## Honest notes
- The language model fixes spelling-type errors (edit distance <= 2) towards its training vocabulary; it cannot fix
  words it has never seen, and it can wrongly "correct" a rare correct word. Report WER before and after on
  your own recordings, including cases where it did not help.
- Q&A is retrieval (finds the best sentence), not generation.
- NER is trained on news text; technical names may be mislabelled.
- Whisper results from earlier clips (base model): clean 0 %, numbers 43.8 % / 0 % (raw / normalised),
  entities 6.7 % / 0 %, technical terms 40 % / 6.7 %.
