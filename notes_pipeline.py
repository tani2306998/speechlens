"""Lecture-to-notes pipeline: everything that happens AFTER the pretrained ASR model.

Stages (each is a plain function; no UI code here):
  1. clean_transcript      regex clean-up of fillers / repeats / spacing
  2. LectureLM.correct     Kneser-Ney trigram language-model rescoring of ASR errors
  3. keywords, summarize   TF-IDF keyphrases and extractive summary
  4. entities, relations   spaCy NER + subject-verb-object triples
  5. QAIndex.answer        TF-IDF retrieval question answering
  6. speak                 text-to-speech (gTTS, needs internet)
  7. build_notes           markdown study notes
"""
from __future__ import annotations

import math
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FLOOR = 1e-6          # probability floor so log never sees 0
HERE = Path(__file__).parent

# --------------------------------------------------------------------------- 1. clean-up
FILLERS = r"\b(?:um+|uh+|erm+|hmm+|ah+|you know|i mean|sort of|kind of)\b[,]?\s*"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]



# --------------------------------------------------------------------------- number words -> digits
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUM = {**_UNITS, **_TENS, **_SCALES, "hundred": 100}
_DIGIT_WORDS = {w: i for w, i in _UNITS.items() if i <= 9}


def _parse_number(ws: list[str]) -> tuple[int, int]:
    """Value and number of words consumed from the start of ws ('twenty five thousand' -> 25000, 3)."""
    total = cur = i = 0
    prev = None                                   # u unit, t tens, e teen, h hundred, s scale
    while i < len(ws):
        w = ws[i]
        if w == "and":
            if prev in ("h", "s") and i + 1 < len(ws) and ws[i + 1] in _NUM and ws[i + 1] not in _SCALES:
                i += 1
                continue
            break
        v = _NUM[w]
        if w in _SCALES:
            if prev is None:
                break
            total += (cur or 1) * v
            cur, prev = 0, "s"
        elif w == "hundred":
            if prev not in ("u", "t", "e", None) or cur >= 100:
                break
            cur, prev = (cur or 1) * 100, "h"          # 'twenty five hundred' -> 2500
        elif v >= 20:
            if prev not in (None, "h", "s"):
                break
            cur, prev = cur + v, "t"
        elif v >= 10:
            if prev not in (None, "h", "s"):
                break
            cur, prev = cur + v, "e"
        else:
            if prev not in (None, "h", "s") and not (prev == "t" and v >= 1):
                break
            cur, prev = cur + v, "u"
        i += 1
    return total + cur, i


def words_to_digits(text: str) -> tuple[str, int]:
    """'eighty mel bins' -> '80 mel bins', 'three point five' -> '3.5', 'fifty percent' -> '50%',
    'twenty twenty-two' -> '2022'. A lone 'one' stays a word ('no one', 'one of the')."""
    toks = list(re.finditer(r"[A-Za-z]+(?:-[A-Za-z]+)*", text))
    edits, n_conv, i = [], 0, 0

    def adjacent(a, b):                          # only whitespace between token a and token b
        return text[toks[a].end():toks[b].start()].strip() == ""

    while i < len(toks):
        first = toks[i].group().lower().split("-")
        if not all(x in _NUM for x in first):
            i += 1
            continue
        # collect a run of number words (and 'and' between hundred/thousand and more words)
        words, k = [], i
        while k < len(toks) and (k == i or adjacent(k - 1, k)):
            parts = toks[k].group().lower().split("-")
            if all(x in _NUM for x in parts):
                words += parts
            elif (parts == ["and"] and words and words[-1] in ("hundred", "thousand", "million")
                  and k + 1 < len(toks) and adjacent(k, k + 1)
                  and all(x in _NUM for x in toks[k + 1].group().lower().split("-"))):
                words.append("and")
            else:
                break
            k += 1
        # split the run into numbers
        nums, j = [], 0
        while j < len(words):
            v, used = _parse_number(words[j:])
            if used == 0:
                j += 1
                continue
            nums.append(str(v))
            j += used
        if not nums:
            i = k
            continue
        # years: 'twenty twenty-two', 'nineteen eighty four'
        if len(nums) >= 2 and nums[0] in ("19", "20") and 10 <= int(nums[1]) <= 99:
            nums = [nums[0] + nums[1].zfill(2)] + nums[2:]
        end_tok = k - 1
        # decimals: 'three point five'
        if (len(nums) >= 1 and end_tok + 2 < len(toks) + 0 and toks[end_tok + 1:end_tok + 2]
                and toks[end_tok + 1].group().lower() == "point" and adjacent(end_tok, end_tok + 1)):
            d, m = "", end_tok + 2
            while (m < len(toks) and adjacent(m - 1, m) and toks[m].group().lower() in _DIGIT_WORDS):
                d += str(_DIGIT_WORDS[toks[m].group().lower()])
                m += 1
            if d:
                nums[-1] = nums[-1] + "." + d
                end_tok = m - 1
        # percent
        pct = (end_tok + 1 < len(toks) and toks[end_tok + 1].group().lower() == "percent"
               and adjacent(end_tok, end_tok + 1))
        if pct:
            end_tok += 1
        lone_one = (len(words) == 1 and words[0] == "one" and len(nums) == 1 and nums[0] == "1"
                    and end_tok == i)
        if not lone_one:
            rep = " ".join(nums) + ("%" if pct else "")
            edits.append((toks[i].start(), toks[end_tok].end(), rep))
            n_conv += 1
        i = end_tok + 1
    for a, b, rep in reversed(edits):
        text = text[:a] + rep + text[b:]
    text = re.sub(r"(\d)\s+percent\b", r"\1%", text, flags=re.I)
    return text, n_conv


_OPND = r"(?:\d[\d.,]*%?[²³]?|\b[b-hj-zB-HJ-Z][²³]?(?!\w))"          # a number or a single-letter variable (not a / i)
_SYM = [                                                 # (pattern, replacement); order matters
    (r"(?<=[\dA-Za-z])\s+squared\b", "²"),
    (r"(?<=[\dA-Za-z])\s+cubed\b", "³"),
    (r"\b(?:is\s+)?equals?\b", "="),
    (r"\bis\s+equal\s+to\b(?=\s+" + _OPND + ")", "="),
    (r"\b(?:is\s+)?greater\s+than\b(?=\s+" + _OPND + ")", ">"),
    (r"\b(?:is\s+)?less\s+than\b(?=\s+" + _OPND + ")", "<"),
    (r"(" + _OPND + r")\s+plus\s+(?=" + _OPND + ")", r"\1 + "),
    (r"(" + _OPND + r")\s+minus\s+(?=" + _OPND + ")", r"\1 - "),
    (r"(" + _OPND + r")\s+(?:times|multiplied\s+by)\s+(?=" + _OPND + ")", r"\1 × "),
    (r"(" + _OPND + r")\s+divided\s+by\s+(?=" + _OPND + ")", r"\1 ÷ "),
    (r"(?<=\d)\s+degrees?\b(?!\s+of\b)", "°"),
    (r"\b(\d[\d,]*(?:\.\d+)?)\s+dollars?\b", r"$\1"),
]


def words_to_symbols(text: str) -> tuple[str, int]:
    """'two plus two equals four' (after digit conversion '2 plus 2 equals 4') -> '2 + 2 = 4'."""
    n = 0
    for pat, rep in _SYM:
        text, k = re.subn(pat, rep, text, flags=re.I)
        n += k
    text = re.sub(r"\s*=\s*", " = ", text)
    text = re.sub(r"\s*([<>])\s*", r" \1 ", text)
    return text, n


def clean_transcript(text: str, digits: bool = True) -> tuple[str, dict]:
    """Regex clean-up. Returns (clean text, counts of what was removed)."""
    stats = {"fillers": 0, "repeats": 0, "numbers": 0, "symbols": 0}
    t = text

    t, n = re.subn(FILLERS, "", t, flags=re.I)
    stats["fillers"] = n
    # "the the" -> "the"
    t, n = re.subn(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.I)
    stats["repeats"] = n
    if digits:                                       # number words -> digits, "percent" -> %
        t, stats["numbers"] = words_to_digits(t)
        t, stats["symbols"] = words_to_symbols(t)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)           # space before punctuation
    # space after punctuation, but never inside numbers (3.5, 1,000, 10:30) or abbreviations (e.g.)
    t = re.sub(r"(?<=[A-Za-z])([,;!?])(?=[A-Za-z])", r"\1 ", t)
    t = re.sub(r"(?<=[a-z])([.:])(?=[A-Z])", r"\1 ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    # capitalise sentence starts, make sure the text ends with a full stop
    def _cap(m):                                    # keep "e.g. x = 3" as it is
        before = m.string[:m.start(2)].rstrip().lower()
        if before.endswith(("e.g.", "i.e.", "etc.", "vs.", "fig.", "eq.")):
            return m.group(0)
        return m.group(1) + m.group(2).upper()
    t = re.sub(r"(^|[.!?]\s+)([a-z])", _cap, t)
    if t and t[-1] not in ".!?":
        t += "."
    return t, stats


# --------------------------------------------------------------------------- 2. language model
def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]



try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP
except Exception:                                           # pragma: no cover
    STOP = frozenset()


@lru_cache(maxsize=1)
def _english_checker():
    """A general English word list, so real words are never 'corrected'. Best source first."""
    try:
        from spellchecker import SpellChecker           # pip install pyspellchecker
        sp = SpellChecker()
        return lambda w: bool(sp.known([w]))
    except Exception:
        pass
    for f in ("/usr/share/dict/words", "/usr/share/dict/web2"):
        if Path(f).exists():
            words = {w.strip().lower() for w in open(f, encoding="utf-8", errors="ignore")}
            return lambda w: w in words
    return lambda w: False


def is_english(word: str) -> bool:
    return _english_checker()(word.lower())


_WORD = re.compile(r"[A-Za-z']+")


def _norm_tokens(sentence: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(sentence)]


class LectureLM:
    """Trigram Kneser-Ney language model trained on a domain corpus (course notes).

    correct() fixes ASR spelling errors by beam-searching over candidate words
    (the heard word, plus vocabulary words within edit distance 1-2) and keeping
    the sentence with the best  log10 P_LM(sentence) - lam * (total edits).
    """

    def __init__(self, corpus_lines: list[str], order: int = 3):
        from nltk.lm import KneserNeyInterpolated, Laplace
        from nltk.lm.preprocessing import padded_everygram_pipeline

        self.order = order
        sents = [_norm_tokens(s) for s in corpus_lines if s.strip()]
        sents = [s for s in sents if s]
        train, vocab = padded_everygram_pipeline(order, sents)
        # Kneser-Ney needs a context, so the unigram baseline uses add-one (Laplace) smoothing
        self.lm = Laplace(1) if order == 1 else KneserNeyInterpolated(order)
        self.smoothing = "add-one" if order == 1 else "Kneser-Ney"
        self.lm.fit(train, vocab)
        self.vocab = sorted({w for s in sents for w in s})
        self._vocab_set = set(self.vocab)
        self.n_sentences = len(sents)
        self._score = lru_cache(maxsize=200_000)(self._score_raw)

    @classmethod
    def from_file(cls, path: str | Path | None = None):
        path = Path(path) if path else HERE / "domain_corpus.txt"
        return cls(path.read_text(encoding="utf-8").splitlines())

    def _ctx(self, chosen: list[str]) -> tuple:
        n = self.order - 1
        return tuple((["<s>"] * n + chosen)[-n:]) if n > 0 else ()

    def _score_raw(self, word: str, ctx: tuple) -> float:
        return max(self.lm.score(word, list(ctx)), FLOOR)

    def logprob_tokens(self, toks: list[str]) -> float:
        seq = ["<s>"] * (self.order - 1) + toks + ["</s>"]
        total = 0.0
        for i in range(self.order - 1, len(seq)):
            total += math.log10(self._score(seq[i], tuple(seq[max(0, i - self.order + 1):i]) if self.order > 1 else ()))
        return total

    def logprob(self, sentence: str) -> float:
        return self.logprob_tokens(_norm_tokens(sentence))

    def perplexity(self, sentence: str) -> float:
        toks = _norm_tokens(sentence)
        if not toks:
            return float("nan")
        return 10 ** (-self.logprob_tokens(toks) / (len(toks) + 1))

    @lru_cache(maxsize=50_000)
    def _candidates(self, word: str, max_edit: int) -> tuple:
        """Heard word + possible fixes. Real words are trusted: only NON-words are corrected."""
        out = [(word, 0)]
        if (len(word) < 3 or "'" in word or word in STOP or word in self._vocab_set
                or is_english(word)):
            return tuple(out)
        for v in self.vocab:
            if v == word or v in STOP or len(v) < 3 or abs(len(v) - len(word)) > max_edit:
                continue
            d = edit_distance(word, v)
            if d <= (1 if len(word) < 7 else max_edit):
                out.append((v, d))
        return tuple(out)

    def correct_sentence(self, sentence: str, lam: float = 1.5, beam: int = 6,
                         max_edit: int = 2) -> tuple[str, list[tuple[str, str]]]:
        """Return (corrected sentence, list of (heard, corrected))."""
        pieces = re.split(r"([A-Za-z']+)", sentence)    # keeps punctuation / spaces
        word_idx = [i for i, p in enumerate(pieces) if _WORD.fullmatch(p)]
        words = [pieces[i].lower() for i in word_idx]
        if not words:
            return sentence, []

        beams = [(0.0, [])]                              # (score, chosen words)
        for w in words:
            new = []
            for score, chosen in beams:
                ctx = self._ctx(chosen)
                for cand, d in self._candidates(w, max_edit):
                    s = score + math.log10(self._score(cand, ctx)) - lam * d
                    new.append((s, chosen + [cand]))
            new.sort(key=lambda x: -x[0])
            beams = new[:beam]
        # add the end-of-sentence probability, then pick the best
        best = max(beams, key=lambda b: b[0] + math.log10(
            self._score("</s>", self._ctx(b[1]))))[1]

        changes = []
        for i, heard, fixed in zip(word_idx, words, best):
            if heard != fixed:
                orig = pieces[i]
                if orig.isupper() and len(orig) > 1:
                    fixed_c = fixed.upper()
                elif orig[:1].isupper():
                    fixed_c = fixed.capitalize()
                else:
                    fixed_c = fixed
                changes.append((orig, fixed_c))
                pieces[i] = fixed_c
        return "".join(pieces), changes

    def word_probs(self, sentence: str) -> list[dict]:
        """P(word | previous words) for every word: what the language model 'believes'."""
        toks, rows, chosen = _norm_tokens(sentence), [], []
        for w in toks + ["</s>"]:
            ctx = self._ctx(chosen)
            p = self._score(w, ctx)
            rows.append({"word": w, "context": " ".join(ctx) if ctx else "(none)",
                         "P": p, "log10 P": round(math.log10(p), 2),
                         "in vocabulary": w in self._vocab_set or w == "</s>"})
            chosen.append(w)
        return rows

    def trace(self, sentence: str, lam: float = 1.5, max_edit: int = 2) -> list[dict]:
        """For each word with alternatives: every candidate, its LM score and the edit penalty."""
        fixed, _ = self.correct_sentence(sentence, lam=lam, max_edit=max_edit)
        orig = _norm_tokens(sentence)
        new = _norm_tokens(fixed)
        rows, left = [], []
        for heard, chosen in zip(orig, new):
            cands = self._candidates(heard, max_edit)
            if len(cands) > 1:
                ctx = self._ctx(left)
                for c, d in cands:
                    lp = math.log10(self._score(c, ctx))
                    rows.append({"heard": heard, "candidate": c, "edits": d,
                                 "log10 P(candidate | context)": round(lp, 2),
                                 "score = logP - 1.5 x edits": round(lp - lam * d, 2),
                                 "chosen": c == chosen})
            left.append(chosen)
        return rows

    def correct(self, text: str, **kw) -> tuple[str, list[tuple[str, str]]]:
        out, changes = [], []
        for s in split_sentences(text) or [text]:
            cs, ch = self.correct_sentence(s, **kw)
            out.append(cs)
            changes += ch
        return " ".join(out), changes


# --------------------------------------------------------------------------- WER
def wer_score(reference: str, hypothesis: str) -> dict:
    """WER after Whisper's English normaliser (falls back to a simple one)."""
    import jiwer

    try:
        from whisper.normalizers import EnglishTextNormalizer
        norm = EnglishTextNormalizer()
    except Exception:                                       # pragma: no cover
        def norm(s):
            return re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("-", " "))
    r, h = " ".join(norm(reference).split()), " ".join(norm(hypothesis).split())
    if not r:
        return {"wer": float("nan"), "S": 0, "D": 0, "I": 0, "N": 0}
    o = jiwer.process_words(r, h if h else "")
    return {"wer": o.wer, "S": o.substitutions, "D": o.deletions, "I": o.insertions,
            "N": o.substitutions + o.deletions + o.hits}


# --------------------------------------------------------------------------- 3. keywords + summary
GENERIC = {"example", "thing", "something", "anything", "everything", "nothing", "way", "lot", "kind",
           "sort", "bit", "fact", "question", "answer", "lecture", "video", "today", "class", "section",
           "course", "case", "idea", "part", "one", "use", "time", "point", "problem", "value", "second",
           "term", "person", "effort", "place", "form", "side", "left", "right", "list", "step", "moment",
           "information", "ability", "tutorial", "diagram", "number", "way", "lot", "kind", "type", "bunch",
           "stuff", "word", "reason", "result", "mention", "branch"}


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:                                         # model missing: download it once
        try:
            from spacy.cli import download
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as e:                              # pragma: no cover
            raise RuntimeError("Run:  python -m spacy download en_core_web_sm") from e


def _phrases(doc) -> list[str]:
    """Noun phrases (lemmatised, without stop words) of one sentence."""
    out = []
    for nc in doc.noun_chunks:
        toks = [t for t in nc if t.pos_ in ("NOUN", "PROPN", "ADJ") and not t.is_stop
                and t.is_alpha and len(t.text) > 2]
        if not any(t.pos_ in ("NOUN", "PROPN") for t in toks):
            continue
        words = [t.lemma_.lower() for t in toks]
        if all(w in GENERIC for w in words):
            continue
        out.append(" ".join(w for w in words if w not in GENERIC))
    return out


def _content_lemmas(sent_doc) -> list[str]:
    return [t.lemma_.lower() for t in sent_doc
            if t.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not t.is_stop and t.is_alpha
            and len(t.text) > 2 and t.lemma_.lower() not in GENERIC]


def _idf(df: Counter, n: int, key) -> float:
    return math.log((1 + n) / (1 + df[key])) + 1


def keywords(text: str, k: int = 8) -> list[tuple[str, float]]:
    """Noun phrases ranked by TF-IDF: how often in the lecture x how few sentences contain them."""
    sents = split_sentences(text)
    if not sents:
        return []
    cands = [_phrases(d) for d in _nlp().pipe(sents)]
    tf = Counter(p for c in cands for p in c)
    df = Counter(p for c in cands for p in set(c))
    n = len(sents)
    score = {p: tf[p] * _idf(df, n, p) * (1 + 0.3 * (len(p.split()) - 1)) for p in tf}
    top = sorted(score.items(), key=lambda x: (-x[1], x[0]))[:k]
    return [(p, float(sc)) for p, sc in top]


_LEAD = re.compile(r"^(?:(?:and|so|now|okay|ok|well|but|then|alright|right|also|yeah)\b[,\s]*)+", re.I)
_DEIXIS = {"this", "that", "these", "those", "here", "there", "it"}
_TALK = re.compile(r"\b(i'm going to|i am going to|we'll|we will|let's|let us|going to look|you'll notice|"
                   r"as you can see|so what i|you'll notice|you will notice|we often see|if you look|look at|what i'm|welcome to|thank you|in this video|in this lecture)\b", re.I)
_DEFN = re.compile(r"\b(is|are|means|equals|refers to|represents|defined as|measures|describes|gives|states)\b", re.I)


def _tidy(sentence: str) -> str:
    """Drop spoken lead-ins ('And', 'Now', 'So') and recapitalise."""
    t = _LEAD.sub("", sentence).strip()
    return t[:1].upper() + t[1:] if t else sentence


def sentence_scores(text: str) -> list[dict]:
    """Importance of each sentence = sum of TF-IDF weights of its content words / (#content words)^0.7
    (so length alone does not win), then adjusted for lecture-speak:
      x0.5 if it leans on 'this/that/these/it' (it only makes sense with context)
      x0.35 if it is procedural talk ("I'm going to...", "let's...", "you'll notice...")
      x2 if it looks like a definition (X is / means / equals ...)
      x1.25 for the opening sentence.
    Very short sentences and questions are not eligible for the summary."""
    sents = split_sentences(text)
    if not sents:
        return []
    docs = list(_nlp().pipe(sents))
    lem = [_content_lemmas(d) for d in docs]
    tf = Counter(w for l in lem for w in l)
    df = Counter(w for l in lem for w in set(l))
    n = len(sents)
    rows = []
    for i, (s, d, l) in enumerate(zip(sents, docs, lem)):
        w = sum(math.sqrt(tf[x]) * _idf(df, n, x) for x in l)
        base = w / max(1, len(l)) ** 0.7
        body = _LEAD.sub("", s).lower().split()
        deictic = sum(1 for x in body if x.strip(".,") in _DEIXIS) >= 2 or (body[:1] and body[0] in _DEIXIS)
        talk = bool(_TALK.search(s))
        defn = bool(_DEFN.search(s)) and not deictic
        mult = (0.5 if deictic else 1) * (0.35 if talk else 1) * (2.0 if defn else 1) * (1.25 if i == 0 else 1)
        eligible = len(body) >= 6 and not s.rstrip().endswith("?") and len(l) >= 2
        rows.append({"#": i + 1, "sentence": s, "content words": len(l), "score": round(base * mult, 2),
                     "eligible": eligible, "_lem": set(l)})
    order = sorted([r for r in rows if r["eligible"]], key=lambda r: -r["score"])
    rank = {r["#"]: j + 1 for j, r in enumerate(order)}
    for r in rows:
        r["rank"] = rank.get(r["#"], "-")
    return rows


def summarize(text: str, n: int = 3) -> list[str]:
    """Extractive summary: highest-scoring eligible sentences, skipping near-duplicates
    (>=50% shared content words with one already chosen), shown in lecture order."""
    rows = sentence_scores(text)
    pool = sorted([r for r in rows if r["eligible"]] or rows, key=lambda r: -r["score"])
    chosen = []
    for r in pool:
        if len(chosen) == n:
            break
        if any(len(r["_lem"] & c["_lem"]) / max(1, len(r["_lem"] | c["_lem"])) >= 0.5 for c in chosen):
            continue
        chosen.append(r)
    return [_tidy(r["sentence"]) for r in sorted(chosen, key=lambda r: r["#"])]


# --------------------------------------------------------------------------- 4. entities + relations
_KEEP_ENT = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC",
             "PERCENT", "QUANTITY", "MONEY", "TIME", "DATE"}


def entities(text: str) -> list[tuple[str, str]]:
    """spaCy named entities, minus noise: bare numbers/ordinals, 1-2 letter 'names' (VL, D.), symbols such as
    second², and dates that are not real years."""
    seen, out = set(), []
    for e in _nlp()(text).ents:
        t = re.sub(r"^(?:a|an|the|about|around|at least|over)\s+", "", e.text.strip(), flags=re.I)
        if e.label_ not in _KEEP_ENT or len(re.findall(r"[A-Za-z]", t)) <= 2 and e.label_ not in ("PERCENT", "MONEY"):
            continue
        if any(c in t for c in "²³") or t.lower().startswith("negative"):
            continue
        if e.label_ == "DATE" and not re.search(r"\b(1[0-9]|20)\d\d\b", t):
            continue
        if e.label_ in ("PERSON", "ORG", "NORP", "PRODUCT") and (len(t) <= 3 or t.isupper() and len(t) <= 3):
            continue
        key = (t, e.label_)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _np_text(tok) -> str:
    """Short noun phrase around a head word: no determiners, at most 4 words."""
    for nc in tok.doc.noun_chunks:
        if nc.root.i == tok.i:
            keep = [t.text for t in nc if t.pos_ in ("NOUN", "PROPN", "ADJ", "NUM") and not t.is_stop]
            if keep:
                return " ".join(keep[-4:])
    return tok.text


def relations(text: str) -> list[tuple[str, str, str]]:
    """Subject - verb - object triples between nouns (pronoun subjects such as 'that' are resolved)."""
    triples, seen = [], set()
    for sent in _nlp()(text).sents:
        for tok in sent:
            if tok.pos_ not in ("VERB", "AUX") or (tok.pos_ == "AUX" and tok.lemma_ != "be"):
                continue
            subj = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
            obj = [c for c in tok.children if c.dep_ in ("dobj", "attr", "dative")]
            if not subj or not obj:
                continue
            s, o = subj[0], obj[0]
            if s.pos_ == "PRON" and tok.dep_ == "relcl":      # "objects that have ..." -> objects
                s = tok.head
            if s.pos_ not in ("NOUN", "PROPN") or o.pos_ not in ("NOUN", "PROPN"):
                continue
            if any(len(x.text) < 3 or not x.text.isalpha() or x.is_stop or x.lemma_.lower() in GENERIC
                   for x in (s, o)):                       # drops 'V subscripte O', 'person make effort'
                continue
            verb = tok.text.lower() if tok.lemma_ == "be" else tok.lemma_
            t = (_np_text(s), verb, _np_text(o))
            if t[0].lower() != t[2].lower() and t not in seen:
                seen.add(t)
                triples.append(t)
    return triples


def study_cards(res: dict) -> list[tuple[str, str]]:
    """For each keyword, the lecture sentence that best defines it."""
    sents = split_sentences(res["corrected"])
    cards = []
    for kw, _ in res["keywords"]:
        word = rf"\b{re.escape(kw)}s?\b"
        define = re.compile(word + r".{0,50}\b(is|are|has|have|means|refers|called|defined|measures)\b", re.I)
        best, best_score = None, 0.0
        for s in sents:
            if not re.search(word, s, re.I) or s.rstrip().endswith("?") or len(s.split()) < 6:
                continue
            score = 3 * bool(define.search(s)) + min(len(s.split()), 20) / 20
            if score > best_score:
                best, best_score = s, score
        if best:
            cards.append((kw, best))
    return cards


# --------------------------------------------------------------------------- 5. question answering
@dataclass
class QAIndex:
    sentences: list[str]
    vec: TfidfVectorizer = field(init=False)
    X: object = field(init=False)
    ents: list = field(init=False)

    def __post_init__(self):
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.X = self.vec.fit_transform(self.sentences)
        self.ents = [{e.label_ for e in _nlp()(s).ents} for s in self.sentences]

    def rank(self, question: str) -> list[dict]:
        """Cosine similarity of the question to every lecture sentence."""
        sims = cosine_similarity(self.vec.transform([question]), self.X).ravel()
        return [{"#": int(i) + 1, "sentence": self.sentences[i], "similarity": round(float(sims[i]), 3)}
                for i in sims.argsort()[::-1]]

    def answer(self, question: str, min_score: float = 0.12) -> dict:
        q = self.vec.transform([question])
        sims = cosine_similarity(q, self.X).ravel()
        ql = question.lower().strip()
        want = ({"PERSON", "ORG"} if ql.startswith("who") else
                {"DATE", "TIME"} if ql.startswith("when") else
                {"GPE", "LOC"} if ql.startswith("where") else
                {"CARDINAL", "QUANTITY", "PERCENT"} if ql.startswith("how many") else set())
        if want:                                    # question word prefers matching entity types
            sims = sims + 0.15 * np.array([bool(e & want) for e in self.ents])
        order = sims.argsort()[::-1]
        best = int(order[0])
        if sims[best] < min_score:
            return {"answer": "I could not find that in the lecture.", "score": float(sims[best]),
                    "sentence_id": None}
        return {"answer": self.sentences[best], "score": float(sims[best]), "sentence_id": best}


# --------------------------------------------------------------------------- 6. text to speech
def speak(text: str) -> bytes | None:
    """MP3 bytes from gTTS, or None when offline / not installed."""
    try:
        from io import BytesIO
        from gtts import gTTS
        buf = BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


# --------------------------------------------------------------------------- 7. notes file
def process(raw: str, lm: LectureLM | None, n_summary: int = 3, n_keywords: int = 8,
            digits: bool = True) -> dict:
    """Run every text stage on a raw ASR transcript."""
    cleaned, clean_stats = clean_transcript(raw, digits)
    changes = []
    corrected = cleaned
    if lm is not None:
        corrected, changes = lm.correct(cleaned)
    return {
        "raw": raw, "cleaned": cleaned, "clean_stats": clean_stats,
        "corrected": corrected, "changes": changes,
        "summary": summarize(corrected, n_summary),
        "keywords": keywords(corrected, n_keywords),
        "entities": entities(corrected),
        "relations": relations(corrected),
    }


def build_notes(title: str, res: dict, qa_log: list[tuple[str, str]] | None = None) -> str:
    md = [f"# {title}", "", "## Summary"]
    md += [f"- {s}" for s in res["summary"]]
    md += ["", "## Keywords"] + [f"- {k}" for k, _ in res["keywords"]]
    md += ["", "## Entities"] + ([f"- {t} ({l})" for t, l in res["entities"]] or ["- none found"])
    md += ["", "## Relations (subject, verb, object)"] + (
        [f"- {s} | {v} | {o}" for s, v, o in res["relations"]] or ["- none found"])
    if qa_log:
        md += ["", "## Questions asked"] + [f"- **Q:** {q}\n  **A:** {a}" for q, a in qa_log]
    md += ["", "## Full transcript (cleaned and corrected)", "", textwrap.fill(res["corrected"], 100), ""]
    return "\n".join(md)
