"""SpeechLens: lecture to notes.

Pretrained Whisper turns the lecture audio into a raw transcript; everything after that is our own
pipeline (clean-up, n-gram language-model correction, TF-IDF keywords and summary, entities and
relations, question answering, text to speech, study cards, downloadable notes).

Run:  streamlit run app.py      (second page in the sidebar: "Inside the model")
"""
import html as H
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import notes_pipeline as npl

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "tests" / "jfk.flac"
LANGS = {"English (en)": "en", "Auto-detect": None, "Hindi (hi)": "hi", "Tamil (ta)": "ta",
         "Telugu (te)": "te", "Spanish (es)": "es", "French (fr)": "fr"}
WHISPER_SIZES = {"tiny": "39 M parameters", "base": "74 M parameters · 512 dims · 6 layers · 8 heads",
                 "small": "244 M parameters"}
ENT_COLORS = {"PERSON": "#f472b6", "ORG": "#60a5fa", "DATE": "#fbbf24", "GPE": "#34d399",
              "CARDINAL": "#a78bfa", "PRODUCT": "#fb923c", "WORK_OF_ART": "#f87171"}

st.set_page_config(page_title="SpeechLens: lecture to notes", page_icon="🎙️", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp {font-family: 'Inter', sans-serif;}
.stApp {background: radial-gradient(1200px 600px at 85% -10%, rgba(139,92,246,.18), transparent 60%),
        radial-gradient(900px 500px at -10% 10%, rgba(45,212,191,.16), transparent 60%), #0b1220;}
[data-testid="stToolbar"], #MainMenu, footer, .stDeployButton {display:none !important;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1180px; padding-top: 0.6rem; padding-bottom: 3rem;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.01em;}
@keyframes rise {from {opacity:0; transform: translateY(18px);} to {opacity:1; transform:none;}}
@keyframes bar {0%,100% {transform: scaleY(.25);} 50% {transform: scaleY(1);}}
@keyframes glow {0%,100% {box-shadow:0 0 0 0 rgba(45,212,191,.0);} 50% {box-shadow:0 0 40px 2px rgba(45,212,191,.18);}}
.nav {display:flex; align-items:center; justify-content:space-between; padding:14px 6px 6px 6px;}
.logo {font-family:'Space Grotesk'; font-weight:700; font-size:1.35rem; color:#e6edf3;}
.logo span {background:linear-gradient(90deg,#2dd4bf,#8b5cf6); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.navlinks a {color:#93a4bd; margin-left:22px; font-size:.92rem; text-decoration:none;}
.navlinks a:hover {color:#2dd4bf;}
.hero {text-align:center; padding: 46px 10px 18px 10px; animation: rise .7s ease both;}
.badge {display:inline-block; padding:6px 14px; border:1px solid #28405f; border-radius:999px; color:#7dd3c8;
        font-size:.82rem; background:rgba(45,212,191,.08); margin-bottom:18px;}
.hero h1 {font-size:4.1rem; line-height:1.02; margin:0 0 14px 0; color:#f2f6fb;}
.grad {background:linear-gradient(90deg,#2dd4bf 0%,#60a5fa 50%,#a78bfa 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p {color:#a9b8cf; font-size:1.15rem; max-width:720px; margin:0 auto;}
.wave {display:flex; justify-content:center; align-items:center; gap:4px; height:90px; margin:30px 0 6px 0;}
.wave i {display:block; width:5px; height:70px; border-radius:4px; background:linear-gradient(#2dd4bf,#8b5cf6);
         transform-origin:center; animation: bar 1.3s ease-in-out infinite;}
.stats {display:flex; gap:14px; justify-content:center; flex-wrap:wrap; margin: 18px 0 8px 0;}
.stat {background:rgba(17,26,46,.8); border:1px solid #1f2b45; border-radius:14px; padding:12px 22px; min-width:150px;}
.stat b {display:block; font-family:'Space Grotesk'; font-size:1.5rem; color:#2dd4bf;}
.stat small {color:#93a4bd;}
.sec {text-align:center; margin:54px 0 20px 0;}
.sec small {color:#2dd4bf; letter-spacing:.14em; text-transform:uppercase; font-weight:600; font-size:.78rem;}
.sec h2 {margin:6px 0 0 0; font-size:2.2rem; color:#f2f6fb;}
.grid {display:grid; gap:16px;}
.g4 {grid-template-columns: repeat(4, 1fr);}
.g3 {grid-template-columns: repeat(3, 1fr);}
@media (max-width: 900px) {.g4,.g3 {grid-template-columns: 1fr 1fr;} .hero h1 {font-size:2.6rem;}}
.card {background:linear-gradient(180deg,rgba(21,32,56,.9),rgba(15,23,42,.9)); border:1px solid #1f2b45; border-radius:18px;
       padding:22px; transition: transform .25s, border-color .25s, box-shadow .25s; animation: rise .8s ease both;}
.card:hover {transform: translateY(-6px); border-color:#2dd4bf; box-shadow:0 14px 34px rgba(0,0,0,.35);}
.card .ic {font-size:1.9rem; margin-bottom:10px;}
.card h4 {margin:0 0 6px 0; color:#f2f6fb; font-family:'Space Grotesk'; font-size:1.12rem;}
.card p {margin:0; color:#9db0c9; font-size:.93rem; line-height:1.5;}
.step {position:relative;}
.step .n {position:absolute; top:14px; right:18px; font-family:'Space Grotesk'; font-size:2.4rem; color:rgba(45,212,191,.18); font-weight:700;}
.pill {display:inline-block; padding:5px 13px; border-radius:999px; margin:4px 6px 4px 0; font-size:.88rem; font-weight:600;
       border:1px solid; background:rgba(255,255,255,.03);}
.kw {display:inline-block; margin:5px 8px 5px 0; padding:4px 14px; border-radius:999px; background:rgba(45,212,191,.1);
     border:1px solid rgba(45,212,191,.35); color:#7ee6d8; font-weight:600;}
.tile {background:linear-gradient(135deg,rgba(45,212,191,.12),rgba(139,92,246,.12)); border:1px solid #28405f; border-radius:16px;
       padding:18px 20px; animation: rise .6s ease both;}
.tile b {display:block; font-family:'Space Grotesk'; font-size:1.9rem; color:#f2f6fb;}
.tile small {color:#93a4bd;}
.sumcard {border-left:3px solid #2dd4bf; background:rgba(17,26,46,.75); padding:14px 18px; border-radius:0 12px 12px 0; margin-bottom:10px; color:#dbe5f3;}
.fix {background:rgba(251,191,36,.25); color:#fde68a; padding:0 4px; border-radius:4px;}
.txt {background:rgba(17,26,46,.8); border:1px solid #1f2b45; border-radius:14px; padding:16px 18px; line-height:1.65; color:#d5e0f0;}
.lbl {color:#2dd4bf; font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; font-weight:600; margin-bottom:6px;}
.ready {border:1px solid #1d4d49; background:linear-gradient(90deg,rgba(45,212,191,.16),rgba(45,212,191,.04));
        border-radius:14px; padding:14px 20px; color:#9ff0e4; font-weight:600; margin: 10px 0 14px 0; animation: glow 3s infinite;}
.curious {text-align:center; margin: 40px 0 6px 0;}
.curious h3 {color:#f2f6fb; margin-bottom:4px;}
.curious p {color:#93a4bd;}
.footer {text-align:center; color:#6b7c96; font-size:.85rem; margin-top:56px; padding-top:20px; border-top:1px solid #1a2640;}
/* streamlit widgets */
.stTabs [data-baseweb="tab-list"] {gap:8px; background:transparent; border-bottom:1px solid #1f2b45;}
.stTabs [data-baseweb="tab"] {border-radius:999px 999px 0 0; padding:10px 18px; color:#93a4bd;}
.stTabs [aria-selected="true"] {color:#2dd4bf !important;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:20px !important; border-color:#1f2b45 !important;
        background:rgba(15,23,42,.65);}
.stButton > button[kind="primary"], .stDownloadButton > button {background:linear-gradient(90deg,#14b8a6,#6366f1);
        border:0; color:#fff; font-weight:600; border-radius:12px; padding:.65rem 1.6rem; transition: transform .2s;}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {transform: translateY(-2px); filter:brightness(1.08);}
.stButton > button[kind="secondary"] {border-radius:12px; border:1px solid #2dd4bf; color:#2dd4bf; padding:.65rem 1.6rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s: str):
    """Render HTML without blank lines or leading spaces (which markdown would turn into code blocks)."""
    st.markdown("\n".join(l.strip() for l in s.strip().splitlines() if l.strip()), unsafe_allow_html=True)


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading pretrained Whisper weights...")
def load_whisper(name):
    import whisper
    return whisper.load_model(name, device="cpu")


@st.cache_resource(show_spinner="Training the n-gram language model...")
def load_lm(corpus_text, order: int):
    lines = (corpus_text or npl.HERE.joinpath("domain_corpus.txt").read_text(encoding="utf-8")).splitlines()
    return npl.LectureLM(lines, order=order)


@st.cache_resource(show_spinner=False)
def get_nlp():
    return npl._nlp()


def transcribe(path: str, model_name: str, lang) -> str:
    return load_whisper(model_name).transcribe(path, language=lang, fp16=False)["text"].strip()


def save_upload(f) -> str:
    suffix = Path(getattr(f, "name", "rec.wav")).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        t.write(f.getvalue())
        return t.name


def waveform(path: str, points: int = 900):
    """Peak envelope of the recording, or None if ffmpeg / whisper is not available."""
    try:
        import whisper
        a = np.abs(whisper.load_audio(path))
        n = max(1, len(a) // points)
        return [float(a[i:i + n].max()) for i in range(0, len(a) - n + 1, n)][:points], len(a) / 16000
    except Exception:
        return None, None


def highlight(text: str, changes) -> str:
    out = H.escape(text)
    for _, fixed in changes:
        out = out.replace(H.escape(fixed), f'<span class="fix">{H.escape(fixed)}</span>', 1)
    return out


# ------------------------------------------------------------------ navbar + hero
html("""
<div class="nav"><div class="logo">🎙️ Speech<span>Lens</span></div>
<div class="navlinks"><a href="#how">How it works</a><a href="#features">Features</a><a href="#try">Try it</a></div></div>
<div class="hero">
<div class="badge">Pretrained Whisper · N-gram language model · TF-IDF · spaCy</div>
<h1>Lectures in.<br><span class="grad">Notes out.</span></h1>
<p>Upload a recorded lecture and get a clean transcript, a short summary, keywords, entities and an assistant that answers your questions about it.</p>
""" + '<div class="wave">' + "".join(
    f'<i style="animation-delay:{(i * 0.09) % 1.3:.2f}s;height:{30 + (i * 37) % 55}px"></i>' for i in range(46)) +
     '</div></div>')

res = st.session_state.get("res")

if not res:
    html("""
<div class="stats">
<div class="stat"><b>6</b><small>processing stages</small></div>
<div class="stat"><b>~99</b><small>languages in the ASR model</small></div>
<div class="stat"><b>680k h</b><small>of speech Whisper learned from</small></div>
<div class="stat"><b>3-gram</b><small>Kneser-Ney error correction</small></div></div>
<div class="sec" id="how"><small>How it works</small><h2>From a recording to study notes in four steps</h2></div>
<div class="grid g4">
<div class="card step"><div class="n">1</div><div class="ic">🎧</div><h4>Listen</h4><p>Whisper, a pretrained encoder-decoder Transformer, converts the audio to a raw transcript.</p></div>
<div class="card step"><div class="n">2</div><div class="ic">🧹</div><h4>Clean and correct</h4><p>Regular expressions remove fillers. A trigram language model fixes words the recogniser misheard.</p></div>
<div class="card step"><div class="n">3</div><div class="ic">🧠</div><h4>Understand</h4><p>TF-IDF finds keywords and the key sentences. spaCy finds entities and relations.</p></div>
<div class="card step"><div class="n">4</div><div class="ic">💬</div><h4>Ask and revise</h4><p>Ask questions by text or voice, flip study cards and download the notes.</p></div></div>
<div class="sec" id="features"><small>Features</small><h2>Everything a lecture needs</h2></div>
<div class="grid g3">
<div class="card"><div class="ic">📝</div><h4>Smart summary</h4><p>The sentences that carry the most information, picked by TF-IDF weight, in lecture order.</p></div>
<div class="card"><div class="ic">🔑</div><h4>Keyword chips</h4><p>Key noun phrases ranked by importance, so you see the topics at a glance.</p></div>
<div class="card"><div class="ic">🕸️</div><h4>Entities and relations</h4><p>People, organisations and dates, plus who-did-what-to-whom links drawn as a graph.</p></div>
<div class="card"><div class="ic">🗣️</div><h4>Voice Q&amp;A</h4><p>Ask by typing or speaking. Answers come from the lecture itself and are read back to you.</p></div>
<div class="card"><div class="ic">🎓</div><h4>Study cards</h4><p>Every keyword becomes a flash card with the sentence that explains it.</p></div>
<div class="card"><div class="ic">📉</div><h4>Measured accuracy</h4><p>Paste the true text to see word error rate before and after correction.</p></div></div>
<div class="sec" id="try"><small>Try it</small><h2>Turn your lecture into notes</h2></div>
""")
else:
    html('<div class="sec" id="try" style="margin-top:20px"><small>New lecture</small><h2>Try another recording</h2></div>')

# ------------------------------------------------------------------ input card
with st.container(border=True):
    choices = ["Upload lecture audio", "Record with microphone", "Paste a transcript (skip ASR)"]
    if SAMPLE.exists():
        choices.insert(2, "Sample clip (JFK, 11 s)")
    source = st.radio("Input", choices, horizontal=True)
    audio_file, audio_path, pasted = None, None, ""
    if source == "Upload lecture audio":
        audio_file = st.file_uploader("Drop an audio or video file (MP3, WAV, M4A, FLAC, OGG, MP4, MKV, WEBM, MOV)",
                                      type=["mp3", "wav", "m4a", "flac", "ogg", "mp4", "mkv", "webm", "mov"])
    elif source == "Record with microphone":
        if hasattr(st, "audio_input"):
            audio_file = st.audio_input("Record")
        else:
            st.warning("Recording needs a newer Streamlit: pip install -U streamlit")
    elif source.startswith("Sample"):
        audio_path = str(SAMPLE)
        st.audio(audio_path)
    else:
        pasted = st.text_area("Raw transcript", height=150, placeholder="Paste text from any ASR system...")
    if audio_file is not None:
        if str(getattr(audio_file, "name", "")).lower().endswith((".mp4", ".mkv", ".webm", ".mov")):
            st.video(audio_file)                        # Whisper extracts the audio track with ffmpeg
        else:
            st.audio(audio_file)

    with st.expander("⚙️ Settings: model, language and language model"):
        s1, s2, s3 = st.columns(3)
        model_name = s1.selectbox("Whisper model (pretrained)", list(WHISPER_SIZES), index=1)
        s1.caption(f"{WHISPER_SIZES[model_name]}. Also exist: medium (769 M), large (1.5 B).")
        lang_label = s2.selectbox("Language", list(LANGS))
        n_sum = s2.slider("Summary sentences", 1, 8, 3)
        use_lm = s3.toggle("Correct errors with n-gram LM", value=True)
        use_digits = s3.toggle("Numbers as digits and math symbols (eighty → 80, plus → +)", value=True)
        order = s3.radio("N-gram order", [1, 2, 3], index=2, horizontal=True,
                         format_func={1: "Unigram", 2: "Bigram", 3: "Trigram"}.get)
        n_kw = s3.slider("Keywords", 3, 15, 8)
        corpus_file = st.file_uploader("Domain text for the language model (.txt): your lecture slides or notes",
                                       type=["txt"])
        corpus_text = corpus_file.getvalue().decode("utf-8", "ignore") if corpus_file else None

    c1, c2 = st.columns([3, 2])
    reference = c1.text_area("Reference transcript (optional): what was really said, to measure accuracy", height=90)
    title = c2.text_input("Notes title", "Lecture notes")
    go = st.button("✨ Generate notes", type="primary")

if go:
    try:
        wave, dur = None, None
        if source.startswith("Paste"):
            raw = pasted.strip()
            if not raw:
                st.error("Paste a transcript first.")
                st.stop()
            steps_audio = False
        else:
            if audio_file is not None:
                audio_path = save_upload(audio_file)
            if not audio_path:
                st.error("Add audio first.")
                st.stop()
            steps_audio = True
        with st.status("Working on your lecture...", expanded=True) as status:
            if steps_audio:
                st.write(f"🎧 Whisper ({model_name}) is transcribing the audio...")
                raw = transcribe(audio_path, model_name, LANGS[lang_label])
                wave, dur = waveform(audio_path)
            st.write("🧹 Cleaning the transcript and correcting errors with the language model...")
            lm = load_lm(corpus_text, order) if use_lm else None
            res = npl.process(raw, lm, n_sum, n_kw, digits=use_digits)
            st.write("🧠 Extracting summary, keywords, entities and relations... done")
            status.update(label="Your notes are ready", state="complete", expanded=False)
        st.session_state.update(res=res, ref=reference.strip(), chat=[], title=title, lm_on=use_lm,
                                order=order, corpus=corpus_text, whisper=model_name, wave=wave, dur=dur,
                                show_bts=False)
        st.rerun()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

# ------------------------------------------------------------------ landing ends here when nothing yet
if not res:
    html('<div class="footer">SpeechLens · built with pretrained OpenAI Whisper, NLTK, scikit-learn and spaCy · '
         '21CSE430T Automatic Speech Recognition</div>')
    st.stop()

# ------------------------------------------------------------------ results dashboard
html('<div class="ready">✅ Your notes are ready. Explore the results below.</div>')
ref = st.session_state.get("ref")
wer_txt, wer_sub = "n/a", "add a reference to measure"
if ref:
    w0 = npl.wer_score(ref, res["raw"])["wer"]
    w1 = npl.wer_score(ref, res["corrected"])["wer"]
    wer_txt, wer_sub = f"{100*w0:.1f}% → {100*w1:.1f}%", "word error rate, raw → final"
dur = st.session_state.get("dur")
html(f"""
<div class="grid g4">
<div class="tile"><b>{len(res['corrected'].split())}</b><small>words in the transcript</small></div>
<div class="tile"><b>{res['clean_stats']['fillers'] + res['clean_stats']['repeats']}</b><small>fillers and repeats removed</small></div>
<div class="tile"><b>{len(res['changes'])}</b><small>words fixed by the language model</small></div>
<div class="tile"><b>{wer_txt}</b><small>{wer_sub}</small></div></div>""")
st.write("")

wave = st.session_state.get("wave")
if wave:
    st.markdown(f'<div class="lbl">Your recording · {dur:.0f} s</div>', unsafe_allow_html=True)
    st.area_chart(pd.DataFrame({"loudness": wave}), height=110, color="#2dd4bf")

tabs = st.tabs(["📄 Notes", "🔤 Transcript", "💬 Ask the lecture", "🎓 Study cards", "⬇️ Download"])

# ---- Notes
with tabs[0]:
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown('<div class="lbl">Summary</div>', unsafe_allow_html=True)
        for i, s in enumerate(res["summary"], 1):
            st.markdown(f'<div class="sumcard"><b style="color:#2dd4bf">{i}.</b> {H.escape(s)}</div>',
                        unsafe_allow_html=True)
        if res["relations"]:
            st.markdown('<div class="lbl" style="margin-top:18px">Relations: who does what</div>',
                        unsafe_allow_html=True)
            dot = ('digraph G { bgcolor="transparent"; rankdir=LR; node [shape=box, style="rounded,filled", '
                   'fillcolor="#152038", fontcolor="#e6edf3", color="#2dd4bf", fontsize=11]; '
                   'edge [color="#8b5cf6", fontcolor="#c4b5fd", fontsize=10]; ' + " ".join(
                       f'"{s}" -> "{o}" [label="{v}"];' for s, v, o in res["relations"][:10]
                       if '"' not in s + v + o) + " }")
            st.graphviz_chart(dot)
    with b:
        st.markdown('<div class="lbl">Keywords</div>', unsafe_allow_html=True)
        if res["keywords"]:
            top = max(sc for _, sc in res["keywords"]) or 1
            st.markdown("".join(
                f'<span class="kw" style="font-size:{0.85 + 0.55 * sc / top:.2f}rem">{H.escape(k)}</span>'
                for k, sc in res["keywords"]), unsafe_allow_html=True)
        st.markdown('<div class="lbl" style="margin-top:18px">Entities</div>', unsafe_allow_html=True)
        if res["entities"]:
            st.markdown("".join(
                f'<span class="pill" style="color:{ENT_COLORS.get(l, "#93a4bd")};'
                f'border-color:{ENT_COLORS.get(l, "#93a4bd")}">{H.escape(t)} · {l}</span>'
                for t, l in res["entities"]), unsafe_allow_html=True)
        if not res["entities"]:
            st.caption("No named people, organisations, places or dates in this lecture. The keywords above are its main concepts.")
        else:
            st.caption("Entities come from a model trained on news text, so technical names can be mislabelled.")

# ---- Transcript
with tabs[1]:
    c1, c2, c3 = st.columns(3, gap="medium")
    for col, lab, body in [(c1, "1 · Raw Whisper output", H.escape(res["raw"])),
                           (c2, "2 · After regex clean-up", H.escape(res["cleaned"])),
                           (c3, "3 · After language-model correction", highlight(res["corrected"], res["changes"]))]:
        col.markdown(f'<div class="lbl">{lab}</div><div class="txt">{body}</div>', unsafe_allow_html=True)
    if res["changes"]:
        st.markdown('<div class="lbl" style="margin-top:16px">Words changed by the language model</div>',
                    unsafe_allow_html=True)
        st.markdown("".join(f'<span class="pill" style="color:#fde68a;border-color:#a16207">'
                            f'{H.escape(a)} → {H.escape(b)}</span>' for a, b in res["changes"]),
                    unsafe_allow_html=True)
    if ref:
        st.markdown('<div class="lbl" style="margin-top:22px">Word error rate by stage</div>', unsafe_allow_html=True)
        rows = []
        for label, hyp in [("Whisper output", res["raw"]), ("After clean-up", res["cleaned"]),
                           ("After LM correction", res["corrected"])]:
            w = npl.wer_score(ref, hyp)
            rows.append({"stage": label, "WER %": round(100 * w["wer"], 1), "substitutions": w["S"],
                         "deletions": w["D"], "insertions": w["I"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.markdown('<div class="lbl" style="margin-top:18px">Unigram vs bigram vs trigram</div>',
                    unsafe_allow_html=True)
        cmp = []
        for o, name in [(1, "Unigram (add-one)"), (2, "Bigram (Kneser-Ney)"), (3, "Trigram (Kneser-Ney)")]:
            out, ch = load_lm(st.session_state.get("corpus"), o).correct(res["cleaned"])
            cmp.append({"model": name, "words changed": len(ch), "WER %": round(100 * npl.wer_score(ref, out)["wer"], 1)})
        st.dataframe(pd.DataFrame(cmp), hide_index=True)
    else:
        st.caption("Add a reference transcript to see word error rate and the unigram / bigram / trigram comparison.")

# ---- Ask
with tabs[2]:
    st.caption("Retrieval assistant: finds the lecture sentence most similar to your question. "
               "It only answers from the lecture.")
    qa = npl.QAIndex(npl.split_sentences(res["corrected"]))
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.write(m["text"])
            if m.get("audio"):
                st.audio(m["audio"], format="audio/mp3")
    q_voice = st.audio_input("🎤 Ask by voice") if hasattr(st, "audio_input") else None
    question = st.chat_input("Ask a question about the lecture...")
    voice_id = (getattr(q_voice, "file_id", None) or q_voice.name) if q_voice is not None else None
    if not question and q_voice is not None and st.session_state.get("last_voice") != voice_id:
        st.session_state["last_voice"] = voice_id
        with st.spinner("Transcribing your question..."):
            question = transcribe(save_upload(q_voice), "base", "en")
    if question:
        ans = qa.answer(question)
        st.session_state["chat"] += [{"role": "user", "text": question},
                                     {"role": "assistant", "text": ans["answer"], "audio": npl.speak(ans["answer"])}]
        st.session_state["last_rank"] = (question, qa.rank(question))
        st.rerun()
    if not st.session_state["chat"]:
        st.info("Try: *What does the encoder do?* or *What is word error rate?*")

# ---- Study cards
with tabs[3]:
    cards = npl.study_cards(res)
    st.caption("Each keyword with the lecture sentence that explains it. Try to recall it first, then open the card.")
    if not cards:
        st.write("No cards could be made from this lecture.")
    for i in range(0, len(cards), 2):
        cols = st.columns(2, gap="medium")
        for col, (kw, sent) in zip(cols, cards[i:i + 2]):
            with col:
                with st.expander(f"❓  What is “{kw}”?"):
                    st.markdown(f'<div class="sumcard">{H.escape(sent)}</div>', unsafe_allow_html=True)

# ---- Download
with tabs[4]:
    chat = st.session_state["chat"]
    qa_log = [(chat[i]["text"], chat[i + 1]["text"]) for i in range(0, len(chat) - 1, 2)]
    notes = npl.build_notes(st.session_state.get("title", "Lecture notes"), res, qa_log)
    st.download_button("⬇️ Download notes (.md)", notes, file_name="lecture_notes.md", mime="text/markdown")
    with st.container(border=True):
        st.markdown(notes)

# ------------------------------------------------------------------ reveal behind the scenes (at the end)
html("""
<div class="curious"><h3>Curious how this actually works?</h3>
<p>See every stage with its numbers: the regex rules, language-model probabilities, TF-IDF scores and more.</p></div>""")
_, mid, _ = st.columns([1, 1, 1])
label = "🙈 Hide behind the scenes" if st.session_state.get("show_bts") else "🔍 Reveal behind the scenes"
if mid.button(label, use_container_width=True):
    st.session_state["show_bts"] = not st.session_state.get("show_bts")
    st.rerun()

if st.session_state.get("show_bts"):
    sents = npl.split_sentences(res["cleaned"])
    lm_on = st.session_state.get("lm_on")
    st.markdown('<div class="sec" style="margin-top:20px"><small>Behind the scenes</small>'
                '<h2>What happened to your lecture</h2></div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.tabs(["Clean-up", "N-gram language model", "TF-IDF", "NER & parse", "Q&A and Whisper"])

    with s1:
        st.markdown("Regular expressions, applied in this order:")
        st.code(r"""fillers   : \b(um|uh|erm|you know|i mean|sort of|kind of)\b    -> removed
repeats   : \b(\w+)(\s+\1\b)+                                      -> "the the" -> "the"
numbers   : eighty -> 80, twenty-five percent -> 25%, three point five -> 3.5   (a lone "one" stays a word)
spacing   : \s+([,.!?;:])  and  ([,.!?;:])(?=\w)                   -> fix spaces around punctuation
sentences : capitalise the start of each sentence, add a final full stop""", language="text")
        st.write(res["clean_stats"])

    with s2:
        if not lm_on:
            st.info("The language model was switched off for this run.")
        else:
            lm = load_lm(st.session_state.get("corpus"), st.session_state["order"])
            a, b, c, d = st.columns(4)
            a.metric("N-gram order", st.session_state["order"])
            b.metric("Smoothing", lm.smoothing)
            c.metric("Training sentences", lm.n_sentences)
            d.metric("Vocabulary", len(lm.vocab))
            st.markdown("Each word gets **P(word | previous words)**, estimated from counts in the domain text. "
                        "Unseen words get a tiny floor (1e-6). **Real English words are trusted and never changed**: "
                        "only words that are not in an English dictionary or the domain text are treated as recognition errors.")
            pick = st.selectbox("Sentence to inspect", range(len(sents)),
                                format_func=lambda i: f"{i+1}. {sents[i][:70]}")
            st.dataframe(pd.DataFrame(lm.word_probs(sents[pick])), hide_index=True)
            st.write(f"Perplexity of this sentence: **{lm.perplexity(sents[pick]):.1f}** (lower = more expected)")
            tr = lm.trace(sents[pick])
            if tr:
                st.markdown("**Candidate words tried** (the heard word plus vocabulary words within 1-2 edits). "
                            "The sentence with the best total score is kept:")
                st.dataframe(pd.DataFrame(tr), hide_index=True)
            else:
                st.caption("Every word in this sentence is known, so nothing was replaced.")

    with s3:
        sc = pd.DataFrame(npl.sentence_scores(res["corrected"]))
        st.markdown("Each word gets a TF-IDF weight (how often it occurs in the lecture x how few sentences contain it). "
                    "A sentence scores the sum of its content-word weights, divided by its length to the power 0.7, "
                    "so long sentences do not win by length alone. Questions and very short sentences are not eligible. "
                    "The top sentences, in lecture order, form the summary.")
        st.dataframe(sc, hide_index=True)
        if len(sc):
            st.bar_chart(sc.set_index("#")["score"])

    with s4:
        i = st.selectbox("Sentence", range(len(sents)), format_func=lambda i: f"{i+1}. {sents[i][:70]}", key="parse")
        doc = get_nlp()(sents[i])
        st.dataframe(pd.DataFrame([{"token": t.text, "lemma": t.lemma_, "POS": t.pos_, "dependency": t.dep_,
                                    "head": t.head.text, "entity": t.ent_type_} for t in doc]), hide_index=True)
        st.caption("Relations are subject-verb-object triples read from these dependency labels.")

    with s5:
        lr = st.session_state.get("last_rank")
        if lr:
            st.markdown(f"Question: **{lr[0]}**. Cosine similarity to every lecture sentence:")
            st.dataframe(pd.DataFrame(lr[1]), hide_index=True)
        else:
            st.caption("Ask a question in the assistant tab to see the ranking here.")
        wn = st.session_state.get("whisper")
        st.markdown(f"ASR model used: **Whisper {wn}** ({WHISPER_SIZES.get(wn, '')}).")
        st.page_link("pages/2_Inside_the_model.py", icon="🔬",
                     label="Open “Inside the model”: log-Mel, encoder, decoder, attention and WER experiments")

html('<div class="footer">SpeechLens · built with pretrained OpenAI Whisper, NLTK, scikit-learn and spaCy · '
     '21CSE430T Automatic Speech Recognition</div>')
