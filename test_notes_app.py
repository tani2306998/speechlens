from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=180).run()
assert not at.exception, at.exception
next(r for r in at.radio if r.label == "Input").set_value("Paste a transcript (skip ASR)").run()
ta = at.text_area
ta[0].set_value("um the encoder converts the the log mell spectrogram into contextual vectors. kneser nay smoothing improves an n gram language model. whisper was released by OpenAI in 2022. word error rate measures the accuracy of speech recognition.")
ta[-1].set_value("The encoder converts the log mel spectrogram into contextual vectors. Kneser Ney smoothing improves an n gram language model. Whisper was released by OpenAI in 2022. Word error rate measures the accuracy of speech recognition.")
next(b for b in at.button if "Generate" in b.label).click().run()
assert not at.exception, at.exception
print([t.label for t in at.tabs])
at.chat_input[0].set_value("When was whisper released?").run()
assert not at.exception, at.exception
print([m.markdown[0].value[:60] for m in at.chat_message])
assert not any("Hide behind" in b.label for b in at.button)
next(b for b in at.button if "Reveal" in b.label).click().run()
assert not at.exception, at.exception
print("BTS tabs:", [t.label for t in at.tabs][-5:])
print("OK")
