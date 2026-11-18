import time
from notes_pipeline import *

lm = LectureLM.from_file()
raw = ("um the encoder converts the the log mell spectrogram into contextual vectors. "
       "you know the decoder generates the transcript one token at a time. "
       "kneser ney smoothing improves an n gram language model. "
       "word error rate measures the acuracy of speach recognition. "
       "whisper was released by OpenAI in 2022.")
ref = ("The encoder converts the log mel spectrogram into contextual vectors. "
       "The decoder generates the transcript one token at a time. "
       "Kneser Ney smoothing improves an n gram language model. "
       "Word error rate measures the accuracy of speech recognition. "
       "Whisper was released by OpenAI in 2022.")
res = process(raw, lm)
print(res["clean_stats"], res["changes"])
w0, w1 = wer_score(ref, res["cleaned"])["wer"], wer_score(ref, res["corrected"])["wer"]
print("WER cleaned", round(w0, 3), "corrected", round(w1, 3)); assert w1 < w0
ok = "The decoder generates the transcript one token at a time."
assert lm.correct(ok)[0] == ok
for real in ["we see some examples of each", "it's just the magnitude", "we care about the points in space"]:
    assert lm.correct(real)[1] == [], real           # real words must not be changed
print("summary", res["summary"]); print("kw", res["keywords"][:4]); print("rel", res["relations"])
qa = QAIndex(split_sentences(res["corrected"]))
assert "2022" in qa.answer("When was whisper released?")["answer"]
print("OK")


def test_digits():
    from notes_pipeline import words_to_digits as w
    assert w("we use eighty mel bins")[0] == "we use 80 mel bins"
    assert w("twenty-five percent")[0] == "25%"
    assert w("three point five")[0] == "3.5"
    assert w("twenty twenty-two")[0] == "2022"
    assert w("twenty five hundred")[0] == "2500"
    assert w("no one knows")[0] == "no one knows"
    assert w("pi is 3.5 e.g. x")[0] == "pi is 3.5 e.g. x"

test_digits(); print("digits ok")


def test_symbols():
    from notes_pipeline import clean_transcript as c
    assert c("two plus two equals four")[0] == "2 + 2 = 4."
    assert c("five minus three is less than ten")[0] == "5 - 3 < 10."
    assert c("ten divided by two")[0] == "10 ÷ 2."
    assert c("x squared plus y squared equals r squared")[0] == "X² + y² = r²."
    assert c("it was a plus point")[0] == "It was a plus point."
    assert c("three times faster")[0] == "3 times faster."
    assert c("the cost is five dollars")[0] == "The cost is $5."

test_symbols(); print("symbols ok")
