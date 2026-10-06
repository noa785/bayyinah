"""
التعرف على آيات القرآن الكريم في الكلام.

نص المصحف من مكتبة quran-json (رخصة CC-BY-SA 4.0) بالرسم العثماني.
نحوله لنفس صورة الكتابة المعتادة قبل المقارنة، ثم نطابقه بنفس طريقة الأحاديث.
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from arabic import normalize

MIN_SCORE = 0.62      # أشد من الأحاديث، لأن القرآن ٦٢٣٦ آية والتشابه بينها كثير
EXACT_SCORE = 0.85

# عبارات افتتاحية تتكرر في كل مقطع، ما نعتبرها اقتباساً من القرآن
OPENERS = {
    "بسم الله الرحمن الرحيم",
    "اعوذ بالله من الشيطان الرجيم",
    "الحمد لله رب العالمين",
}
BASMALA = "بسم الله الرحمن الرحيم"


def _is_opener(t):
    t = t.replace("الرحمان", "الرحمن")
    return t in OPENERS or (t.startswith("و") and t[1:] in OPENERS)

STOP = set("من في على الي عن ان او ثم و يا ما لا لم قال يقول تعالي سبحانه وتعالي عز وجل ربنا".split())


def uthmani_to_plain(t: str) -> str:
    """يحول الرسم العثماني للكتابة المعتادة: الصلوٰة إلى الصلاة، ومَٰلك إلى مالك."""
    t = t.replace("ىٰ", "ى").replace("وٰ", "ا").replace("ٰ", "ا")
    t = re.sub("[ۥۦ]", "", t)
    return normalize(t)


def _close(a, b):
    """كلمتين متساويتين، أو بينهم فرق حرف واحد (الرسم العثماني والتفريغ يختلفون في حرف)."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def _ordered_hits(q, h):
    """كم كلمة من q موجودة في h بنفس الترتيب، مع التسامح بحرف."""
    pos, n = 0, 0
    for w in q:
        for j in range(pos, len(h)):
            if _close(w, h[j]):
                pos, n = j + 1, n + 1
                break
    return n


class QuranIndex:
    def __init__(self, path="data/quran.json"):
        self.verses = json.loads(Path(path).read_text(encoding="utf-8"))
        self.norm = [uthmani_to_plain(v["text"]) for v in self.verses]
        # تجزئة بدل قاموس كلمات، عشان الذاكرة على الخادم المجاني
        self.hash = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 20,
                                      alternate_sign=False, norm=None, dtype=np.float32)
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.matrix = self.tfidf.fit_transform(self.hash.transform(self.norm))

    def _score(self, idx, words, content, sim):
        vw = self.norm[idx].split()
        if len(vw) < 3 or _is_opener(self.norm[idx]):
            return 0.0, False
        score, full = sim, False
        # الآية كاملة داخل الكلام
        vcont = [w for w in vw if w not in STOP]
        if len(vcont) >= 3 and _ordered_hits(vcont, words) / len(vcont) >= 0.9:
            score, full = max(score, 0.95), True
        # جزء من آية طويلة
        if len(content) >= 4:
            q = _ordered_hits(content, vw) / len(content)
            if q >= 0.85:
                score = max(score, 0.6 + 0.24 * q)
        return score, full

    def search(self, text):
        """يرجع (الدرجة، [أرقام الآيات]) للآية أو الآيات المتتالية الموجودة في الكلام، أو None."""
        n = normalize(text)
        words = n.split()
        if len(words) < 3 or _is_opener(n):
            return None
        sims = cosine_similarity(self.tfidf.transform(self.hash.transform([n])), self.matrix)[0]
        content = [w for w in words if w not in STOP]
        scored = []
        for idx in sims.argsort()[::-1][:15]:
            sc, full = self._score(int(idx), words, content, float(sims[idx]))
            if sc >= MIN_SCORE:
                scored.append((sc, int(idx), full))
        if not scored:
            return None
        scored.sort(reverse=True)
        # إذا الكلام فيه آيات كاملة متتالية من نفس السورة، نعرضها مع بعض، ونختار أطول تسلسل
        best = None
        for sc, idx, full in scored:
            run = self._extend(idx, words) if full else [idx]
            # نفضّل الآية اللي تغطي أكثر الكلام، عشان آية الكرسي ما تنحسب أول آل عمران
            vw = [w for i in run for w in self.norm[i].split()]
            cov = round(_ordered_hits(content, vw) / max(len(content), 1), 2)
            key = (cov, len(run), sc)
            if best is None or key > best[0]:
                best = (key, sc, run)
        return best[1], best[2]

    def _inside(self, i, words):
        vw = self.norm[i].split()
        return bool(vw) and _ordered_hits(vw, words) / len(vw) >= 0.9

    def _extend(self, idx, words):
        s = self.verses[idx]["s"]
        lo = hi = idx
        while lo - 1 >= 0 and self.verses[lo - 1]["s"] == s and not self.norm[lo - 1].replace("الرحمان", "الرحمن") == BASMALA and self._inside(lo - 1, words):
            lo -= 1
        while hi + 1 < len(self.verses) and self.verses[hi + 1]["s"] == s and self._inside(hi + 1, words):
            hi += 1
        return list(range(lo, hi + 1))

    def run_of(self, qid):
        """يحول رقم مثل Q2:255-257 إلى أرقام الآيات في القائمة."""
        sura, rng = qid[1:].split(":")
        a, b = (int(x) for x in rng.split("-"))
        return [k for k, v in enumerate(self.verses) if v["s"] == int(sura) and a <= v["a"] <= b]

    def words_of(self, run):
        out = set()
        for i in run:
            out |= set(self.norm[i].split())
        return out

    def claim(self, run, score, seg):
        v, last = self.verses[run[0]], self.verses[run[-1]]
        text = " ".join(self.verses[i]["text"] + f" ({self.verses[i]['a']})" for i in run) if len(run) > 1 else v["text"]
        ref = f"الآية {v['a']}" if len(run) == 1 else f"الآيات {v['a']} إلى {last['a']}"
        return {
            "quoted_text": seg.strip()[:300],
            "matched": True,
            "kind": "quran",
            "match_type": "exact" if score >= EXACT_SCORE else "partial",
            "hadith": {
                "id": f"Q{v['s']}:{v['a']}-{last['a']}",
                "text": text,
                "grade_raw": "آية من القرآن الكريم",
                "grade_class": "quran",
                "graded_by": None,
                "source": {"book": f"القرآن الكريم، سورة {v['name']}", "ref": ref},
                "url": None,
                "note": None,
                "narrator": None,
                "takhrij": None,
                "alternative": None,
            },
        }
