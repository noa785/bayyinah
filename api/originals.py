"""
المطابقة مع الأصل: أرشيف تجريبي موثق لعدد محدد من العلماء.

نقابل الكلام المفرّغ بالنص الكامل للفتوى أو الدرس كما في الموقع الرسمي للعالم، فإذا وُجد:
- نحدد موضعه في الأصل، ونعرض ما قبله وما بعده.
- ونتحقق: هل في الأصل قبل المقطع أو بعده مباشرة شرط أو استثناء لم يرد في المقطع؟
  فإن وُجد، فقد يكون المقطع مقتطعاً على نحو يغيّر المعنى، فنمتنع ونحيل إلى المختص.

ما نولّد أي كلام: كل ما يُعرض منقول بلفظه من الموقع الرسمي مع رابطه.
"""

import json
from difflib import SequenceMatcher
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from arabic import normalize

DATA = Path(__file__).parent / "data" / "originals.json"
CONTEXT = 30          # كم كلمة نعرض قبل المقطع وبعده
MIN_WORDS = 8         # أقل عدد كلمات متطابقة نقبل به
MIN_COVERAGE = 0.6    # نسبة كلمات المقطع اللي لازم نلقاها في الأصل بالترتيب

# كلمات تبدأ شرطاً أو استثناءً قد يتغير المعنى بحذفه (بعد التطبيع)
# ننظر فقط في الكلمات القليلة التي تلي الكلام مباشرة، لأن الاستدراك المحذوف يأتي عادة بعده فوراً
AFTER_WINDOW, BEFORE_WINDOW = 12, 4
AFTER_MARKERS = {"اذا", "لكن", "ولكن", "لكنه", "ولكنه", "لكنها", "ولكنها", "الا", "بشرط", "بشرطه", "مالم", "الااذا", "بخلاف", "سوي", "عدا", "يستثني", "ويستثني"}
BEFORE_MARKERS = {"اذا", "لو", "اذاكان", "انكان", "بشرط"}
# عبارات تدل على أن الكلام قولٌ ينقله العالم عن غيره، لا رأيه هو
REPORTED = (("بعض", "اهل", "العلم"), ("قال", "بعضهم"), ("وقال", "بعضهم"), ("ذهب", "بعض"), ("وذهب", "بعض"),
            ("قال", "اخرون"), ("وقال", "اخرون"), ("قال", "قوم"), ("وقالوا",), ("قالوا",), ("يري", "بعض"), ("زعم",), ("وزعم",))
PHRASES_AFTER = (("ما", "لم"), ("الا", "اذا"), ("الا", "ان"), ("غير", "ان"), ("بشرط", "ان"))


def _tokens(text):
    """يرجع كلمات النص الأصلي، ومقابل كل كلمة صيغتها بعد التطبيع."""
    raw = text.split()
    norm = [normalize(w) for w in raw]
    keep = [(r, n) for r, n in zip(raw, norm) if n]
    return [r for r, _ in keep], [n for _, n in keep]


class OriginalsArchive:
    def __init__(self, path=DATA):
        self.docs = []
        if Path(path).exists():
            for d in json.loads(Path(path).read_text(encoding="utf-8")):
                raw, norm = _tokens(d["text"])
                if len(norm) >= MIN_WORDS:
                    self.docs.append({**d, "raw": raw, "norm": norm})
        self.vec = None
        if self.docs:
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            self.mat = self.vec.fit_transform([" ".join(d["norm"]) for d in self.docs])

    def __len__(self):
        return len(self.docs)

    def _align(self, q, doc):
        sm = SequenceMatcher(None, q, doc["norm"], autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size >= 2]
        hit = sum(b.size for b in blocks)
        if not blocks:
            return 0, 0, None
        # نأخذ أطول سلسلة متصلة من الكتل اللي ما بينها فجوة كبيرة في الأصل
        best, cur = [], [blocks[0]]
        for b in blocks[1:]:
            prev = cur[-1]
            if b.b - (prev.b + prev.size) <= 12:
                cur.append(b)
            else:
                best = max(best, cur, key=lambda x: sum(y.size for y in x))
                cur = [b]
        best = max(best, cur, key=lambda x: sum(y.size for y in x))
        hit = sum(b.size for b in best)
        span = (best[0].b, best[-1].b + best[-1].size)
        return hit, hit / max(1, len(q)), span

    def match(self, text):
        if not self.docs or not text:
            return None
        q = normalize(text).split()
        if len(q) < MIN_WORDS:
            return None
        sims = cosine_similarity(self.vec.transform([" ".join(q)]), self.mat)[0]
        best = None
        for i in sims.argsort()[::-1][:3]:
            hit, cov, span = self._align(q, self.docs[i])
            if span and hit >= MIN_WORDS and cov >= MIN_COVERAGE and (best is None or cov > best[1]):
                best = (i, cov, span)
        if best is None:
            return None
        doc, (s, e) = self.docs[best[0]], best[2]
        before_n, after_n = doc["norm"][max(0, s - 12):s], doc["norm"][e:e + 25]
        omitted = None
        near = after_n[:AFTER_WINDOW]
        lead = doc["norm"][max(0, s - 12):s] + doc["norm"][s:min(e, s + 6)]
        reported = any(lead[k:k + len(p)] == list(p) for p in REPORTED for k in range(len(lead) - len(p) + 1))
        # الاستدراك المحذوف بعد الكلام أوضح دليل على الاقتطاع، فيُقدَّم، ثم نقل الشيخ عن غيره
        if any(w in AFTER_MARKERS for w in near) or any(
                near[k:k + 2] == list(p) for p in PHRASES_AFTER for k in range(len(near) - 1)):
            omitted = "after"
        elif reported:
            omitted = "reported"
        elif any(w in BEFORE_MARKERS for w in before_n[-BEFORE_WINDOW:]) and s > 0:
            omitted = "before"
        return {
            "doc": doc, "start": s, "end": e, "coverage": best[1], "omitted": omitted,
            "before": " ".join(doc["raw"][max(0, s - CONTEXT):s]),
            "quoted": " ".join(doc["raw"][s:e]),
            "after": " ".join(doc["raw"][e:e + CONTEXT]),
        }

    def evidence(self, text):
        """قرينة المطابقة مع الأصل بصيغة التقرير."""
        if not self.docs:
            return {"status": "not_found", "label": "أرشيف الدروس الأصلية قيد البناء، فلم تُجرَ المطابقة",
                    "source": None, "before": None, "after": None}
        m = self.match(text)
        if m is None:
            return {"status": "not_found",
                    "label": f"لم يُعثر على الكلام في الأرشيف التجريبي المنقول من المواقع الرسمية للعلماء (عدد نصوصه {len(self.docs)})، وهذا لا يعني أنه غير صحيح",
                    "source": None, "before": None, "after": None}
        d = m["doc"]
        source = {"scholar": d.get("scholar"), "title": d.get("title"), "url": d.get("url"), "audio": d.get("audio")}
        whole = m["start"] == 0 and m["end"] >= len(d["norm"]) - 2
        if m["omitted"] == "reported":
            return {"status": "context_omitted", "source": source,
                    "before": m["before"], "quoted": m["quoted"], "after": m["after"],
                    "label": f"الكلام موجود في {d.get('title') or 'الأصل'} على الموقع الرسمي، لكنه قولٌ ينقله الشيخ عن غيره وليس رأيه، فقد يكون مقتطعاً على نحو يغيّر نسبته"}
        if m["omitted"]:
            where = "بعده" if m["omitted"] == "after" else "قبله"
            return {"status": "context_omitted", "source": source,
                    "before": m["before"], "quoted": m["quoted"], "after": m["after"],
                    "label": f"الكلام موجود في {d.get('title') or 'الأصل'} على الموقع الرسمي، وفي الأصل {where} شرط أو استثناء لم يرد في المقطع، فقد يكون مقتطعاً على نحو يغيّر المعنى"}
        return {"status": "found", "source": source,
                "before": m["before"], "quoted": m["quoted"], "after": m["after"],
                "label": ("الكلام منقول كاملاً من " if whole else "الكلام موجود بلفظه في ") + f"{d.get('title') or 'الأصل'} على الموقع الرسمي، ويُعرض ما قبله وما بعده"}
