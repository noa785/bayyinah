"""
محرك التحقق من المضمون.

يبحث عن الحديث في القاعدة المغلقة فقط، ويرجع الحكم كما هو من المصدر.
ما فيه أي نموذج لغوي يكتب حكم من عنده.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from arabic import QUOTE_MARKERS, candidate_segments, normalize
from quran import QuranIndex, _close

# حدود المطابقة. اللي تحت MIN_SCORE يعتبر غير موجود في القاعدة.
EXACT_SCORE = 0.85
MIN_SCORE = 0.55

# كلمات شائعة ما نحسبها في المطابقة
STOP = set("من في على الى عن ان او ثم قال قالوا قلنا الله رسول النبي صلي عليه وسلم ما لا لم و يا هذا هذه ذلك التي الذي كان".split())


def _in_order(query_words, hadith_words):
    """يتأكد إن كلمات الاقتباس جاية بنفس ترتيبها في الحديث."""
    pos = 0
    for w in query_words:
        try:
            pos = hadith_words.index(w, pos) + 1
        except ValueError:
            return False
    return True


def _cov(a, b):
    """نسبة كلمات a الموجودة في b مع التسامح بحرف."""
    if not a:
        return 0.0
    return sum(1 for w in a if any(_eq(w, x) for x in b)) / len(a)


def _eq(a, b):
    """نفس الكلمة مع التسامح بحرف، ومع واو أو فاء العطف في أولها (خير ووخير)."""
    if _close(a, b):
        return True
    if b[:1] in "وف" and len(b) > 3 and _close(a, b[1:]):
        return True
    return a[:1] in "وف" and len(a) > 3 and _close(a[1:], b)


# كلمات وظيفية تتكرر في كل نص، ما نحسبها في بوابة الكلمات
FUNC = set("الا اذا كل بعد قبل حتى انما لما له لها لهم به بها فيه هو هي ان قد ثم او الي علي عن مع".split())


def _word_gate(query_content, hadith_content, need=0.7):
    q = list(dict.fromkeys(w for w in query_content if w not in FUNC))
    h = list(dict.fromkeys(w for w in hadith_content if w not in FUNC))
    if not q or not h:
        return False
    return max(_cov(q, h), _cov(h, q)) >= need


class HadithEngine:
    def __init__(self, path="data/hadiths.json"):
        self.db = json.loads(Path(path).read_text(encoding="utf-8"))
        self.by_id = {h["id"]: h for h in self.db}
        self.norm = [normalize(h["text"]) for h in self.db]
        # مقارنة على مستوى الحروف عشان تتحمل اختلاف الإملاء والتشكيل
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.norm)
        # آيات القرآن الكريم، عشان إذا ورد في المقطع آية نعرّفها ولا نقول غير محسوم
        qpath = Path(path).with_name("quran.json")
        self.quran = QuranIndex(qpath) if qpath.exists() else None

    def _containment(self, seg_words, idx):
        """نسبة كلمات الحديث الموجودة في المقطع وبنفس الترتيب، مفيدة لما الحديث قصير والكلام طويل."""
        hw = [w for w in self.norm[idx].split() if w not in STOP]
        if len(hw) < 3:
            return 0.0
        s = set(seg_words)
        ratio = sum(1 for w in hw if w in s) / len(hw)
        if ratio < 0.9 or not _in_order([w for w in hw if w in s], seg_words):
            return 0.0
        return ratio

    def search(self, text, top_k=3):
        """يرجع أفضل المطابقات لمقطع واحد."""
        n = normalize(text)
        if not n:
            return []
        sims = cosine_similarity(self.vec.transform([n]), self.matrix)[0]
        words = n.split()
        content = [w for w in words if w not in STOP]
        results = []
        for idx in sims.argsort()[::-1][: max(top_k, 8)]:
            cont = self._containment(words, idx)
            score = max(float(sims[idx]), cont if len(self.norm[idx].split()) >= 3 else 0.0)
            # لما المستخدم يقتبس أول الحديث الطويل بس: كم من كلماته موجودة في الحديث بالترتيب
            if len(content) >= 4:
                hw = self.norm[idx].split()
                hset = set(hw)
                qcont = sum(1 for w in content if w in hset) / len(content)
                if qcont >= 0.85 and _in_order(content, hw):
                    score = max(score, 0.6 + 0.24 * qcont)  # تطلع "partial" لأنها جزء من الحديث
            # بوابة الكلمات: لازم أغلب كلمات الحديث موجودة في الكلام، أو أغلب كلمات الكلام موجودة في الحديث.
            # تمنع إن «الحياء من الإيمان» ينطابق مع «النظافة من الإيمان» بسبب كلمة مشتركة
            if not _word_gate(content, [w for w in self.norm[idx].split() if w not in STOP]):
                score = 0.0
            results.append((score, idx))
        results.sort(reverse=True)
        return results[:top_k]

    def find_claims(self, text, max_claims=5, kind="auto"):
        """يطلع كل الأحاديث الموجودة في النص، ولو ما لقى شي يرجع النص كادعاء غير موجود."""
        best = {}
        # kind: "hadith" يبحث في الأحاديث بس، "quran" في القرآن بس، "auto" في الاثنين
        for seg in (candidate_segments(text) if kind != "quran" else []):
            # المقاطع القصيرة جدا (كلمتين) نطلب لها تطابق أعلى عشان ما تطلع نتائج غلط
            limit = MIN_SCORE if len(normalize(seg).split()) >= 3 else 0.8
            hits = self.search(seg, top_k=1)
            if not hits or hits[0][0] < limit:
                continue
            score, idx = hits[0]
            hid = self.db[idx]["id"]
            if hid not in best or score > best[hid][0]:
                best[hid] = (score, seg)

        if self.quran and kind != "hadith":
            for seg in candidate_segments(text):
                hit = self.quran.search(seg)
                if hit:
                    key = "Q#" + ",".join(map(str, hit[1]))
                    if key not in best or hit[0] > best[key][0]:
                        best[key] = (hit[0], seg)

        # إذا الحديث وبديله الصحيح طلعوا مع بعض، ناخذ الأقوى بس من كل زوج
        chosen = sorted(best.items(), key=lambda kv: (-kv[1][0], -kv[0].count(",")))
        claims, used_roots, used_segs, used_verses = [], set(), [], set()
        for hid, (score, seg) in chosen:
            root = hid.replace("-ALT", "")
            if root in used_roots:
                continue
            words = set(normalize(seg).split())
            # إذا نفس الكلام انطابق مع حديث أقوى، ما نحسبه مرة ثانية
            if any(len(words & u) / max(len(words), 1) > 0.5 for u in used_segs):
                continue
            used_roots.add(root)
            used_segs.append(words)
            if hid.startswith("Q#"):
                run = [int(i) for i in hid[2:].split(",")]
                if set(run) <= used_verses:
                    continue
                used_verses |= set(run)
                claims.append(self.quran.claim(run, score, seg))
            else:
                claims.append(self._claim(hid, score, seg))
            if len(claims) >= max_claims:
                break

        # جزء قصير من أول حديث طويل، مثل «إنما الأعمال بالنيات»: تشابه الحروف يضعف مع طول الحديث،
        # فنقبله إذا وردت كلماته كلها متتالية في الحديث وبنفس الترتيب
        if not claims and kind != "quran":
            hit = self._short_part(text)
            if hit is not None:
                claims.append(self._claim(self.db[hit]["id"], EXACT_SCORE, text))

        if not claims:
            claims.append({"quoted_text": text.strip()[:300], "matched": False, "match_type": None, "hadith": None})
        elif all(c.get("kind") == "quran" for c in claims):
            # الآية ثابتة، بس لو معها كلام كثير ما لقيناه ما نقول إن المقطع كله مؤيد
            covered = set()
            for hid, _ in chosen:
                if hid.startswith("Q#"):
                    covered |= self.quran.words_of([int(i) for i in hid[2:].split(",")])
            rest = [w for w in normalize(text).split() if w not in STOP and not any(_close(w, c) for c in covered)]
            if len(rest) >= 6:
                claims.append({"quoted_text": " ".join(rest)[:300], "matched": False, "match_type": None, "hadith": None})
        return claims

    def _short_part(self, text):
        q = [w for w in normalize(QUOTE_MARKERS.sub(" ", text)).split() if w not in STOP]
        if not (2 <= len(q) <= 8) or len("".join(q)) < 9:
            return None
        found = []
        for idx, h in enumerate(self.norm):
            hw = [w for w in h.split() if w not in STOP]
            for i in range(len(hw) - len(q) + 1):
                if all(_eq(a, b) for a, b in zip(q, hw[i:i + len(q)])):
                    found.append(idx)
                    break
        roots = {self.db[i]["id"].replace("-ALT", "") for i in found}
        return found[0] if len(roots) == 1 else None

    def _claim(self, hid, score, seg):
        h = self.by_id[hid]
        alt = h.get("alternative")
        return {
            "quoted_text": seg.strip()[:300],
            "matched": True,
            "match_type": "exact" if score >= EXACT_SCORE else "partial",
            "hadith": {
                "id": h["id"],
                "text": h["text"],
                "grade_raw": h["grade_raw"],
                "grade_class": h["grade_class"],
                "graded_by": h.get("graded_by"),
                "source": h.get("source") or {"book": None, "ref": None},
                "url": h.get("url"),
                "note": h.get("note"),
                "narrator": h.get("narrator") or None,
                "takhrij": h.get("takhrij") or None,
                "alternative": (
                    {
                        "text": alt["text"],
                        "grade_raw": alt.get("grade_raw"),
                        "graded_by": alt.get("graded_by"),
                        "source": alt.get("source"),
                        "url": alt.get("url"),
                    }
                    if alt and alt.get("text")
                    else None
                ),
            },
        }
