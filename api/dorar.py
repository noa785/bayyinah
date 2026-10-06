"""
البحث في الموسوعة الحديثية بموقع الدرر السنية، عبر واجهتها المنشورة للمطورين:
https://dorar.net/article/389

نستخدمها فقط إذا ما لقينا النص في قاعدتنا المغلقة. وما نقبل نتيجة إلا إذا طابق نصها
الكلام المدخل في أغلب كلماته وبالترتيب، عشان ما ننسب حكم حديث لحديث ثاني يشبهه.
الحكم ينقل بلفظه مع اسم المحدث والمصدر، ولا يولّد النظام حكماً من عنده.
"""

import html
import re
import time
import urllib.parse
import urllib.request
import json

from arabic import normalize, QUOTE_MARKERS
from quran import _close

API = "https://dorar.net/dorar_api.json?skey="
TIMEOUT = 8
_cache = {}

# كلمات وظيفية ما نحسبها في المطابقة
STOP = set("من في على الى الي عن ان او ثم و يا ما لا لم قال قالوا الله رسول النبي صلي عليه وسلم هذا هذه ذلك التي الذي كان".split())

AUTHENTIC = ("صحيح", "حسن", "ثابت", "إسناده جيد", "اسناده جيد")
NOT_AUTH = ("موضوع", "لا أصل له", "لا اصل له", "باطل", "منكر", "مكذوب", "لا يصح", "ليس بحديث", "كذب")
WEAK = ("ضعيف", "فيه ضعف", "مرسل", "منقطع", "واه")


def grade_class(g):
    g = g or ""
    if any(k in g for k in NOT_AUTH):
        return "not_authentic"
    if any(k in g for k in WEAK):
        return "weak"
    if any(k in g for k in AUTHENTIC):
        return "authentic"
    return "unknown"


def _text(fragment):
    t = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def parse(result_html):
    """يحوّل نتيجة الواجهة (HTML) إلى قائمة أحاديث: النص والراوي والمحدث والمصدر والرقم والحكم."""
    out = []
    parts = re.split(r'<div[^>]*class="hadith-info"[^>]*>', result_html)
    for i in range(1, len(parts)):
        # نص الحديث آخر div قبل خانة المعلومات
        before = parts[i - 1]
        m = re.findall(r'<div[^>]*class="hadith"[^>]*>(.*?)</div>', before, re.S)
        if not m:
            m = re.findall(r"<div[^>]*>(.*?)</div>", before, re.S)
        if not m:
            continue
        text = re.sub(r"^\s*\d+\s*-\s*", "", _text(m[-1]))
        info = parts[i].split("</div>")[0]
        fields = {}
        for label, value in re.findall(r'<span[^>]*class="info-subtitle"[^>]*>(.*?)</span>(.*?)(?=<span[^>]*class="info-subtitle"|$)', info, re.S):
            fields[_text(label).rstrip(":： ").strip()] = _text(value)
        out.append({
            "text": text,
            "rawi": fields.get("الراوي", ""),
            "mohdith": fields.get("المحدث", ""),
            "book": fields.get("المصدر", ""),
            "ref": fields.get("الصفحة أو الرقم", ""),
            "grade": fields.get("خلاصة حكم المحدث", "").strip("[] "),
        })
    return out


def _query(text):
    t = QUOTE_MARKERS.sub(" ", text)
    words = normalize(t).split()
    return " ".join(words[:12])


def fetch(q):
    if q in _cache:
        return _cache[q]
    req = urllib.request.Request(API + urllib.parse.quote(q), headers={"User-Agent": "Bayyinah/1.0 (+https://bayyinah-web.onrender.com)"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.loads(r.read().decode("utf-8"))
    res = parse(((data or {}).get("ahadith") or {}).get("result") or "")
    _cache[q] = res
    return res


def _hits(a, b):
    """كم كلمة من a موجودة في b بنفس الترتيب، مع التسامح بحرف."""
    pos, n = 0, 0
    for w in a:
        for j in range(pos, len(b)):
            if _close(w, b[j]):
                pos, n = j + 1, n + 1
                break
    return n


def matches(query_text, hadith_text):
    q = [w for w in normalize(QUOTE_MARKERS.sub(" ", query_text)).split() if w not in STOP]
    h = [w for w in normalize(hadith_text).split() if w not in STOP]
    if len(q) < 2 or not h:
        return False
    # المدخل جزء من الحديث، أو الحديث كامل داخل المدخل
    return _hits(q, h) / len(q) >= 0.85 or (len(h) >= 3 and _hits(h, q) / len(h) >= 0.9)


def lookup(text):
    """يرجع ادعاء واحد (claim) بصيغة المحرك، أو None إذا ما لقينا مطابقة مؤكدة."""
    q = _query(text)
    if len(q.split()) < 2:
        return None
    try:
        results = fetch(q)
    except Exception:
        return None
    good = [r for r in results if r["text"] and r["grade"] and matches(text, r["text"])]
    if not good:
        return None
    # نفضّل الأحاديث اللي نصها قريب من المدخل، لا اللي فيها الكلمات ضمن حديث أطول بكثير
    qw = [w for w in normalize(QUOTE_MARKERS.sub(" ", text)).split() if w not in STOP]
    def tight(r):
        h = [w for w in normalize(r["text"]).split() if w not in STOP]
        return h and _hits(h, qw) / len(h) >= 0.6
    good = [r for r in good if tight(r)] or good
    # نقارن «ثابت» مقابل «لا يثبت»: الضعيف والموضوع كلاهما لا يثبت، فما نعتبرهم اختلافاً
    known = [r for r in good if grade_class(r["grade"]) != "unknown"] or good
    groups = {"ok" if grade_class(r["grade"]) == "authentic" else "bad" for r in known}
    first = known[0]
    if len(groups) == 1:
        gc, grade_raw = grade_class(first["grade"]), first["grade"]
    else:
        gc, grade_raw = "mixed", "اختلفت أحكام المحدثين"
    others = "؛ ".join(f'{r["mohdith"]}: {r["grade"]} ({r["book"]} {r["ref"]})'.strip() for r in good[:4])
    return {
        "quoted_text": text.strip()[:300],
        "matched": True,
        "kind": "dorar",
        "match_type": "exact",
        "hadith": {
            "id": "DORAR",
            "text": first["text"],
            "grade_raw": grade_raw,
            "grade_class": gc,
            "graded_by": first["mohdith"] or None,
            "source": {"book": first["book"] or None, "ref": first["ref"] or None},
            "url": "https://dorar.net/hadith/search?q=" + urllib.parse.quote(q),
            "note": "من الموسوعة الحديثية في موقع الدرر السنية، وليس من القاعدة التي راجعتها متخصصة الفريق.",
            "narrator": first["rawi"] or None,
            "takhrij": ("أحكام المحدثين: " + others) if len(good) > 1 else None,
            "alternative": None,
        },
    }
