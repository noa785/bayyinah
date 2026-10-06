"""
تقييم محرك التحقق من المضمون على مجموعة اختبار.

التشغيل:
    python evaluate.py

يختبر:
1. كل حديث في القاعدة بنصه، هل يرجع حكمه الصحيح
2. نفس الحديث مكتوب بدون تشكيل ومع كلام قبله وبعده، مثل ما ينقال في مقطع
3. نصوص مو موجودة في القاعدة، هل النظام يمتنع بدل ما يحكم غلط
4. أخطاء تفريغ مصطنعة: نغير حرف في كلمة من كل خمس كلمات تقريباً، مثل ما يغلط Whisper
5. جزء من الحديث: أول ستين بالمية من كلماته بس، مثل ما ينقله الناس مختصر

ويطلع الأرقام اللي نحطها في العرض التقديمي.
"""

import json
import random
import time

from arabic import DIACRITICS
from engine import HadithEngine
from verdict import decide, content_block

# نصوص مو موجودة في القاعدة. الصحيح إن النظام يمتنع (غير محسوم)
NEGATIVES = [
    "الجو اليوم حار جدا في الرياض والناس في الاسواق",
    "اشتريت سيارة جديدة الاسبوع الماضي ولونها ابيض",
    "من صام يوم كذا كتب الله له اجر الف شهيد",
    "من قرأ هذه الرسالة وارسلها لعشرة اشخاص سمع خبرا سارا",
    "الحمد لله رب العالمين والصلاة والسلام على اشرف المرسلين",
    "اجتماع الفريق بكره الساعة تسعة الصبح ان شاء الله",
    "قال احد الحكماء العلم نور والجهل ظلام",
    "من نام بعد العصر فاختلس عقله فلا يلومن الا نفسه",
    "صلوا على النبي واكثروا من الدعاء في هذا اليوم",
    "التقنية الحديثة غيرت حياة الناس في كل مكان",
]

EXPECTED = {"authentic": "supported", "weak": "contradicted", "not_authentic": "contradicted",
            "detailed": "contradicted", "unknown": "undetermined"}

NO_AUDIO = {"signal": None}

# حروف يخلط بينها التفريغ الصوتي عادة
CONFUSE = {"ح": "ه", "ه": "ح", "ع": "ا", "ا": "ع", "ص": "س", "س": "ص", "ض": "د", "د": "ض",
           "ط": "ت", "ت": "ط", "ظ": "ز", "ز": "ظ", "ذ": "ز", "ق": "ك", "ك": "ق", "ث": "س"}


def noisy(text, rnd, rate=0.2):
    """يغير حرف واحد في نسبة من الكلمات، أو يحذف حرف."""
    words = DIACRITICS.sub("", text).split()
    out = []
    for w in words:
        if len(w) > 2 and rnd.random() < rate:
            i = rnd.randrange(len(w))
            c = w[i]
            w = w[:i] + CONFUSE.get(c, "") + w[i + 1:]
        out.append(w)
    return " ".join(out)


def partial(text, ratio=0.6):
    words = DIACRITICS.sub("", text).split()
    n = max(4, round(len(words) * ratio))
    return " ".join(words[:n])


def verdict_for(engine, text):
    claims = engine.find_claims(text)
    v, *_ = decide(content_block(claims), NO_AUDIO, None)
    return v, claims


def run():
    e = HadithEngine()
    rows = {"exact": [0, 0], "spoken": [0, 0], "noisy": [0, 0], "partial": [0, 0], "attribution": [0, 0]}
    rnd = random.Random(7)
    wrong = []
    t0 = time.time()

    for h in e.db:
        want = EXPECTED[h["grade_class"]]
        for mode, text in (
            ("exact", h["text"]),
            ("spoken", "يقول النبي صلى الله عليه وسلم " + DIACRITICS.sub("", h["text"]) + " فانتبهوا لهذا"),
            ("noisy", noisy(h["text"], rnd)),
            ("partial", partial(h["text"])),
        ):
            v, claims = verdict_for(e, text)
            ok = v == want
            rows[mode][0] += ok
            rows[mode][1] += 1
            if not ok:
                wrong.append(f"[{mode}] {h['id']} توقعنا {want} وطلع {v}: {h['text'][:40]}")
            # دقة الإسناد: هل الحديث اللي رجع هو نفسه
            if mode == "exact":
                hit = any(c["matched"] and c["hadith"]["id"].replace("-ALT", "") == h["id"].replace("-ALT", "")
                          for c in claims)
                rows["attribution"][0] += hit
                rows["attribution"][1] += 1

    abstain_ok = 0
    for t in NEGATIVES:
        v, _ = verdict_for(e, t)
        abstain_ok += v == "undetermined"
        if v != "undetermined":
            wrong.append(f"[negative] توقعنا امتناع وطلع {v}: {t}")

    # آيات القرآن: عينة ثابتة من ٢٠٠ آية فيها خمس كلمات أو أكثر، بالكتابة المعتادة ومع أخطاء تفريغ
    from quran import uthmani_to_plain
    q_ok, q_noisy_ok, q_n = 0, 0, 0
    if e.quran:
        pool = [i for i, v in enumerate(e.quran.verses) if len(e.quran.norm[i].split()) >= 5]
        for i in random.Random(11).sample(pool, 200):
            v = e.quran.verses[i]
            plain = uthmani_to_plain(v["text"])
            for mode, text in (("plain", plain), ("noisy", noisy(plain, rnd))):
                claims = e.find_claims(text)
                # الآيات المتكررة بنفس اللفظ في أكثر من سورة تُحسب صحيحة إذا رجع أي موضع منها
                hit = any(c["matched"] and c.get("kind") == "quran"
                          and e.quran.norm[i] in {e.quran.norm[k] for k in e.quran.run_of(c["hadith"]["id"])}
                          for c in claims)
                if mode == "plain":
                    q_ok += hit
                else:
                    q_noisy_ok += hit
                if not hit and mode == "plain":
                    wrong.append(f"[quran] {v['s']}:{v['a']} ما انعرفت: {plain[:40]}")
            q_n += 1

    ms = (time.time() - t0) * 1000 / (len(e.db) * 4 + len(NEGATIVES) + 2 * q_n)
    pct = lambda a: f"{100 * a[0] / max(a[1], 1):.1f}% ({a[0]}/{a[1]})"
    report = {
        "عدد الأحاديث في القاعدة": len(e.db),
        "دقة الحكم بالنص المطابق": pct(rows["exact"]),
        "دقة الحكم بالنص المنطوق مع كلام قبله وبعده": pct(rows["spoken"]),
        "دقة الحكم مع أخطاء تفريغ مصطنعة": pct(rows["noisy"]),
        "دقة الحكم على جزء من الحديث": pct(rows["partial"]),
        "دقة الإسناد (رجع نفس الحديث)": pct(rows["attribution"]),
        "الامتناع الصحيح عند غياب النص": pct([abstain_ok, len(NEGATIVES)]),
        "التعرف على الآية وموضعها (عينة 200 آية)": pct([q_ok, q_n]),
        "التعرف على الآية مع أخطاء تفريغ مصطنعة": pct([q_noisy_ok, q_n]),
        "متوسط زمن التحقق للنص": f"{ms:.0f} ملي ثانية",
    }
    for k, v in report.items():
        print(f"{k}: {v}")
    if wrong:
        print("\nالحالات اللي ما ضبطت:")
        print("\n".join(wrong))
    with open("data/eval_report.json", "w", encoding="utf-8") as f:
        json.dump({"report": report, "errors": wrong}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run()
