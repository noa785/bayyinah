"""
قواعد الحكم النهائي ودمج القرائن.

المبدأ: ما نقول "مؤيد" أو "مخالف" إلا بقرينة واضحة من المصدر، وغير كذا نمتنع ونحيل للمختص.
"""

TITLES = {
    "supported": "مؤيَّد بالمصدر",
    "contradicted": "مخالف للمصدر",
    "undetermined": "غير محسوم",
}

DISCLAIMER = "التقرير قرائن مساعدة وليس حكماً قطعياً، والمرجع في الحكم أهل العلم."
NOT_FOUND_NOTE = "عدم العثور على النص في قاعدتنا لا يعني أنه موضوع. "

BAD = {"not_authentic", "weak", "detailed"}


def decide(content, auth, duration, prov=None, original=None):
    """يرجع (verdict, summary, abstain_reason, disclaimer)."""
    prov = prov or {}
    original = original or {}
    claims = content["claims"]
    matched = [c for c in claims if c["matched"]]
    classes = [c["hadith"]["grade_class"] for c in matched]
    synthetic = auth.get("signal") == "likely_synthetic"

    # 0. توقيع اعتماد المحتوى (C2PA) السليم يصرّح بأن المقطع مولَّد: تصريح من أداة التوليد نفسها
    if prov.get("status") == "ai_declared" and prov.get("source") == "c2pa":
        extra = "، والحديث الوارد لا يثبت" if any(c in BAD for c in classes) else ""
        return ("contradicted", f"توقيع اعتماد المحتوى في الملف يصرّح بأن المقطع مولَّد بالذكاء الاصطناعي{extra}", None, DISCLAIMER)

    # 1. صوت مولَّد بقرينة قوية: مخالف مهما كان المضمون
    if synthetic:
        extra = "، والحديث الوارد لا يثبت" if any(c in BAD for c in classes) else ""
        return ("contradicted", f"مع المقطع قرائن قوية على توليد الصوت{extra}", None, DISCLAIMER)

    # 2. في المقطع حديث لا يثبت
    if any(c in BAD for c in classes):
        n = sum(1 for c in classes if c in BAD)
        what = "الحديث الوارد في المقطع لا يثبت" if n == 1 else f"في المقطع {n} أحاديث لا تثبت"
        if "detailed" in classes and n == 1:
            what = "الحديث الوارد في المقطع لا يثبت بهذا اللفظ أو النسبة، وفي الحكم تفصيل"
        return ("contradicted", what, None, DISCLAIMER)

    # 2أ. بيانات الملف تذكر أداة توليد معروفة: تنبيه قابل للتعديل، فنمتنع ونحيل للمختص
    if prov.get("status") == "ai_declared":
        tool = f" ({prov['tool']})" if prov.get("tool") else ""
        return ("undetermined", f"بيانات الملف تذكر أداة توليد معروفة{tool}، فيُحال المقطع إلى المختص", "generator_metadata", DISCLAIMER)

    # 2ب. في الملف توقيع لا يطابق محتواه: قد يكون عُدّل أو اقتُطع بعد توقيعه
    if prov.get("status") == "signed_invalid":
        return ("undetermined", "في الملف توقيع اعتماد لا يطابق محتواه، فقد يكون عُدّل بعد توقيعه، ويُحال إلى المختص", "signature_mismatch", DISCLAIMER)

    # إذا كان كل ما في المقطع حديثاً أو آية تعرّفنا عليها، فالحكم لها، والأصل يُعرض للسياق فقط
    whole_text_known = bool(claims) and all(c["matched"] for c in claims)
    if whole_text_known:
        original = {}

    # 2ج. الكلام موجود في الأصل، وفيه قبله أو بعده شرط أو استثناء لم يرد في المقطع
    if original.get("status") == "context_omitted":
        if "ينقله الشيخ عن غيره" in (original.get("label") or ""):
            return ("undetermined", "الكلام موجود في الأصل، لكنه قولٌ ينقله الشيخ عن غيره وليس رأيه، ويُحال إلى المختص", "reported_view", DISCLAIMER)
        return ("undetermined", "الكلام موجود في الأصل، لكن حُذف منه شرط أو استثناء قد يغيّر المعنى، ويُحال إلى المختص", "context_omitted", DISCLAIMER)

    # 2د. الكلام موجود بلفظه في الموقع الرسمي للعالم، وما فيه حديث لا يثبت
    if original.get("status") == "found" and not any(c in ("mixed", "unknown") for c in classes):
        who = (original.get("source") or {}).get("scholar")
        where = f"موقع {who} الرسمي" if who else "الموقع الرسمي للعالم"
        return ("supported", f"الكلام المنسوب موجود بلفظه في {where}، ويُعرض ما قبله وما بعده", None, DISCLAIMER)

    # 3أ. اختلف المحدثون في الحكم على الحديث: نعرض أقوالهم ونحيل للمختص
    if any(c == "mixed" for c in classes):
        return ("undetermined", "اختلفت أحكام المحدثين في الحديث الوارد، ويُحال إلى المختص", "mixed_grades", DISCLAIMER)

    # 3. حكم غير مصنف في القاعدة: نمتنع
    if any(c == "unknown" for c in classes):
        if any(c.get("kind") == "dorar" for c in matched):
            return ("undetermined", "أحكام المحدثين في الموسوعة الحديثية على هذه الرواية تحتاج نظر المختص", "unknown_grade", DISCLAIMER)
        return ("undetermined", "حكم الحديث في القاعدة يحتاج مراجعة المختص", "unknown_grade", DISCLAIMER)

    # 4. ما لقينا شي في القاعدة: نمتنع
    if not matched:
        reason = "no_match"
        if duration is not None and duration < 6:
            reason = "short_audio_and_no_match"
        return ("undetermined", "لا تكفي القرائن للحكم، ويُحال المقطع إلى المختص", reason,
                NOT_FOUND_NOTE + DISCLAIMER)

    # 5. فيه نص ثابت ومعه كلام ما لقيناه في القاعدة: نمتنع عن الحكم على المقطع كله
    if any(not c["matched"] for c in claims):
        return ("undetermined", "في المقطع ما ثبت في المصادر، ومعه كلام لم يُعثر عليه، فيُحال إلى المختص",
                "partial_match", NOT_FOUND_NOTE + DISCLAIMER)

    # 6. كل ما ورد آيات من القرآن الكريم
    if classes and all(c == "quran" for c in classes):
        what = "النص الوارد آية من القرآن الكريم، مطابق لنص المصحف" if len(classes) == 1 else "النصوص الواردة آيات من القرآن الكريم، مطابقة لنص المصحف"
        return ("supported", what, None, DISCLAIMER)

    # 7. كل الأحاديث صحيحة، والصوت ما فيه قرينة توليد
    if auth.get("signal") == "inconclusive":
        return ("supported", "المضمون ثابت في المصادر المعتمدة، وفحص أصالة الصوت غير حاسم", None, DISCLAIMER)
    return ("supported", "المضمون ثابت في المصادر المعتمدة", None, DISCLAIMER)


def content_block(claims):
    matched = [c for c in claims if c["matched"]]
    if not matched:
        return {"status": "not_found", "label": "لم يُعثر على هذا النص في المصادر المتاحة", "claims": claims}
    bad = [c for c in matched if c["hadith"]["grade_class"] in BAD]
    label = "الحديث الوارد لا يثبت" if bad else "الحديث ثابت"
    if all(c["hadith"]["grade_class"] == "quran" for c in matched):
        label = "آية من القرآن الكريم"
    if len(matched) > 1:
        label = f"وُجد {len(matched)} أحاديث في المصادر"
    return {"status": "verified", "label": label, "claims": claims}
