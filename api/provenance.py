"""
فحص بيانات الملف وتوقيع اعتماد المحتوى (C2PA).

هذا أول جزء مفعّل من قرينة الأصالة التقنية، وهو كما وصفناه في ملف الفكرة «قرينة داعمة»:
- إذا كان في الملف توقيع C2PA سليم يصرّح بأن المحتوى مولَّد بالذكاء الاصطناعي، فهذا تصريح من أداة التوليد نفسها.
- إذا ذكرت بيانات الملف أداة توليد صوت أو فيديو معروفة، نرفع تنبيهاً ونحيل للمختص، لأن هذه البيانات قابلة للتعديل.
- إذا كان فيه توقيع سليم بلا تصريح بالتوليد، يرتفع الاطمئنان لمصدر الملف.
- وغياب التوقيع لا يعني التزييف، لأن أغلب المقاطع المتداولة بلا توقيع.

ما نحفظ الملف ولا بياناته بعد الفحص.
"""

import json
import re
import subprocess

try:
    import c2pa
except Exception:  # المكتبة اختيارية، وبدونها نكتفي ببيانات الملف
    c2pa = None

# أنواع المصدر في معيار IPTC اللي تعني أن المحتوى مولَّد أو فيه جزء مولَّد
AI_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia",
                   "algorithmicmedia", "compositesynthetic")

# أدوات توليد الصوت والفيديو المعروفة كما تظهر في بيانات الملفات
GENERATORS = {
    "elevenlabs": "ElevenLabs", "eleven labs": "ElevenLabs", "play.ht": "PlayHT", "playht": "PlayHT",
    "resemble": "Resemble AI", "murf": "Murf", "speechify": "Speechify", "wellsaid": "WellSaid",
    "lovo": "LOVO", "uberduck": "Uberduck", "fakeyou": "FakeYou", "voicemod": "Voicemod",
    "voice.ai": "Voice.ai", "overdub": "Descript Overdub", "coqui": "Coqui TTS", "tortoise-tts": "Tortoise TTS",
    "so-vits": "So-VITS", "rvc": "RVC", "amazon polly": "Amazon Polly", "text-to-speech": "Text-to-Speech",
    "tts": "Text-to-Speech", "suno": "Suno", "udio": "Udio", "heygen": "HeyGen", "synthesia": "Synthesia",
    "d-id": "D-ID", "hedra": "Hedra", "deepfacelab": "DeepFaceLab", "faceswap": "FaceSwap",
    "sora": "Sora", "runway": "Runway", "pika": "Pika", "kling": "Kling", "veo": "Veo",
}
_GEN_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(k) for k in sorted(GENERATORS, key=len, reverse=True)) + r")(?![a-z0-9])")

# الحقول اللي تكتب فيها الأدوات اسمها
# (ما نبحث في العنوان واسم الفنان، لأنها تُكتب يدوياً وقد تشابه أسماء الأدوات)
TAG_KEYS = ("encoder", "encoded_by", "software", "comment", "description",
            "handler_name", "creation_tool", "tool", "tsse", "tenc", "tsoft")


def _probe(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                             capture_output=True, check=True, timeout=30).stdout
        return json.loads(out or b"{}")
    except Exception:
        return {}


def _tags(info):
    found = {}
    for block in [info.get("format", {})] + list(info.get("streams", [])):
        for k, v in (block.get("tags") or {}).items():
            if k.lower() in TAG_KEYS and isinstance(v, str) and v.strip():
                found.setdefault(k.lower(), v.strip()[:120])
    return found


def _generator(tags):
    for v in tags.values():
        m = _GEN_RE.search(v.lower())
        if m:
            return GENERATORS[m.group(1)]
    return None


def _read_c2pa(path):
    """يرجع None إذا ما فيه توقيع، وإلا ملخص التوقيع."""
    if c2pa is None:
        return None
    try:
        with c2pa.Reader(path) as r:
            store = json.loads(r.json())
            state = r.get_validation_state()
    except Exception:
        return None
    manifests = store.get("manifests") or {}
    active = manifests.get(store.get("active_manifest")) or {}
    ai = False
    for m in manifests.values():
        for a in m.get("assertions") or []:
            if str(a.get("label", "")).startswith("c2pa.actions"):
                for act in (a.get("data") or {}).get("actions") or []:
                    if any(t in str(act.get("digitalSourceType", "")).lower() for t in AI_SOURCE_TYPES):
                        ai = True
    gen = active.get("claim_generator_info") or []
    tool = (gen[0].get("name") if gen and isinstance(gen[0], dict) else None) or active.get("claim_generator")
    issuer = (active.get("signature_info") or {}).get("issuer")
    return {"state": str(state or ""), "ai": ai, "tool": tool, "issuer": issuer}


def inspect(path):
    """يفحص الملف ويرجع قرينة وصفية للتقرير، بدون أي رقم خام."""
    info = _probe(path)
    tags = _tags(info)
    encoder = tags.get("encoder") or tags.get("software") or tags.get("handler_name")
    signed = _read_c2pa(path)
    base = {"encoder": encoder, "c2pa": signed is not None}

    if signed and signed["ai"] and signed["state"].lower() != "invalid":
        tool = f"، والأداة المذكورة فيه: {signed['tool']}" if signed.get("tool") else ""
        return {**base, "status": "ai_declared", "source": "c2pa",
                "label": f"توقيع اعتماد المحتوى (C2PA) في الملف يصرّح بأن المقطع مولَّد بالذكاء الاصطناعي{tool}"}

    gen = _generator(tags)
    if gen:
        return {**base, "status": "ai_declared", "source": "metadata", "tool": gen,
                "label": f"بيانات الملف تذكر أداة توليد معروفة ({gen})، وهذه البيانات قابلة للتعديل فتُعد تنبيهاً لا حكماً"}

    if signed and signed["state"].lower() == "invalid":
        return {**base, "status": "signed_invalid",
                "label": "في الملف توقيع اعتماد (C2PA) لا يطابق محتواه، فقد يكون المقطع عُدّل بعد توقيعه"}

    if signed:
        who = f" من «{signed['issuer']}»" if signed.get("issuer") else ""
        trust = "، وجهة التوقيع موثوقة" if signed["state"].lower() == "trusted" else "، ولم تُتحقق موثوقية جهة التوقيع"
        return {**base, "status": "signed",
                "label": f"يحمل الملف توقيع اعتماد المحتوى (C2PA) سليماً{who}{trust}، ولا يصرّح بالتوليد"}

    if not info:
        return {**base, "status": "failed", "label": "تعذّر قراءة بيانات الملف"}

    return {**base, "status": "none",
            "label": "لا يحمل الملف توقيع اعتماد ولا أثراً لأدوات توليد معروفة، وغياب التوقيع لا يعني التزييف لأن أغلب المقاطع المتداولة بلا توقيع"}
