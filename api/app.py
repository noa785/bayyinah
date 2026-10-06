"""
بيّنة: الواجهة البرمجية.

التشغيل على الجهاز:
    uvicorn app:app --reload --port 7860
"""

import os
import shutil
import tempfile
import time
import uuid

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from audio import AudioError, authenticity, load_audio, transcribe
import dorar
from engine import HadithEngine
from verdict import TITLES, content_block, decide

MAX_MB = 50

app = FastAPI(title="Bayyinah API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = HadithEngine()


def download_audio(link):
    """يحمّل صوت المقطع من الرابط. يرجع (المسار، المجلد المؤقت) أو يرمي خطأ إذا منعت المنصة التحميل."""
    import yt_dlp
    folder = tempfile.mkdtemp(prefix="bayyinah_")
    opts = {
        "format": "bestaudio/best", "outtmpl": os.path.join(folder, "clip.%(ext)s"),
        "noplaylist": True, "quiet": True, "no_warnings": True, "socket_timeout": 20, "retries": 1, "extractor_retries": 1,
        "max_filesize": MAX_MB * 1024 * 1024,
        "match_filter": yt_dlp.utils.match_filter_func("!duration | duration <= 900"),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([link])
        files = [f for f in os.listdir(folder) if not f.endswith(".part")]
        if not files:
            raise RuntimeError("no file")
        return os.path.join(folder, files[0]), folder
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


MEDIA_EVIDENCE = {
    "speaker_match": {"status": "no_reference", "label": "لا تتوفر عينة مرجعية لصوت المتحدث المنسوب إليه"},
    "original_match": {"status": "not_found", "label": "أرشيف الدروس الأصلية قيد البناء، فلم تُجرَ المطابقة",
                       "source": None, "before": None, "after": None},
}


def error(status, code, message):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.get("/health")
def health():
    return {"status": "ok", "hadiths": len(engine.db)}


NOT_APPLICABLE = {
    "speaker_match": {"status": "not_applicable", "label": "المدخل نص"},
    "original_match": {"status": "not_applicable", "label": "المدخل نص", "source": None, "before": None, "after": None},
    "authenticity": {"status": "not_applicable", "signal": None, "label": "المدخل نص، فلم يُجرَ الفحص التقني", "waveform": None},
}


@app.post("/analyze")
async def analyze(
    input_type: str = Form(...),
    kind: str = Form("auto"),
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    text: str | None = Form(None),
    context: str | None = Form(None),
):
    t0 = time.time()
    input_type = (input_type or "").strip()
    if input_type not in ("audio", "video", "url", "text"):
        return error(400, "unsupported_format", "نوع المدخل غير مدعوم.")

    duration = None
    evidence = {k: dict(v) for k, v in NOT_APPLICABLE.items()}

    # ---------- النص ----------
    if input_type == "text":
        transcript = (text or "").strip()
        if len(transcript) < 8:
            return error(422, "no_speech", "اكتب نصاً لا يقل عن ٨ أحرف.")

    # ---------- الرابط: نحاول نحمّل المقطع، وإذا منعت المنصة نعتمد على النص المكتوب ----------
    elif input_type == "url":
        link = (url or "").strip()
        wav, folder = None, None
        if link.startswith(("http://", "https://")):
            try:
                path, folder = await run_in_threadpool(download_audio, link)
                wav, duration = load_audio(path)
            except Exception:
                wav = None
            finally:
                if folder:
                    shutil.rmtree(folder, ignore_errors=True)
        transcript = ""
        if wav is not None:
            evidence["authenticity"] = authenticity(wav, duration)
            evidence.update({k: dict(v) for k, v in MEDIA_EVIDENCE.items()})
            try:
                transcript = await run_in_threadpool(transcribe, wav)
            except AudioError:
                transcript = ""
        if not transcript:
            transcript = (context or "").strip()
            if len(transcript) < 8:
                if "youtube.com" in link or "youtu.be" in link:
                    return error(422, "download_failed",
                                 "يوتيوب يمنع تحميل المقاطع من الخوادم. نزّل المقطع وارفعه من تبويب «فيديو» أو «صوت»، أو اكتب ما قيل فيه في الخانة.")
                return error(422, "download_failed",
                             "تعذّر تحميل المقطع من هذا الرابط، فبعض المنصات تمنع التحميل. ارفع المقطع من تبويب «فيديو» أو «صوت»، أو اكتب ما قيل فيه في الخانة.")
            if wav is None:
                evidence["authenticity"]["label"] = "تعذّر تحميل المقطع من الرابط، فتُحقّق من النص المكتوب"

    # ---------- الصوت والفيديو ----------
    else:
        if file is None:
            return error(422, "unsupported_format", "اختر ملفاً أولاً.")
        data = await file.read()
        if len(data) > MAX_MB * 1024 * 1024:
            return error(413, "file_too_large", f"حجم الملف يتجاوز {MAX_MB} ميقابايت.")
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"
        # على ويندوز ما ينفع نفتح الملف المؤقت وهو مفتوح، فنقفله ونحذفه بأنفسنا
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(data)
            tmp.close()
            wav, duration = load_audio(tmp.name)
        except AudioError as e:
            return error(422, e.code, e.message)
        finally:
            tmp.close()
            os.remove(tmp.name)
        # الملف ينحذف هنا، وما نحتفظ بأي مقطع بعد التحليل

        evidence["authenticity"] = authenticity(wav, duration)
        evidence["speaker_match"] = {"status": "no_reference",
                                     "label": "لا تتوفر عينة مرجعية لصوت المتحدث المنسوب إليه"}
        evidence["original_match"] = {"status": "not_found",
                                      "label": "أرشيف الدروس الأصلية قيد البناء، فلم تُجرَ المطابقة",
                                      "source": None, "before": None, "after": None}
        try:
            transcript = transcribe(wav)
        except AudioError as e:
            if context and len(context.strip()) >= 8:
                transcript = context.strip()
            else:
                return error(422, e.code, e.message)

    # ---------- التحقق من المضمون ----------
    kind = kind if kind in ("auto", "hadith", "quran") else "auto"
    claims = engine.find_claims(transcript, kind=kind)
    # إذا ما تعرّف النظام على شي من التفريغ، وكتب المستخدم وش انقال في المقطع، نتحقق من كلامه
    if input_type in ("audio", "video", "url") and context and len(context.strip()) >= 8 \
            and not any(c["matched"] for c in claims) and context.strip() != transcript:
        ctx_claims = engine.find_claims(context.strip(), kind=kind)
        if any(c["matched"] for c in ctx_claims):
            claims = ctx_claims
    # ما لقيناه في قاعدتنا ولا في القرآن: نبحث في الموسوعة الحديثية بالدرر السنية
    if kind != "quran" and not any(c["matched"] for c in claims):
        for q in dict.fromkeys(x for x in (transcript, (context or "").strip()) if x and len(x) >= 8):
            found = await run_in_threadpool(dorar.lookup, q)
            if found:
                claims = [found]
                break
    evidence["content"] = content_block(claims)
    verdict, summary, abstain, disclaimer = decide(evidence["content"], evidence["authenticity"], duration)

    return {
        "request_id": uuid.uuid4().hex[:12],
        "input_type": input_type,
        "processing_ms": int((time.time() - t0) * 1000),
        "verdict": verdict,
        "verdict_title": TITLES[verdict],
        "verdict_summary": summary,
        "abstain_reason": abstain,
        "transcript": {"available": True, "text": transcript, "duration_sec": round(duration, 1) if duration else None},
        "evidence": evidence,
        "disclaimer": disclaimer,
    }
