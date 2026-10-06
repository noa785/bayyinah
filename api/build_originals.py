"""
يبني الأرشيف التجريبي للمطابقة مع الأصل من جدول «الأرشيف التجريبي للمطابقة مع الأصل.xlsx».

الأعمدة: م، العالم، العنوان، الرابط، النص الكامل، رابط الصوت، ملاحظة.
ما نقبل إلا روابط الموقعين الرسميين، والنص يُنقل كما هو دون تعديل.

الاستخدام:
    python build_originals.py "الأرشيف التجريبي للمطابقة مع الأصل.xlsx"
"""

import json
import sys
from pathlib import Path

from openpyxl import load_workbook

OFFICIAL = ("binbaz.org.sa", "binothaimeen.net")
OUT = Path(__file__).parent / "data" / "originals.json"


def main(path):
    ws = load_workbook(path, read_only=True).worksheets[0]
    docs, skipped = [], []
    for row in ws.iter_rows(min_row=2, values_only=True):
        n, scholar, title, url, text, audio = (list(row) + [None] * 7)[:6]
        if not text or not str(text).strip():
            continue
        url = (url or "").strip()
        if not any(d in url for d in OFFICIAL):
            skipped.append(f"{n}: الرابط ليس من الموقعين الرسميين")
            continue
        docs.append({"id": f"O{len(docs) + 1:03d}", "scholar": (scholar or "").strip(), "title": (title or "").strip(),
                     "url": url, "audio": (audio or "").strip() or None, "text": " ".join(str(text).split())})
    OUT.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"حُفظ {len(docs)} نصاً في {OUT}")
    for s in skipped:
        print("تُرك:", s)


if __name__ == "__main__":
    main(sys.argv[1])
