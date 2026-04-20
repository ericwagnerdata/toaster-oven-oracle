"""Diagnostic: scan each page and guess English vs French."""
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "BOV900_USCM_IB_Rv1.pdf"

FR_HINTS = (" le ", " la ", " les ", " est ", " pour ", " avec ", " une ", "é", "è", "ç", "à")
EN_HINTS = (" the ", " and ", " for ", " with ", " press ", " rack ", " oven ")


def guess_lang(text: str) -> str:
    t = " " + text.lower() + " "
    fr = sum(t.count(h) for h in FR_HINTS)
    en = sum(t.count(h) for h in EN_HINTS)
    if en == 0 and fr == 0:
        return "??"
    return "EN" if en >= fr else "FR"


reader = PdfReader(str(PDF))
print(f"{len(reader.pages)} pages\n")
for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    lang = guess_lang(text)
    snippet = text.replace("\n", " ")[:70]
    print(f"p{i:3d}  {lang}  {len(text):5d} chars  | {snippet}")
