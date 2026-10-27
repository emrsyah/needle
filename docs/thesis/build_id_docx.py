"""Build the Indonesian thesis (chapters-id/*.md) into a Word file styled like the TA proposal.

Usage (from the repo root):
    uv run --no-project --with pypandoc_binary --with python-docx python docs/thesis/build_id_docx.py [reference.docx]
The reference .docx supplies styles and page margins (default: the student's proposal).
Output: docs/thesis/output/tugas-akhir-id.docx. Export to PDF from Word (or export_pdf.ps1).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "chapters-id"
OUT_DIR = HERE / "output"
DEFAULT_REFERENCE = Path.home() / "Downloads" / "Proposal TA Last.docx"
ORDER = [
    "abstrak",
    "bab1-pendahuluan",
    "bab2-kajian-pustaka",
    "bab3-perancangan-sistem",
    "bab4-hasil-dan-analisis",
    "bab5-kesimpulan-dan-saran",
    "daftar-pustaka",
    "lampiran-a-prompt",
]

# Cover text taken from the proposal's cover page.
COVER = [
    ("Rewarding the Evidence, Not Just the Answer: Verifiable Evidence-Grounded "
     "Rewards for Training Search Agents with GRPO", 14, True),
    ("", 12, False),
    ("Tugas Akhir", 12, True),
    ("", 12, False),
    ("103012300339", 12, True),
    ("Muhammad Emirsyah Makarim", 12, True),
    ("", 12, False),
    ("Program Studi Sarjana Informatika", 12, True),
    ("Fakultas Informatika", 12, True),
    ("Universitas Telkom", 12, True),
    ("Bandung", 12, True),
    ("2025/2026", 12, True),
]


def main() -> None:
    reference = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REFERENCE
    OUT_DIR.mkdir(exist_ok=True)
    raw = OUT_DIR / "tugas-akhir-id.raw.docx"
    target = OUT_DIR / "tugas-akhir-id.docx"
    files = [str(SOURCE / f"{name}.md") for name in ORDER]
    pypandoc.convert_file(
        files,
        "docx",
        format="markdown-implicit_figures",
        outputfile=str(raw),
        extra_args=[
            f"--reference-doc={reference}",
            f"--resource-path={SOURCE}",
            "--toc",
            "--toc-depth=2",
            "--metadata=toc-title:DAFTAR ISI",
        ],
    )
    doc = Document(str(raw))

    # The proposal's body text: Times New Roman 12 pt, 1.5 line spacing, black headings.
    for style in doc.styles:
        if style.type != 1:  # paragraph styles only
            continue
        style.font.name = "Times New Roman"
        rpr = style.element.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.append(fonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            fonts.set(qn(attr), "Times New Roman")
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            fonts.attrib.pop(qn(attr), None)
        if style.name.startswith("Heading"):
            style.font.color.rgb = RGBColor(0, 0, 0)
            style.font.bold = True
            style.font.size = Pt({"Heading 1": 14}.get(style.name, 12))
        elif style.name in ("Normal", "Body Text", "First Paragraph", "Compact"):
            style.font.size = Pt(12)
            style.paragraph_format.line_spacing = 1.5

    # Cover page before everything else (pandoc puts the table of contents first).
    body = doc.element.body
    cover = []
    for text, size, bold in COVER:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.bold = bold
        run.font.size = Pt(size)
        cover.append(p._p)
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    cover.append(p._p)
    for index, element in enumerate(cover):
        body.remove(element)
        body.insert(index, element)

    # Tables: single borders, 10 pt single-spaced text, as in the proposal.
    for table in doc.tables:
        tbl_pr = table._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            line = OxmlElement(f"w:{edge}")
            line.set(qn("w:val"), "single")
            line.set(qn("w:sz"), "4")
            line.set(qn("w:color"), "000000")
            borders.append(line)
        tbl_pr.append(borders)
        for row in table.rows:
            for cell in row.cells:
                for cp in cell.paragraphs:
                    cp.paragraph_format.line_spacing = 1.0
                    cp.paragraph_format.space_after = Pt(0)
                    for run in cp.runs:
                        run.font.size = Pt(10)

    for p in doc.paragraphs:
        style = p.style.name if p.style is not None else ""
        if style == "Heading 1":
            p.paragraph_format.page_break_before = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        text = p.text.strip()
        if text.startswith(("Gambar ", "Tabel ")) and p.runs and p.runs[0].bold:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if style in ("Body Text", "First Paragraph", "Compact", "Normal") and not text.startswith(("Gambar ", "Tabel ")):
            if p.alignment is None:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        # Images: centre the paragraph that holds a picture.
        if p._p.xpath(".//pic:pic"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.save(str(target))
    raw.unlink()
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
