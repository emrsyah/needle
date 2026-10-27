"""Build the Indonesian thesis (chapters-id/*.md) into a Word file styled like the TA proposal.

Usage (from the repo root):
    uv run --no-project --with pypandoc_binary --with python-docx python docs/thesis/build_id_docx.py [reference.docx]
The reference .docx supplies styles and page margins (default: the student's proposal).
Output: docs/thesis/output/tugas-akhir-id.docx. Export to PDF from Word (or export_pdf.ps1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "chapters-id"
OUT_DIR = HERE / "output"
BODY_STYLES = ("normal", "body text", "first paragraph", "compact", "bibliography")
DEFAULT_REFERENCE =Path.home() / "Downloads" / "Proposal TA Last.docx"
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

LOGO = HERE / "figures" / "telkom_logo.png"

# Cover and approval-sheet details, as on the student's proposal.
TITLE_ID = ("Pemanfaatan Reward Berbasis Evidence untuk Pelatihan Agen Retrieval-Augmented Generation "
            "(RAG) dengan Reinforcement Learning di Multi-Hop Question Answering (MHQA)")
TITLE_EN = ("Utilizing Evidence-Based Rewards for Training Retrieval-Augmented Generation (RAG) "
            "Agents with Reinforcement Learning in Multi-Hop Question Answering (MHQA)")
NIM = "103012300339"
NAME = "Muhammad Emirsyah Makarim"
COURSE = "IF-47-DSIS.02 (CAK4EAB2)"
SUPERVISOR = "Dr. Gamma Kosala, S.Si"
SUPERVISOR_NIP = "NIP: 26920005"
APPROVAL_DATE = "Bandung, 27 Oktober 2026"


def pandoc_to_docx(files: list[Path], resource_dir: Path, raw: Path, reference: Path) -> Document:
    pypandoc.convert_file(
        [str(f) for f in files],
        "docx",
        format="markdown-implicit_figures",
        outputfile=str(raw),
        extra_args=[
            f"--reference-doc={reference}",
            f"--resource-path={resource_dir}",
            "--toc",
            "--toc-depth=2",
            "--metadata=toc-title:DAFTAR ISI",
        ],
    )
    return Document(str(raw))


def apply_styles(doc: Document) -> None:
    """The proposal's body text: Times New Roman 12 pt, 1.5 line spacing, black headings."""
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
            # Sub-headings start at the text margin, like the body text.
            style.paragraph_format.left_indent = Pt(0)
            style.paragraph_format.first_line_indent = Pt(0)
            # Room between a heading and the text under it.
            style.paragraph_format.space_after = Pt(24 if style.name == "Heading 1" else 6)
        elif style.name.lower() in BODY_STYLES:
            style.font.size = Pt(12)
            style.paragraph_format.line_spacing = 1.5
            style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def centered(doc: Document, text: str = "", size: int = 12, bold: bool = True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Front-matter lines are single-spaced so the cover and approval sheet fit one page each.
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(6)
    if text:
        run = p.add_run(text)
        run.bold = bold
        run.font.size = Pt(size)
    return p


def page_break(doc: Document):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    return p


def move_to_front(doc: Document, elements: list) -> None:
    """Move freshly appended block elements to the start of the body, in order."""
    body = doc.element.body
    for index, element in enumerate(elements):
        body.remove(element)
        body.insert(index, element)


def style_tables(doc: Document) -> None:
    """Single borders, 10 pt single-spaced text, as in the proposal."""
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


def align_paragraphs(doc: Document) -> None:
    for p in doc.paragraphs:
        style = p.style.name if p.style is not None else ""
        if style == "Heading 1":
            p.paragraph_format.page_break_before = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        text = p.text.strip()
        if text.startswith(("Gambar ", "Tabel ")) and p.runs and p.runs[0].bold:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # The proposal template names its body style "normal" (lower case), so compare case-insensitively.
        if style.lower() in BODY_STYLES and text and not text.startswith(("Gambar ", "Tabel ")):
            if p.alignment != WD_ALIGN_PARAGRAPH.CENTER:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.line_spacing = 1.5
        # Images: centre the paragraph that holds a picture.
        if p._p.xpath(".//pic:pic"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def front_matter(doc: Document, document_type: str, sheet_title: str, sheet_text: str, supervisor_role: str) -> list:
    """Cover page and approval sheet in the layout of the student's proposal."""
    elements = []
    add = elements.append
    add(centered(doc, TITLE_ID, 14)._p)
    add(centered(doc)._p)
    add(centered(doc, document_type)._p)
    add(centered(doc)._p)
    add(centered(doc, COURSE)._p)
    add(centered(doc)._p)
    add(centered(doc, NIM)._p)
    add(centered(doc, NAME)._p)
    add(centered(doc)._p)
    logo = centered(doc)
    logo.add_run().add_picture(str(LOGO), width=Inches(1.37), height=Inches(1.83))
    add(logo._p)
    add(centered(doc)._p)
    for line in ("Program Studi Sarjana Informatika", "Fakultas Informatika", "Universitas Telkom", "Bandung", "2025/2026"):
        add(centered(doc, line)._p)

    heading = doc.add_paragraph(sheet_title, style="Heading 1")
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    heading.paragraph_format.page_break_before = True
    add(heading._p)
    add(centered(doc, TITLE_ID)._p)
    add(centered(doc, TITLE_EN)._p)
    add(centered(doc)._p)
    add(centered(doc, f"NIM : {NIM}")._p)
    add(centered(doc, NAME)._p)
    add(centered(doc)._p)
    add(centered(doc, sheet_text, bold=False)._p)
    add(centered(doc, "Program Studi Informatika", bold=False)._p)
    add(centered(doc, "Fakultas Informatika Universitas Telkom", bold=False)._p)
    add(centered(doc)._p)
    add(centered(doc, APPROVAL_DATE, bold=False)._p)
    add(centered(doc, "Menyetujui", bold=False)._p)
    add(centered(doc)._p)
    add(centered(doc, supervisor_role, bold=False)._p)
    for _ in range(4):
        add(centered(doc)._p)
    add(centered(doc, SUPERVISOR)._p)
    add(centered(doc, SUPERVISOR_NIP, bold=False)._p)
    return elements


def place_front_matter(doc: Document, front: list) -> None:
    """Order: cover, approval sheet, ABSTRAK, DAFTAR ISI, BAB 1 (pandoc puts the TOC first)."""
    body = doc.element.body
    toc = body.find(qn("w:sdt"))
    bab1 = next(p for p in doc.paragraphs if p.text.strip().upper().startswith("BAB 1"))
    if toc is not None:
        body.remove(toc)
        bab1._p.addprevious(toc)
        toc.addprevious(page_break(doc)._p)
    move_to_front(doc, front)


def main() -> None:
    reference = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REFERENCE
    OUT_DIR.mkdir(exist_ok=True)
    raw = OUT_DIR / "tugas-akhir-id.raw.docx"
    target = OUT_DIR / "tugas-akhir-id.docx"
    doc = pandoc_to_docx([SOURCE / f"{name}.md" for name in ORDER], SOURCE, raw, reference)
    apply_styles(doc)
    front = front_matter(
        doc,
        "Tugas Akhir",
        "Lembar Pengesahan",
        "Tugas akhir ini diajukan sebagai salah satu syarat kelulusan pada",
        "Pembimbing 1",
    )
    place_front_matter(doc, front)
    style_tables(doc)
    align_paragraphs(doc)
    doc.save(str(target))
    raw.unlink()
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
