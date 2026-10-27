"""Build the Indonesian *proposal* (proposal-id/*.md) into a Word file in the TA proposal format.

Usage (from the repo root):
    uv run --no-project --with pypandoc_binary --with python-docx python docs/thesis/build_proposal_docx.py [reference.docx]
Output: docs/thesis/output/proposal-ta-id.docx (export to PDF with export_pdf.ps1 -Name proposal-ta-id).

Citations in proposal-id/ use the thesis numbering of chapters-id/daftar-pustaka.md. This script
renumbers them by first appearance and writes a DAFTAR PUSTAKA with only the cited entries.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches

from build_id_docx import (
    DEFAULT_REFERENCE,
    HERE,
    OUT_DIR,
    align_paragraphs,
    apply_styles,
    centered,
    move_to_front,
    page_break,
    pandoc_to_docx,
    style_tables,
)

SOURCE = HERE / "proposal-id"
REFERENCES = HERE / "chapters-id" / "daftar-pustaka.md"
LOGO = HERE / "figures" / "telkom_logo.png"
ORDER = ["abstrak", "bab1-pendahuluan", "jadwal-kegiatan", "bab2-kajian-pustaka", "bab3-perancangan-sistem"]

TITLE_ID = ("Memberi Reward pada Bukti, Bukan Hanya pada Jawaban: Verifiable Evidence-Grounded "
            "Reward untuk Melatih Agen Pencarian dengan GRPO")
TITLE_EN = ("Rewarding the Evidence, Not Just the Answer: Verifiable Evidence-Grounded Rewards "
            "for Training Search Agents with GRPO")
NIM = "103012300339"
NAME = "Muhammad Emirsyah Makarim"
COURSE = "IF-47-DSIS.02 (CAK4EAB2)"
SUPERVISOR = "Dr. Gamma Kosala, S.Si"
SUPERVISOR_NIP = "NIP: 26920005"
APPROVAL_DATE = "Bandung, 27 Oktober 2026"

CITATION = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)((?:,\s*Lampiran[^\]]*)?)\]")
MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$|`[^`\n]*`", re.S)


def load_references() -> dict[int, str]:
    entries = {}
    for line in REFERENCES.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\\?\[(\d+)\\?\]\s*(.*)", line.strip())
        if match:
            entries[int(match.group(1))] = match.group(2)
    return entries


def renumber(texts: list[str], known: set[int]) -> tuple[list[str], list[int]]:
    """Renumber IEEE citations by first appearance across texts; return new texts and old ids in order."""
    order: list[int] = []

    def replace(match: re.Match) -> str:
        olds = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        if not all(n in known for n in olds):
            return match.group(0)
        news = []
        for old in olds:
            if old not in order:
                order.append(old)
            news.append(str(order.index(old) + 1))
        return f"[{', '.join(news)}{match.group(2)}]"

    out = []
    for text in texts:
        # Leave maths and code untouched.
        pieces, last = [], 0
        for m in MATH.finditer(text):
            pieces.append(CITATION.sub(replace, text[last:m.start()]))
            pieces.append(m.group(0))
            last = m.end()
        pieces.append(CITATION.sub(replace, text[last:]))
        out.append("".join(pieces))
    return out, order


def shade_schedule(doc) -> None:
    """Cells marked X in the Jadwal table become grey blocks, as in the old proposal."""
    for table in doc.tables:
        if not table.rows or "Kegiatan" not in table.rows[0].cells[0].text:
            continue
        table.autofit = False
        for row in table.rows:
            for index, cell in enumerate(row.cells):
                cell.width = Inches(2.4) if index == 0 else Inches(0.55)
        for row in table.rows[1:]:
            for cell in row.cells[1:]:
                if cell.text.strip() == "X":
                    for p in cell.paragraphs:
                        for run in p.runs:
                            run.text = ""
                    shading = OxmlElement("w:shd")
                    shading.set(qn("w:val"), "clear")
                    shading.set(qn("w:color"), "auto")
                    shading.set(qn("w:fill"), "808080")
                    cell._tc.get_or_add_tcPr().append(shading)


def front_matter(doc) -> list:
    elements = []
    add = elements.append
    add(centered(doc, TITLE_ID, 14)._p)
    add(centered(doc)._p)
    add(centered(doc, "Proposal Tugas Akhir")._p)
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

    heading = doc.add_paragraph("Lembar Persetujuan", style="Heading 1")
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    heading.paragraph_format.page_break_before = True
    add(heading._p)
    add(centered(doc, TITLE_ID)._p)
    add(centered(doc, TITLE_EN)._p)
    add(centered(doc)._p)
    add(centered(doc, f"NIM : {NIM}")._p)
    add(centered(doc, NAME)._p)
    add(centered(doc)._p)
    add(centered(doc, "Proposal ini diajukan sebagai usulan pembuatan tugas akhir pada", bold=False)._p)
    add(centered(doc, "Program Studi Informatika", bold=False)._p)
    add(centered(doc, "Fakultas Informatika Universitas Telkom", bold=False)._p)
    add(centered(doc)._p)
    add(centered(doc, APPROVAL_DATE, bold=False)._p)
    add(centered(doc, "Menyetujui", bold=False)._p)
    add(centered(doc)._p)
    add(centered(doc, "Calon Pembimbing 1", bold=False)._p)
    for _ in range(4):
        add(centered(doc)._p)
    add(centered(doc, SUPERVISOR)._p)
    add(centered(doc, SUPERVISOR_NIP, bold=False)._p)
    return elements


def main() -> None:
    reference = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REFERENCE
    OUT_DIR.mkdir(exist_ok=True)
    entries = load_references()
    sources = [(SOURCE / f"{name}.md").read_text(encoding="utf-8") for name in ORDER]
    texts, order = renumber(sources, set(entries))
    bibliography = "# DAFTAR PUSTAKA {.unnumbered}\n\n" + "\n\n".join(
        f"\\[{new}\\] {entries[old]}" for new, old in enumerate(order, start=1)
    )
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        # Numbered names: pypandoc sorts the input list, so the prefix fixes the order.
        for index, (name, text) in enumerate(zip(ORDER + ["daftar-pustaka"], texts + [bibliography])):
            path = Path(tmp) / f"{index:02d}-{name}.md"
            path.write_text(text, encoding="utf-8")
            files.append(path)
        raw = OUT_DIR / "proposal-ta-id.raw.docx"
        doc = pandoc_to_docx(files, SOURCE, raw, reference)
    apply_styles(doc)

    # Old proposal order: cover, Lembar Persetujuan, ABSTRAK, DAFTAR ISI, BAB 1 ...
    body = doc.element.body
    toc = body.find(qn("w:sdt"))
    bab1 = next(p for p in doc.paragraphs if p.text.strip().upper().startswith("BAB 1"))
    if toc is not None:
        body.remove(toc)
        bab1._p.addprevious(toc)
        toc.addprevious(page_break(doc)._p)
    move_to_front(doc, front_matter(doc))

    style_tables(doc)
    shade_schedule(doc)
    align_paragraphs(doc)
    target = OUT_DIR / "proposal-ta-id.docx"
    doc.save(str(target))
    raw.unlink()
    print(f"wrote {target} ({len(order)} references)")


if __name__ == "__main__":
    main()
