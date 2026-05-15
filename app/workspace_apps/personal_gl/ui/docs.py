"""Documentation rendering helpers."""

from __future__ import annotations

from datetime import datetime
import hashlib
import io
from pathlib import Path
import re

import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, Preformatted, Spacer
from reportlab.platypus.doctemplate import BaseDocTemplate, Frame, PageTemplate
from reportlab.platypus.tableofcontents import TableOfContents

from app.workspace_apps.personal_gl.constants import DOC_SECTIONS

PDF_DOC_EXPORT_VERSION = "v3"

DOC_MD_ORDER: list[str] | None = None
DOC_PREFERRED_TITLE_ORDER = [
    "Using the App",
    "Getting Started",
    "Monthly Adjustments",
    "Data Stored",
    "Financial Statements",
]


def docs_dir_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs"


def list_doc_markdown_files() -> list[Path]:
    docs_dir = docs_dir_path()
    if not docs_dir.exists() or not docs_dir.is_dir():
        return []

    md_files = [path for path in docs_dir.glob("*.md") if path.is_file()]
    if not md_files:
        return []

    by_name = {path.name: path for path in md_files}
    if isinstance(DOC_MD_ORDER, list) and DOC_MD_ORDER:
        return [by_name[name] for name in DOC_MD_ORDER if name in by_name]

    title_map: dict[str, Path] = {}
    for path in md_files:
        try:
            md_text = path.read_text(encoding="utf-8")
        except Exception:
            md_text = path.read_text(errors="ignore")
        title_map[infer_doc_title(md_text, path.stem).strip().casefold()] = path

    matched = [
        title_map[key]
        for key in [(title or "").strip().casefold() for title in DOC_PREFERRED_TITLE_ORDER]
        if key in title_map
    ]
    if len(matched) >= 2:
        return matched
    return sorted(md_files, key=lambda path: path.name.lower())


def infer_doc_title(md_text: str, fallback_name: str) -> str:
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback_name.replace("_", " ").replace("-", " ").strip().title()


def md_inline_to_rl(text: str) -> str:
    if text is None:
        return ""
    rendered = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    rendered = re.compile(r"`([^`]+)`").sub(
        lambda match: f"<font face='Courier'>{match.group(1)}</font>",
        rendered,
    )
    rendered = re.compile(r"\*\*([^*]+)\*\*").sub(
        lambda match: f"<b>{match.group(1)}</b>",
        rendered,
    )
    return rendered


def markdown_to_flowables(md_text: str, styles) -> list:
    story = []
    lines = md_text.splitlines()
    i = 0
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        if not paragraph_buffer:
            return
        raw = "\n".join([line.rstrip() for line in paragraph_buffer]).strip()
        paragraph_buffer = []
        if not raw:
            return
        converted_lines = [md_inline_to_rl(part) for part in raw.split("\n") if part.strip()]
        if converted_lines:
            story.append(Paragraph("<br/>".join(converted_lines), styles["Body"]))

    def read_list(start_idx: int):
        items = []
        j = start_idx
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped:
                break
            if stripped.startswith(("- ", "* ", "• ")):
                items.append(stripped[2:].strip())
                j += 1
                continue
            if len(stripped) >= 3 and stripped[0].isdigit() and stripped[1] == "." and stripped[2] == " ":
                items.append(stripped[3:].strip())
                j += 1
                continue
            break
        return items, j

    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("```"):
            flush_paragraph()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            story.append(Preformatted("\n".join(code_lines), styles["Code"]))
            story.append(Spacer(1, 0.08 * inch))
            i += 1
            continue

        if stripped in ("---", "***", "___"):
            flush_paragraph()
            story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
            story.append(Spacer(1, 0.08 * inch))
            i += 1
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            safe = stripped.lstrip("#").strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if safe:
                story.append(Paragraph(safe, styles["H1" if level == 1 else "H2" if level == 2 else "H3"]))
                story.append(Spacer(1, 0.08 * inch if level == 1 else 0.06 * inch))
            i += 1
            continue

        if stripped.startswith(("- ", "* ", "• ")) or (len(stripped) >= 3 and stripped[0].isdigit() and stripped[1] == "." and stripped[2] == " "):
            flush_paragraph()
            items, next_i = read_list(i)
            for item in items:
                story.append(Paragraph(f"• {md_inline_to_rl(item)}", styles["List"]))
            story.append(Spacer(1, 0.05 * inch))
            i = next_i
            continue

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        paragraph_buffer.append(lines[i])
        i += 1

    flush_paragraph()
    return story


class DocsPdfTemplate(BaseDocTemplate):
    def __init__(self, filename_or_buffer, **kwargs):
        super().__init__(filename_or_buffer, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="All", frames=[frame])])

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = getattr(flowable.style, "name", "")
            if style_name == "DocTitle":
                text = flowable.getPlainText()
                key = getattr(flowable, "_bookmarkName", None)
                if key:
                    self.canv.bookmarkPage(key)
                    self.notify("TOCEntry", (0, text, self.page, key))


@st.cache_data(show_spinner=False)
def build_docs_pdf_bytes(_docs_fingerprint: str) -> bytes:
    md_files = list_doc_markdown_files()
    if not md_files:
        return b""

    base_styles = getSampleStyleSheet()
    styles = {
        "CoverTitle": ParagraphStyle("CoverTitle", parent=base_styles["Title"], alignment=TA_CENTER, fontSize=24, spaceAfter=18),
        "CoverSub": ParagraphStyle("CoverSub", parent=base_styles["Normal"], alignment=TA_CENTER, fontSize=11, textColor=colors.grey),
        "Body": ParagraphStyle("Body", parent=base_styles["BodyText"], fontSize=10.5, leading=13, spaceAfter=2),
        "DocTitle": ParagraphStyle("DocTitle", parent=base_styles["Heading1"], fontSize=16, leading=18, spaceBefore=6, spaceAfter=8),
        "H1": ParagraphStyle("H1", parent=base_styles["Heading1"], fontSize=16, leading=18, spaceBefore=6, spaceAfter=6),
        "H2": ParagraphStyle("H2", parent=base_styles["Heading2"], fontSize=13, leading=15, spaceBefore=4, spaceAfter=4),
        "H3": ParagraphStyle("H3", parent=base_styles["Heading3"], fontSize=11.5, leading=13, spaceBefore=3, spaceAfter=3),
        "List": ParagraphStyle("List", parent=base_styles["BodyText"], leftIndent=14, fontSize=10.5, leading=13, spaceAfter=1),
        "Code": ParagraphStyle("Code", parent=base_styles["Code"], fontName="Courier", fontSize=9.5, leading=11, backColor=colors.whitesmoke, borderPadding=6),
    }

    buffer = io.BytesIO()
    doc = DocsPdfTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Documentation",
        author="Personal General Ledger",
    )

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            name="TOCLevel0",
            parent=base_styles["Normal"],
            fontSize=11,
            leftIndent=18,
            firstLineIndent=-18,
            spaceBefore=4,
            leading=14,
        )
    ]
    doc._toc = toc

    story = [
        Spacer(1, 1.5 * inch),
        Paragraph("Personal General Ledger", styles["CoverTitle"]),
        Paragraph("Documentation Packet", styles["CoverTitle"]),
        Spacer(1, 0.25 * inch),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["CoverSub"]),
        PageBreak(),
        Paragraph("Table of Contents", styles["H1"]),
        Spacer(1, 0.15 * inch),
        toc,
        PageBreak(),
    ]

    for path in md_files:
        try:
            md_text = path.read_text(encoding="utf-8")
        except Exception:
            md_text = path.read_text(errors="ignore")

        title = infer_doc_title(md_text, path.stem)
        anchor = "doc_" + hashlib.md5((path.name + title).encode("utf-8")).hexdigest()[:10]
        heading = Paragraph(f'<a name="{anchor}"/>{title}', styles["DocTitle"])
        heading._bookmarkName = anchor
        story.append(heading)
        story.append(Spacer(1, 0.10 * inch))
        story.extend(markdown_to_flowables(md_text, styles))
        story.append(PageBreak())

    doc.multiBuild(story)
    buffer.seek(0)
    return buffer.getvalue()


def docs_fingerprint() -> str:
    hasher = hashlib.sha256()
    hasher.update(PDF_DOC_EXPORT_VERSION.encode("utf-8"))
    for path in list_doc_markdown_files():
        hasher.update(path.name.encode("utf-8"))
        try:
            hasher.update(path.read_bytes())
        except Exception:
            pass
    return hasher.hexdigest()


def render_doc_sections() -> None:
    docs_dir = docs_dir_path()
    if not docs_dir.exists():
        st.info("Documentation files are not available in this workspace copy yet.")
        return

    for title, filename in DOC_SECTIONS:
        path = docs_dir / filename
        with st.expander(title):
            try:
                st.markdown(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                st.error(f"Documentation file not found: {path}")
            except Exception as exc:
                st.error(f"Unable to load documentation from {filename}: {exc}")
