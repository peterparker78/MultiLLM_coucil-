"""Briefing-document ingestion: extract plain text from uploaded files so a
council or board can be grounded in your context. Supports .txt/.md, .docx,
.pdf. Several documents are joined by `combine_briefs` under one shared
character budget.

PRIVACY: a briefing doc is injected into every seat's prompt. OpenRouter seats
send it to third-party providers; only local (non-:cloud) Ollama seats keep it
on-device. Do not upload confidential/PHI material to a cloud panel.
"""

from __future__ import annotations

from pathlib import Path


def brief_to_text(path: str, max_chars: int = 12000) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".txt", ".md", ""):
        text = p.read_text(errors="ignore")
    elif suffix == ".docx":
        text = _docx_text(p)
    elif suffix == ".pdf":
        text = _pdf_text(p)
    else:
        raise ValueError(f"Unsupported briefing format '{suffix}'. Use .txt, .md, .docx, or .pdf.")
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[briefing truncated]"
    return text


def combine_briefs(docs: list[tuple[str, str]], max_chars: int = 40000) -> str:
    """Join several (name, text) briefing documents into one context block with
    a heading per document. The character budget is shared: short documents
    keep their full text and the leftover goes to the longer ones, so one big
    file cannot crowd the others out. Empty documents are skipped."""
    docs = [(name, text.strip()) for name, text in docs if text and text.strip()]
    if not docs:
        return ""
    budgets: dict[int, int] = {}
    remaining = max(0, max_chars)
    for idx, (_, text) in sorted(enumerate(docs), key=lambda it: len(it[1][1])):
        left = len(docs) - len(budgets)
        budgets[idx] = min(len(text), remaining // left)
        remaining -= budgets[idx]
    parts = []
    for idx, (name, text) in enumerate(docs):
        if len(text) > budgets[idx]:
            text = text[: budgets[idx]] + "\n\n[document truncated]"
        parts.append(f"### Document {idx + 1}: {name}\n\n{text}")
    return "\n\n".join(parts)


def text_from_bytes(filename: str, data: bytes, max_chars: int = 12000) -> str:
    """Ingest an uploaded file given its bytes (for the web upload endpoint)."""
    import os
    import tempfile

    suffix = Path(filename).suffix.lower() or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return brief_to_text(tmp_path, max_chars=max_chars)
    finally:
        os.unlink(tmp_path)  # briefings can be confidential; leave no copy in /tmp


def _docx_text(p: Path) -> str:
    from docx import Document

    doc = Document(str(p))
    return "\n".join(para.text for para in doc.paragraphs)


def _pdf_text(p: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(p))
    return "\n".join((page.extract_text() or "") for page in reader.pages)
