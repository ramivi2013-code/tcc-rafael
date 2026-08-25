from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import io

from pypdf import PdfReader
from docx import Document


SUPPORTED_EXTS = {"txt", "pdf", "docx"}


@dataclass
class ReadResult:
    text: str
    filename: str
    extension: str


def read_uploaded_file(upload) -> ReadResult:
    """
    Lê arquivo enviado pelo Streamlit e retorna texto.

    Suporta:
    - .txt  (UTF-8 com fallback)
    - .pdf  (extração de texto)
    - .docx (Microsoft Word)
    """
    name = getattr(upload, "name", "arquivo")
    ext = name.split(".")[-1].lower() if "." in name else ""

    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Extensão não suportada: .{ext}")

    raw = upload.read()

    if ext == "txt":
        # tenta UTF-8, depois latin-1 como fallback
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")
        return ReadResult(text=text, filename=name, extension=ext)

    if ext == "pdf":
        bio = io.BytesIO(raw)
        reader = PdfReader(bio)
        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")
        text = "\n".join(pages_text)
        return ReadResult(text=text, filename=name, extension=ext)

    if ext == "docx":
        bio = io.BytesIO(raw)
        doc = Document(bio)
        text = "\n".join(p.text for p in doc.paragraphs)
        return ReadResult(text=text, filename=name, extension=ext)

    raise ValueError("Formato não reconhecido.")
