"""Step 1 of the pipeline: read .txt, .docx or .pdf and return plain text.

Everything after this step works on plain text, so the algorithms
(LCS / Edit Distance) don't care which file type the text came from.
"""
from io import BytesIO
from pathlib import Path

SUPPORTED = (".txt", ".docx", ".pdf")


def read_bytes(data: bytes, filename: str) -> str:
    """Extract text from raw file bytes (used by the Streamlit uploader)."""
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        return data.decode("utf-8", errors="ignore")

    if ext == ".docx":
        from docx import Document                      # pip install python-docx
        doc = Document(BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:                       # text inside tables too
            for row in table.rows:
                parts.extend(cell.text for cell in row.cells)
        return "\n".join(parts)

    if ext == ".pdf":
        from pypdf import PdfReader                    # pip install pypdf
        reader = PdfReader(BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if not text.strip():
            raise ValueError("No text found in this PDF. It may be a scanned image "
                             "(this tool cannot read scanned PDFs without OCR).")
        return text

    raise ValueError(f"Unsupported file type '{ext}'. Use: {', '.join(SUPPORTED)}")


def read_file(path: str) -> str:
    """Extract text from a file on disk (used by the command line)."""
    return read_bytes(Path(path).read_bytes(), path)
