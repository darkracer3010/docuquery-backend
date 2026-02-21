import fitz  # pymupdf
from typing import List
from app.documents.parsers.base import BaseParser
from app.documents.schemas import ParsedElement


class PDFParser(BaseParser):
    """Parse PDF documents using PyMuPDF."""

    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedElement]:
        elements = []
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block["type"] != 0:  # Skip image blocks
                    continue

                block_text_parts = []
                is_heading = False

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        block_text_parts.append(text)

                        # Detect headings by font size (larger than 14pt)
                        font_size = span.get("size", 12)
                        if font_size >= 14 or span.get("flags", 0) & 2**4:  # bold flag
                            is_heading = True

                block_text = " ".join(block_text_parts)
                cleaned = self._clean_text(block_text)

                if not cleaned or len(cleaned) < 3:
                    continue

                elements.append(
                    ParsedElement(
                        text=cleaned,
                        element_type="heading" if is_heading else "paragraph",
                        page_number=page_num + 1,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                        },
                    )
                )

        doc.close()
        return elements
