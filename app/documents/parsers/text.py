import re
from typing import List
from app.documents.parsers.base import BaseParser
from app.documents.schemas import ParsedElement


class TextParser(BaseParser):
    """Parse plain text files by splitting into paragraphs."""

    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedElement]:
        elements = []
        text = file_bytes.decode("utf-8", errors="replace")

        # Split by double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n", text)

        for para in paragraphs:
            cleaned = self._clean_text(para)
            if cleaned and len(cleaned) > 3:
                elements.append(
                    ParsedElement(
                        text=cleaned,
                        element_type="paragraph",
                        metadata={"source": filename},
                    )
                )

        return elements
