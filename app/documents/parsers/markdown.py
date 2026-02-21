import re
from typing import List
from app.documents.parsers.base import BaseParser
from app.documents.schemas import ParsedElement


class MarkdownParser(BaseParser):
    """Parse Markdown files by splitting on headers."""

    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedElement]:
        elements = []
        text = file_bytes.decode("utf-8", errors="replace")

        # Split by markdown headers (# through ######)
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        sections = header_pattern.split(text)

        # First section (before any headers)
        preamble = sections[0].strip() if sections else ""
        if preamble:
            elements.append(
                ParsedElement(
                    text=self._clean_text(preamble),
                    element_type="paragraph",
                    metadata={"source": filename, "section": "Introduction"},
                )
            )

        # Process header + content pairs
        i = 1
        current_heading = None
        while i < len(sections):
            if i + 1 < len(sections):
                heading_level = len(sections[i])  # Number of # chars
                heading_text = sections[i + 1].strip()
                current_heading = heading_text

                elements.append(
                    ParsedElement(
                        text=heading_text,
                        element_type="heading",
                        metadata={
                            "source": filename,
                            "heading_level": heading_level,
                        },
                    )
                )

                # Content after this heading (before next header)
                if i + 2 < len(sections):
                    content = sections[i + 2].strip()
                    if content:
                        # Split large content blocks into paragraphs
                        paragraphs = re.split(r"\n\s*\n", content)
                        for para in paragraphs:
                            cleaned = self._clean_text(para)
                            if cleaned and len(cleaned) > 3:
                                elements.append(
                                    ParsedElement(
                                        text=cleaned,
                                        element_type="paragraph",
                                        metadata={
                                            "source": filename,
                                            "section": current_heading or "Document",
                                        },
                                    )
                                )
                    i += 3
                else:
                    i += 2
            else:
                i += 1

        return elements
