from abc import ABC, abstractmethod
from typing import List
from app.documents.schemas import ParsedElement


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedElement]:
        """
        Parse a document and return structured elements.

        Args:
            file_bytes: Raw file content
            filename: Original filename (used for type detection)

        Returns:
            List of ParsedElement with text, type, page number, and metadata
        """
        pass

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove excessive whitespace while preserving paragraph breaks."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned.append(stripped)
            elif cleaned and cleaned[-1] != "":
                cleaned.append("")
        return "\n".join(cleaned).strip()
