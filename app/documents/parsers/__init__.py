from typing import List, Type
from app.documents.parsers.base import BaseParser
from app.documents.parsers.pdf import PDFParser
from app.documents.parsers.docx import DocxParser
from app.documents.parsers.excel import ExcelParser
from app.documents.parsers.csv_parser import CSVParser
from app.documents.parsers.json_parser import JSONParser
from app.documents.parsers.markdown import MarkdownParser
from app.documents.parsers.text import TextParser
from app.documents.schemas import ParsedElement

# Map file extensions to their parser classes
PARSER_MAP: dict[str, Type[BaseParser]] = {
    ".pdf": PDFParser,
    ".docx": DocxParser,
    ".xlsx": ExcelParser,
    ".xls": ExcelParser,
    ".csv": CSVParser,
    ".json": JSONParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".txt": TextParser,
}

SUPPORTED_EXTENSIONS = set(PARSER_MAP.keys())


def get_parser(filename: str) -> BaseParser:
    """Get the appropriate parser for a file based on its extension."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    parser_class = PARSER_MAP.get(ext)
    if not parser_class:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return parser_class()


def parse_document(file_bytes: bytes, filename: str) -> List[ParsedElement]:
    """Parse a document and return structured elements."""
    parser = get_parser(filename)
    return parser.parse(file_bytes, filename)
