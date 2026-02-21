import json
from typing import List
from app.documents.parsers.base import BaseParser
from app.documents.schemas import ParsedElement


class JSONParser(BaseParser):
    """Parse JSON files into structured elements."""

    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedElement]:
        elements = []
        data = json.loads(file_bytes.decode("utf-8"))
        self._extract_elements(data, elements, filename, path="root")
        return elements

    def _extract_elements(
        self, data, elements: List[ParsedElement], filename: str, path: str
    ):
        """Recursively extract elements from JSON structure."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}"

                if isinstance(value, (str, int, float, bool)):
                    elements.append(
                        ParsedElement(
                            text=f"{key}: {value}",
                            element_type="key_value",
                            metadata={
                                "source": filename,
                                "json_path": current_path,
                            },
                        )
                    )
                elif isinstance(value, dict):
                    # Serialize small dicts inline, recurse large ones
                    serialized = json.dumps(value, indent=2)
                    if len(serialized) < 500:
                        elements.append(
                            ParsedElement(
                                text=f"{key}: {serialized}",
                                element_type="key_value",
                                metadata={
                                    "source": filename,
                                    "json_path": current_path,
                                },
                            )
                        )
                    else:
                        elements.append(
                            ParsedElement(
                                text=f"Section: {key}",
                                element_type="heading",
                                metadata={
                                    "source": filename,
                                    "json_path": current_path,
                                },
                            )
                        )
                        self._extract_elements(value, elements, filename, current_path)

                elif isinstance(value, list):
                    self._extract_elements(value, elements, filename, current_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]"

                if isinstance(item, (str, int, float, bool)):
                    elements.append(
                        ParsedElement(
                            text=str(item),
                            element_type="list_item",
                            metadata={
                                "source": filename,
                                "json_path": item_path,
                            },
                        )
                    )
                elif isinstance(item, dict):
                    serialized = json.dumps(item, indent=2)
                    if len(serialized) < 800:
                        elements.append(
                            ParsedElement(
                                text=serialized,
                                element_type="paragraph",
                                metadata={
                                    "source": filename,
                                    "json_path": item_path,
                                },
                            )
                        )
                    else:
                        self._extract_elements(item, elements, filename, item_path)
                elif isinstance(item, list):
                    self._extract_elements(item, elements, filename, item_path)
