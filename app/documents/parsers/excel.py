import pandas as pd
from typing import List
from app.documents.parsers.base import BaseParser
from app.documents.schemas import ParsedElement
import io


class ExcelParser(BaseParser):
    """Parse Excel (.xlsx/.xls) files using pandas + openpyxl."""

    ROWS_PER_CHUNK = 10  # Group rows to avoid tiny chunks

    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedElement]:
        elements = []
        xlsx = pd.ExcelFile(io.BytesIO(file_bytes))

        for sheet_name in xlsx.sheet_names:
            df = xlsx.parse(sheet_name)
            df = df.dropna(how="all")  # Drop empty rows

            if df.empty:
                continue

            headers = list(df.columns)

            # Add sheet header
            elements.append(
                ParsedElement(
                    text=f"Sheet: {sheet_name} | Columns: {', '.join(str(h) for h in headers)}",
                    element_type="heading",
                    metadata={
                        "source": filename,
                        "sheet": sheet_name,
                        "total_rows": len(df),
                    },
                )
            )

            # Group rows into chunks
            for start_idx in range(0, len(df), self.ROWS_PER_CHUNK):
                chunk_df = df.iloc[start_idx : start_idx + self.ROWS_PER_CHUNK]
                row_texts = []

                for _, row in chunk_df.iterrows():
                    row_parts = []
                    for col in headers:
                        val = row[col]
                        if pd.notna(val):
                            row_parts.append(f"{col}: {val}")
                    if row_parts:
                        row_texts.append(" | ".join(row_parts))

                if row_texts:
                    elements.append(
                        ParsedElement(
                            text="\n".join(row_texts),
                            element_type="table_row",
                            metadata={
                                "source": filename,
                                "sheet": sheet_name,
                                "row_range": f"{start_idx + 1}-{start_idx + len(chunk_df)}",
                                "headers": [str(h) for h in headers],
                            },
                        )
                    )

        return elements
