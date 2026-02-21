import pandas as pd
from typing import List
from app.documents.parsers.base import BaseParser
from app.documents.schemas import ParsedElement
import io


class CSVParser(BaseParser):
    """Parse CSV files using pandas."""

    ROWS_PER_CHUNK = 10

    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedElement]:
        elements = []

        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            # Try with different encoding
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")

        df = df.dropna(how="all")

        if df.empty:
            return elements

        headers = list(df.columns)

        elements.append(
            ParsedElement(
                text=f"CSV Data | Columns: {', '.join(str(h) for h in headers)} | Total Rows: {len(df)}",
                element_type="heading",
                metadata={
                    "source": filename,
                    "total_rows": len(df),
                },
            )
        )

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
                            "row_range": f"{start_idx + 1}-{start_idx + len(chunk_df)}",
                            "headers": [str(h) for h in headers],
                        },
                    )
                )

        return elements
