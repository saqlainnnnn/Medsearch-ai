from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ParsedPage:
    """
    Represents a single page extracted from a PDF.
    """

    page_number: int
    text: str