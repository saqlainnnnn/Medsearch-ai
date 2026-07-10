"""
Custom exceptions for the document ingestion pipeline.
"""


class IngestionError(Exception):
    """
    Base exception for all ingestion-related errors.
    """


class PDFLoadError(IngestionError):
    """
    Raised when a PDF cannot be loaded.
    """


class DocumentParseError(IngestionError):
    """
    Raised when a document cannot be parsed.
    """


class TextCleaningError(IngestionError):
    """
    Raised when text cleaning fails.
    """