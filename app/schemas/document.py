from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """
    Metadata describing a medical document.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        ...,
        description="Title of the document."
    )

    authors: list[str] = Field(
        default_factory=list,
        description="List of document authors."
    )

    journal: str | None = Field(
        default=None,
        description="Journal or publication name."
    )

    publication_year: int | None = Field(
        default=None,
        ge=1900,
        le=datetime.now().year,
        description="Publication year."
    )

    doi: str | None = Field(
        default=None,
        description="Digital Object Identifier."
    )

    source: str = Field(
        ...,
        description="Source of the document (PubMed, WHO, NIH, CDC, etc.)."
    )

    url: str | None = Field(
        default=None,
        description="Original document URL."
    )

    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords extracted from the document."
    )


class Document(BaseModel):
    """
    Represents one processed medical document.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: UUID = Field(
        default_factory=uuid4,
        description="Unique document identifier."
    )

    metadata: DocumentMetadata

    content: str = Field(
        ...,
        min_length=1,
        description="Cleaned full document text."
    )

    language: str = Field(
        default="en",
        description="Document language."
    )

    word_count: int = Field(
        default=0,
        ge=0,
        description="Total number of words."
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class Chunk(BaseModel):
    """
    Represents one searchable chunk of a document.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: UUID = Field(
        default_factory=uuid4,
        description="Unique chunk identifier."
    )

    document_id: UUID = Field(
        ...,
        description="Parent document ID."
    )

    chunk_index: int = Field(
        ...,
        ge=0,
        description="Sequential chunk number."
    )

    content: str = Field(
        ...,
        min_length=1,
        description="Chunk text."
    )

    page_number: int | None = Field(
        default=None,
        ge=1,
        description="Original PDF page."
    )

    section: str | None = Field(
        default=None,
        description="Document section."
    )

    token_count: int = Field(
        default=0,
        ge=0,
        description="Estimated token count."
    )