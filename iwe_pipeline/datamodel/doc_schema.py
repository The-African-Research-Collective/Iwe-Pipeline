"""
Document schema conventions and metadata field definitions.

This module defines the standard metadata fields used throughout the pipeline
for consistent Document representation.
"""

from typing import TypedDict


class OCRMetadata(TypedDict, total=False):
    """OCR-specific metadata fields."""

    model: str
    confidence: float
    language_detected: str
    num_pages: int
    processing_time: float


class SourceMetadata(TypedDict, total=False):
    """Source document metadata."""

    blob_url: str
    blob_name: str
    blob_etag: str
    blob_size: int
    container: str
    fetched_at: str


class PathsMetadata(TypedDict, total=False):
    """File paths for intermediate artifacts."""

    pdf: str
    ocr_json: str
    images: list[str]


class QualityMetadata(TypedDict, total=False):
    """Quality scoring metadata."""

    quality_score: float
    quality_model: str
    scores_per_chunk: list[float]


class DocumentMetadata(TypedDict, total=False):
    """Complete document metadata schema."""

    # Core fields
    id: str
    language: str
    language_confidence: float

    # Source
    source: SourceMetadata

    # Processing
    ocr: OCRMetadata
    paths: PathsMetadata

    # Quality
    quality: QualityMetadata

    # Postprocessing
    normalized: bool
    boilerplate_removed: bool
    tables_cleaned: bool

    # Timestamps
    created_at: str
    processed_at: str


# Field name constants
class Fields:
    """Standard field names used across the pipeline."""

    # Core
    ID = "id"
    TEXT = "text"
    LANGUAGE = "language"

    # Metadata
    METADATA = "metadata"
    SOURCE = "source"
    OCR = "ocr"
    PATHS = "paths"
    QUALITY = "quality"

    # Quality
    QUALITY_SCORE = "quality_score"

    # Processing flags
    NORMALIZED = "normalized"
    BOILERPLATE_REMOVED = "boilerplate_removed"
    TABLES_CLEANED = "tables_cleaned"
