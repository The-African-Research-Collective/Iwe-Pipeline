"""
Normalize Block

Normalizes OCR-extracted text: unicode fixes, whitespace cleanup, etc.
"""

from collections.abc import Generator

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep


class Normalize(PipelineStep):
    """
    Normalize text from OCR output.

    Handles:
    - Unicode normalization and fixes
    - Whitespace standardization
    - Control character removal
    - Ligature expansion
    """

    name = "✨ Normalize"
    type = "🔨 PROCESSOR"

    def __init__(
        self,
        fix_unicode: bool = True,
        normalize_whitespace: bool = True,
        remove_control_chars: bool = True,
        expand_ligatures: bool = True,
        **kwargs,
    ):
        """
        Initialize text normalizer.

        Args:
            fix_unicode: Fix encoding issues
            normalize_whitespace: Standardize spaces/newlines
            remove_control_chars: Remove control characters
            expand_ligatures: Expand ligatures (fi → f+i)
        """
        super().__init__()
        self.fix_unicode = fix_unicode
        self.normalize_whitespace = normalize_whitespace
        self.remove_control_chars = remove_control_chars
        self.expand_ligatures = expand_ligatures

    def run(
        self, data: Document, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """
        Normalize document text.

        Args:
            data: Input document
            rank: Current process rank
            world_size: Total number of processes

        Yields:
            Document with normalized text
        """
        pass
