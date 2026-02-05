"""
Language Tagging Block

Identifies document language using GlotLID or similar backend.
Adds language code and confidence to metadata.
"""

from collections.abc import Generator
from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep


class LanguageTag(PipelineStep):
    """
    Detect and tag document language.

    Uses GlotLID, fastText, or similar for language identification.
    Adds language code (e.g., "eng_Latn") to metadata.
    """

    name = "🌍 Language Tag"
    type = "🔨 PROCESSOR"

    def __init__(
        self,
        backend: Literal["glotlid", "fasttext"] = "glotlid",
        threshold: float = 0.5,
        **kwargs,
    ):
        """
        Initialize language tagger.

        Args:
            backend: Language detection backend
            threshold: Minimum confidence threshold
        """
        super().__init__()
        self.backend = backend
        self.threshold = threshold

    def run(
        self, data: Document, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """
        Detect language and add to metadata.

        Args:
            data: Input document
            rank: Current process rank
            world_size: Total number of processes

        Yields:
            Document with language metadata
        """
        pass
