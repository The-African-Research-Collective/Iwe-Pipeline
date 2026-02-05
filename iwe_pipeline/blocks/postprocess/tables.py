"""
Table Cleaning Block

Detects and cleans table structures in OCR output.
"""

from collections.abc import Generator

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep


class TableCleaner(PipelineStep):
    """
    Clean and format tables from OCR output.

    Handles:
    - Table detection
    - Table formatting (markdown/CSV)
    - Malformed table removal
    """

    name = "📊 Tables"
    type = "🔨 PROCESSOR"

    def __init__(
        self,
        detect: bool = True,
        format: bool = True,
        remove_malformed: bool = True,
        **kwargs,
    ):
        """
        Initialize table cleaner.

        Args:
            detect: Detect table structures
            format: Format tables to markdown/CSV
            remove_malformed: Remove corrupted tables
        """
        super().__init__()
        self.detect = detect
        self.format = format
        self.remove_malformed = remove_malformed

    def run(
        self, data: Document, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """
        Clean tables in document.

        Args:
            data: Input document
            rank: Current process rank
            world_size: Total number of processes

        Yields:
            Document with cleaned tables
        """
        pass
