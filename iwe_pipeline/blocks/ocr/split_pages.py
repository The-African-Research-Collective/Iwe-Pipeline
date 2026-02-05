"""
Split Pages Block

Converts a PDF document into separate page documents for page-level OCR.
Each page becomes a Document with id "parent_id::p0000".
"""

from collections.abc import Generator

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep


class SplitPages(PipelineStep):
    """
    Split PDF into page-level documents.

    Takes a document with a PDF file and creates one Document per page.
    Useful for page-level OCR processing.
    """

    name = "📄 Split Pages"
    type = "🔨 PROCESSOR"

    def __init__(self, **kwargs):
        """Initialize page splitter."""
        super().__init__()

    def run(
        self, data: Document, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """
        Split document into pages.

        Args:
            data: Input document with PDF path
            rank: Current process rank
            world_size: Total number of processes

        Yields:
            One Document per page
        """
        pass
