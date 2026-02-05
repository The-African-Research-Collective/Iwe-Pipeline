"""
Boilerplate Removal Block

Removes repetitive headers, footers, and page numbers from OCR output.
"""

from collections.abc import Generator

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep


class BoilerplateRemover(PipelineStep):
    """
    Remove boilerplate text from documents.

    Detects and removes:
    - Page headers and footers
    - Page numbers
    - Repeated patterns
    """

    name = "🗑️  Boilerplate"
    type = "🔨 PROCESSOR"

    def __init__(
        self,
        remove_headers: bool = True,
        remove_footers: bool = True,
        remove_page_numbers: bool = True,
        min_text_length: int = 100,
        **kwargs,
    ):
        """
        Initialize boilerplate remover.

        Args:
            remove_headers: Remove header text
            remove_footers: Remove footer text
            remove_page_numbers: Remove page numbers
            min_text_length: Minimum text length after removal
        """
        super().__init__()
        self.remove_headers = remove_headers
        self.remove_footers = remove_footers
        self.remove_page_numbers = remove_page_numbers
        self.min_text_length = min_text_length

    def run(
        self, data: Document, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """
        Remove boilerplate from document.

        Args:
            data: Input document
            rank: Current process rank
            world_size: Total number of processes

        Yields:
            Document with boilerplate removed
        """
        pass
