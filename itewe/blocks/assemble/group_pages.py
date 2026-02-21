"""
Group Pages Block

Reassembles page-level documents back into document-level documents.
Reverses the split_pages operation.
"""

from collections.abc import Generator

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep


class GroupPages(PipelineStep):
    """
    Group page documents back into parent documents.

    Takes Documents with IDs like "doc_id::p0000" and groups them
    back into a single Document per parent doc_id.
    """

    name = "📑 Group Pages"
    type = "🔨 PROCESSOR"

    def __init__(
        self,
        join_separator: str = "\n\n",
        **kwargs,
    ):
        """
        Initialize page grouper.

        Args:
            join_separator: Separator for joining page texts
        """
        super().__init__()
        self.join_separator = join_separator

    def run(
        self, data: Document, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """
        Group pages into documents.

        This requires buffering all pages before yielding.
        In practice, use datatrove's grouping utilities or
        implement stateful grouping.

        Args:
            data: Input page document
            rank: Current process rank
            world_size: Total number of processes

        Yields:
            Grouped document
        """
        pass
