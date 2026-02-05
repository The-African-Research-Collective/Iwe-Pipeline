"""
Azure Manifest Reader

Reads a manifest file (JSONL/CSV) containing blob URLs and metadata.
This is the preferred approach for reproducible, shardable processing.
"""

from collections.abc import Generator

from datatrove.data import Document
from datatrove.pipeline.readers.base import BaseReader


class AzureManifestReader(BaseReader):
    """
    Reader for Azure blob manifests.

    Reads a JSONL/CSV file with blob metadata and yields Documents.
    Each line should contain: blob_url, blob_name, blob_size, etc.
    """

    name = "📋 Azure Manifest"

    def __init__(
        self,
        manifest_path: str,
        format: str = "jsonl",
        limit: int = -1,
        skip: int = 0,
        **kwargs,
    ):
        """
        Initialize Azure manifest reader.

        Args:
            manifest_path: Path to manifest file
            format: File format ('jsonl' or 'csv')
            limit: Maximum number of documents to read (-1 for unlimited)
            skip: Number of documents to skip
        """
        super().__init__(limit, skip, **kwargs)
        self.manifest_path = manifest_path
        self.format = format

    def run(
        self, data: Document = None, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """
        Read manifest and yield Documents with blob metadata.

        Args:
            data: Input document (unused for readers)
            rank: Current process rank for distributed processing
            world_size: Total number of processes

        Yields:
            Document objects with blob URLs and metadata
        """
        pass
