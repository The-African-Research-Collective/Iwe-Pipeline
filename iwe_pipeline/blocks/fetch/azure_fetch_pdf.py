"""
Azure Fetch PDF Block

Downloads PDFs from Azure Blob Storage to local disk.
Caches downloads and adds file paths to document metadata.
"""

from collections.abc import Generator
from pathlib import Path

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep


class AzureFetchPDF(PipelineStep):
    """
    Fetch PDF files from Azure Blob Storage.

    Downloads blobs to local disk, caches based on etag,
    and updates document with local file path.
    """

    name = "📦 Azure Fetch PDF"
    type = "🔨 PROCESSOR"

    def __init__(
        self,
        download_dir: str = "./data/fetched",
        connection_string: str = None,
        cache_enabled: bool = True,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Initialize Azure PDF fetcher.

        Args:
            download_dir: Local directory for downloaded PDFs
            connection_string: Azure Storage connection string
            cache_enabled: Whether to skip downloads if file exists
            max_retries: Maximum retry attempts for failed downloads
        """
        super().__init__()
        self.download_dir = Path(download_dir)
        self.connection_string = connection_string
        self.cache_enabled = cache_enabled
        self.max_retries = max_retries

        self.download_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self, data: Document, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """
        Download PDF and update document.

        Args:
            data: Input document with blob_url in metadata
            rank: Current process rank
            world_size: Total number of processes

        Yields:
            Document with local PDF path in metadata
        """
        pass
