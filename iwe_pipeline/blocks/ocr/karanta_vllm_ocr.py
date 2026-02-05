"""
KarantaOCR vLLM Block

Runs OCR using KarantaOCR model via vLLM (or compatible) inference server.
Supports both document-level and page-level OCR.
"""

from collections.abc import Generator
from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep


class KarantaVLLMOCR(PipelineStep):
    """
    OCR extraction using KarantaOCR via vLLM inference server.

    Supports:
    - Document-level OCR (processes entire PDF)
    - Page-level OCR (processes individual page images)
    """

    name = "🔍 KarantaOCR"
    type = "🔨 PROCESSOR"

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model: str = "taresco/KarantaOCR",
        mode: Literal["document", "page"] = "document",
        batch_size: int = 1,
        timeout: int = 600,
        max_concurrent: int = 50,
        **kwargs,
    ):
        """
        Initialize KarantaOCR processor.

        Args:
            server_url: vLLM server endpoint
            model: Model name/path
            mode: "document" for full PDF or "page" for single page
            batch_size: Number of pages/docs to process in parallel
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
        """
        super().__init__()
        self.server_url = server_url
        self.model = model
        self.mode = mode
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_concurrent = max_concurrent

    def run(
        self, data: Document, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """
        Process document/page through OCR.

        Args:
            data: Input document with PDF or image
            rank: Current process rank
            world_size: Total number of processes

        Yields:
            Document with extracted text
        """
        pass
