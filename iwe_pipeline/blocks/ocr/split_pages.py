"""
Split Pages Block

Converts a PDF document into separate page documents for page-level OCR.
Each page becomes a Document with id "parent_id::p0000".
"""

from collections.abc import Generator

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from loguru import logger

from iwe_pipeline.utils.utils import get_pdf_num_pages, render_pdf_to_base64png


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
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
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
        # Get number of pages
        for document in data:
            pdf_path = str(document.metadata["source"]["path"])
            num_pages = get_pdf_num_pages(pdf_path)
            logger.info(f"PDF has {num_pages} pages.")

            for page in range(1, num_pages + 1):
                logger.info(f"  Page {page}/{num_pages}")
                page_bytes = render_pdf_to_base64png(pdf_path, page_num=page)

                page_document = Document(
                    id=f"{document.id}::p{page:04d}",
                    text=document.text,
                    metadata={
                        **document.metadata,
                        "source": {
                            **document.metadata.get("source", {}),
                            "page": page,
                            "num_pages": num_pages,
                        },
                    },
                    media=[{"media_bytes": page_bytes, "media_type": "application/pdf"}],
                )

                yield page_document
