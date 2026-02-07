"""
Split Pages Block

Converts a PDF document into separate page documents for page-level OCR.
Each page becomes a Document with id "parent_id::p0000".
"""

import gzip
import json
from collections.abc import Generator
from pathlib import Path

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

    def __init__(
        self,
        output_dir: str | None = None,
        processed_ids_path: str | None = None,
        output_glob: str | None = None,
        **kwargs,
    ):
        """Initialize page splitter."""
        super().__init__()
        self.output_dir = Path(output_dir) if output_dir else None
        self.processed_ids_path = Path(processed_ids_path) if processed_ids_path else None
        self.output_glob = output_glob

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
        processed_ids = self._load_processed_ids()

        for document in data:
            pdf_path = str(document.metadata["source"]["path"])
            num_pages = get_pdf_num_pages(pdf_path)
            logger.info(f"PDF has {num_pages} pages.")

            for page in range(1, num_pages + 1):
                logger.info(f"  Page {page}/{num_pages}")
                page_bytes = render_pdf_to_base64png(pdf_path, page_num=page)

                page_id = f"{document.id}::p{page:04d}"
                if page_id in processed_ids:
                    logger.info(f"  Skipping already processed page {page}/{num_pages}")
                    continue

                page_document = Document(
                    id=page_id,
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

    def _load_processed_ids(self) -> set[str]:
        processed_ids: set[str] = set()

        if self.processed_ids_path and self.processed_ids_path.exists():
            try:
                with self.processed_ids_path.open(
                    "rt", encoding="utf-8", errors="ignore"
                ) as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            processed_ids.add(line)
                logger.info(f"Loaded {len(processed_ids)} processed ids from cache.")
                return processed_ids
            except Exception as exc:
                logger.warning(f"Failed to read processed ids cache: {exc}")

        if self.output_dir and self.output_dir.exists():
            output_files = []
            if self.output_glob:
                output_files.extend(self.output_dir.glob(self.output_glob))
            else:
                output_files.extend(self.output_dir.glob("*.jsonl"))
                output_files.extend(self.output_dir.glob("*.jsonl.gz"))
                output_files.extend(self.output_dir.glob("*.parquet.gz"))

                checkpoints_dir = self.output_dir / "checkpoints"
                if checkpoints_dir.exists():
                    output_files.extend(checkpoints_dir.rglob("chunk_*.jsonl"))

            for path in sorted(set(output_files)):
                for doc_id in self._iter_ids_from_file(path):
                    processed_ids.add(doc_id)

            logger.info(f"Found {len(processed_ids)} processed ids from outputs.")

            if self.processed_ids_path:
                try:
                    self.processed_ids_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.processed_ids_path.open("wt", encoding="utf-8") as handle:
                        for doc_id in sorted(processed_ids):
                            handle.write(f"{doc_id}\n")
                except Exception as exc:
                    logger.warning(f"Failed to write processed ids cache: {exc}")

        return processed_ids

    @staticmethod
    def _iter_ids_from_file(path: Path) -> Generator[str, None, None]:
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    doc_id = payload.get("id")
                    if doc_id:
                        yield doc_id
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning(f"Failed to scan ids from {path}: {exc}")
