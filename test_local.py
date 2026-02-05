#!/usr/bin/env python3
"""
Local Testing Script - Uses Real iwe_pipeline Components

Test the actual Iwe-Pipeline blocks on local PDFs.

Usage:
    python test_local.py --input-dir ./test_pdfs --output-dir ./test_output
"""

import argparse
import logging
from pathlib import Path

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers.base import BaseReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

from iwe_pipeline.blocks.assemble.group_pages import GroupPages
from iwe_pipeline.blocks.ocr.karanta_vllm_ocr import KarantaVLLMOCR

# Import actual pipeline components
from iwe_pipeline.blocks.ocr.split_pages import SplitPages
from iwe_pipeline.blocks.postprocess.boilerplate import BoilerplateRemover
from iwe_pipeline.blocks.postprocess.language_tag import LanguageTag
from iwe_pipeline.blocks.postprocess.normalize import Normalize
from iwe_pipeline.blocks.postprocess.tables import TableCleaner
from iwe_pipeline.blocks.quality.bert_score import BertQualityScore
from iwe_pipeline.datamodel.ids import generate_doc_id

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LocalPDFReader(BaseReader):
    """
    Reader for local PDF files.

    Reads PDFs from a directory and yields Document objects.
    """

    name = "📁 Local PDF"

    def __init__(self, input_dir: str, limit: int = -1, skip: int = 0, **kwargs):
        super().__init__(limit, skip, **kwargs)
        self.input_dir = Path(input_dir)

    def run(self, data=None, rank=0, world_size=1):
        """Yield Documents from local PDFs."""
        pdf_files = sorted(self.input_dir.glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files in {self.input_dir}")

        for idx, pdf_path in enumerate(pdf_files):
            # Apply skip and limit
            if idx < self.skip:
                continue
            if self.limit != -1 and idx >= self.skip + self.limit:
                break

            # Generate stable ID
            doc_id = generate_doc_id(str(pdf_path))

            # Read PDF bytes
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            logger.info(f"[{idx+1}/{len(pdf_files)}] Reading: {pdf_path.name}")

            yield Document(
                id=doc_id,
                text="",
                metadata={
                    "source": {
                        "path": str(pdf_path),
                        "name": pdf_path.name,
                        "size": len(pdf_bytes),
                    }
                },
                media=[{"media_bytes": pdf_bytes, "media_type": "application/pdf"}],
            )


def build_minimal_pipeline(output_dir: str):
    """
    Build a minimal test pipeline using real components.

    This pipeline skips OCR (requires inference server) but tests
    all other components with dummy text.
    """
    pipeline = [
        # Postprocessing (works on text)
        LanguageTag(backend="glotlid", threshold=0.5),
        Normalize(fix_unicode=True, normalize_whitespace=True, remove_control_chars=True),
        BoilerplateRemover(remove_headers=True, remove_footers=True, min_text_length=100),
        TableCleaner(detect=True, format=True),
        # Quality scoring
        BertQualityScore(
            model="bert-base-multilingual-cased", threshold=0.5, filter_low_quality=False
        ),
        # Writer
        JsonlWriter(
            output_folder=output_dir, output_filename="output_${rank}.jsonl.gz", compression="gz"
        ),
    ]

    return pipeline


def build_full_pipeline(output_dir: str, server_url: str = "http://localhost:8000"):
    """
    Build full pipeline including OCR (requires inference server).
    """
    pipeline = [
        # OCR extraction
        KarantaVLLMOCR(server_url=server_url, mode="document", batch_size=1, timeout=600),
        # Postprocessing
        LanguageTag(backend="glotlid", threshold=0.5),
        Normalize(fix_unicode=True, normalize_whitespace=True, remove_control_chars=True),
        BoilerplateRemover(remove_headers=True, remove_footers=True, min_text_length=100),
        TableCleaner(detect=True, format=True),
        # Quality scoring
        BertQualityScore(
            model="bert-base-multilingual-cased", threshold=0.5, filter_low_quality=False
        ),
        # Writer
        JsonlWriter(
            output_folder=output_dir, output_filename="output_${rank}.jsonl.gz", compression="gz"
        ),
    ]

    return pipeline


def build_page_level_pipeline(output_dir: str, server_url: str = "http://localhost:8000"):
    """
    Build page-level pipeline (splits PDFs into pages, processes, then groups).
    """
    pipeline = [
        # Split into pages
        SplitPages(),
        # OCR per page
        KarantaVLLMOCR(server_url=server_url, mode="page", batch_size=1, timeout=600),
        # Group pages back
        GroupPages(join_separator="\n\n"),
        # Postprocessing
        LanguageTag(backend="glotlid", threshold=0.5),
        Normalize(fix_unicode=True, normalize_whitespace=True, remove_control_chars=True),
        BoilerplateRemover(remove_headers=True, remove_footers=True, min_text_length=100),
        TableCleaner(detect=True, format=True),
        # Quality scoring
        BertQualityScore(
            model="bert-base-multilingual-cased", threshold=0.5, filter_low_quality=False
        ),
        # Writer
        JsonlWriter(
            output_folder=output_dir, output_filename="output_${rank}.jsonl.gz", compression="gz"
        ),
    ]

    return pipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Iwe-Pipeline on local PDFs using real components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Modes:
  minimal     - Test postprocessing only (no OCR server needed)
  full        - Full pipeline with document-level OCR (requires server)
  page-level  - Page-level OCR with split/group (requires server)

Examples:
  # Test postprocessing without OCR server
  python test_local.py --input-dir test_pdfs --mode minimal

  # Full pipeline (requires OCR server at http://localhost:8000)
  python test_local.py --input-dir test_pdfs --mode full --server-url http://localhost:8000

  # Page-level processing
  python test_local.py --input-dir test_pdfs --mode page-level --server-url http://localhost:8000
        """,
    )

    parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory containing PDF files to process"
    )

    parser.add_argument(
        "--output-dir", type=str, default="./test_output", help="Directory for output files"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["minimal", "full", "page-level"],
        default="minimal",
        help="Pipeline mode (minimal=no OCR, full=with OCR, page-level=split pages)",
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="OCR inference server URL (for full/page-level modes)",
    )

    parser.add_argument(
        "--limit", type=int, default=-1, help="Limit number of PDFs to process (-1 for all)"
    )

    parser.add_argument("--tasks", type=int, default=1, help="Number of parallel tasks")

    parser.add_argument("--workers", type=int, default=1, help="Number of workers per task")

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        logger.info(f"Create it with: mkdir -p {input_dir}")
        return 1

    # Check for PDFs
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in: {input_dir}")
        return 1

    logger.info("=" * 80)
    logger.info("Iwe-Pipeline: Local Test with Real Components")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Input: {input_dir} ({len(pdf_files)} PDFs)")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Tasks: {args.tasks}, Workers: {args.workers}")

    # Build pipeline based on mode
    if args.mode == "minimal":
        logger.info("Building minimal pipeline (no OCR)")
        logger.warning("Note: This mode skips OCR. Documents need existing text.")
        pipeline_blocks = build_minimal_pipeline(args.output_dir)
    elif args.mode == "full":
        logger.info(f"Building full pipeline with OCR server at {args.server_url}")
        pipeline_blocks = build_full_pipeline(args.output_dir, args.server_url)
    else:  # page-level
        logger.info(f"Building page-level pipeline with OCR server at {args.server_url}")
        pipeline_blocks = build_page_level_pipeline(args.output_dir, args.server_url)

    # Create reader
    reader = LocalPDFReader(input_dir=str(input_dir), limit=args.limit)

    # Full pipeline
    pipeline = [reader] + pipeline_blocks

    logger.info("=" * 80)

    # Execute with LocalPipelineExecutor
    try:
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=args.tasks,
            workers=args.workers,
            logging_dir="./logs/test_local",
        )

        executor.run()

        logger.info("=" * 80)
        logger.info("✓ Pipeline completed successfully!")
        logger.info(f"✓ Output: {args.output_dir}/output_*.jsonl.gz")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
