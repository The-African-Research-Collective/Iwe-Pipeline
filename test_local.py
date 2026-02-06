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
from datatrove.pipeline.inference.run_inference import (
    InferenceConfig,
    InferenceRunner,
)
from datatrove.pipeline.readers.base import BaseReader
from datatrove.pipeline.writers import JsonlWriter

# Import actual pipeline components
from iwe_pipeline.blocks.ocr.split_pages import SplitPages
from iwe_pipeline.datamodel.ids import generate_doc_id
from iwe_pipeline.utils.utils import rollout_postprocess

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LocalPDFReader(BaseReader):
    """
    Reader for local PDF files.

    Reads PDFs from a directory and yields Document objects.
    """

    name = "Local PDF"

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

            yield Document(
                id=doc_id,
                text="",
                metadata={
                    "source": {
                        "path": str(pdf_path),
                        "name": pdf_path.name,
                        "num_pages": None,
                    }
                },
                media=[],
            )


def build_pipeline(output_dir: str, server_url: str = "http://localhost:8000/v1"):
    """
    Build full pipeline including OCR (requires inference server).
    """
    pipeline = [
        SplitPages(),
        InferenceRunner(
            rollout_fn=rollout_postprocess,
            config=InferenceConfig(
                model_name_or_path="taresco/KarantaOCR",
                default_generation_params={"temperature": 0.0},
                max_concurrent_generations=2,
                server_type="endpoint",
                metric_interval=100,
                endpoint_url=server_url,
            ),
            output_writer=JsonlWriter(
                output_folder=output_dir,
            ),
            shared_context={
                "model_name_or_path": "taresco/KarantaOCR",
                "max_tokens": 8192,
            },
        ),
    ]

    return pipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Iwe-Pipeline on local PDFs using real components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
# Test postprocessing without OCR server
python test_local.py --input-dir test_pdfs
        """,
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory containing PDF files to process"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./test_output", help="Directory for output files"
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
    logger.info(f"Input: {input_dir} ({len(pdf_files)} PDFs)")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Tasks: {args.tasks}, Workers: {args.workers}")

    logger.info(f"Building page-level pipeline with OCR server at {args.server_url}")
    pipeline_blocks = build_pipeline(args.output_dir, args.server_url)

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
