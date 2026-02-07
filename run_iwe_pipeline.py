#!/usr/bin/env python3
"""
Local Testing Script - Uses Real iwe_pipeline Components

Test the actual Iwe-Pipeline blocks on local PDFs.

Usage:
    python run_iwe_pipeline.py --config configs/run_iwe_sample.yaml
"""

import argparse
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

import yaml
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
from iwe_pipeline.monitoring.tracker import OCRInferenceProgressMonitor
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


def build_pipeline(
    output_dir: str,
    server_url: str,
    *,
    model_name: str,
    temperature: float,
    max_concurrent: int,
    max_tokens: int,
    output_filename: str,
):
    """
    Build full pipeline including OCR (requires inference server).
    """
    pipeline = [
        SplitPages(
            output_dir=output_dir,
            processed_ids_path=f"{output_dir}/processed_ids.txt",
        ),
        InferenceRunner(
            rollout_fn=rollout_postprocess,
            config=InferenceConfig(
                model_name_or_path=model_name,
                default_generation_params={"temperature": temperature},
                max_concurrent_generations=max_concurrent,
                server_type="endpoint",
                metric_interval=100,
                endpoint_url=server_url,
            ),
            output_writer=JsonlWriter(
                output_folder=output_dir,
                output_filename=output_filename,
            ),
            shared_context={
                "model_name_or_path": model_name,
                "max_tokens": max_tokens,
            },
            checkpoints_local_dir=f"{output_dir}/checkpoints",
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
        "--config",
        type=str,
        default="configs/run_iwe_sample.yaml",
        help="Path to config file in configs/ directory",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    with config_path.open("rt", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    data_cfg = config.get("data", {})
    ocr_cfg = config.get("ocr", {})
    executor_cfg = config.get("executor", {})

    input_dir_value = config.get("input_dir") or data_cfg.get("fetched")
    output_dir_value = config.get("output_dir") or data_cfg.get("ocr_extracted")
    server_url = ocr_cfg.get("server_url")
    limit = config.get("limit")
    tasks = executor_cfg.get("tasks")
    workers = executor_cfg.get("workers")
    monitor = config.get("monitor")
    job_name = config.get("job_name")
    max_concurrent = ocr_cfg.get("max_concurrent")
    max_tokens = ocr_cfg.get("max_tokens")
    model_name = ocr_cfg.get("model_name")
    temperature = ocr_cfg.get("temperature")
    output_filename = config.get("output_filename")

    missing = []
    if not input_dir_value:
        missing.append("input_dir or data.fetched")
    if not output_dir_value:
        missing.append("output_dir or data.ocr_extracted")
    if not server_url:
        missing.append("ocr.server_url")
    if tasks is None:
        missing.append("executor.tasks")
    if workers is None:
        missing.append("executor.workers")
    if limit is None:
        missing.append("limit")
    if monitor is None:
        missing.append("monitor")
    if not job_name:
        missing.append("job_name")
    if max_concurrent is None:
        missing.append("ocr.max_concurrent")
    if max_tokens is None:
        missing.append("ocr.max_tokens")
    if not model_name:
        missing.append("ocr.model_name")
    if temperature is None:
        missing.append("ocr.temperature")
    if not output_filename:
        missing.append("output_filename")

    if missing:
        logger.error("Missing required config values: " + ", ".join(missing))
        return 1

    input_dir = Path(input_dir_value)
    output_dir = Path(output_dir_value)
    if not server_url.endswith("/v1"):
        server_url = server_url.rstrip("/") + "/v1"
    limit = int(limit)
    tasks = int(tasks)
    workers = int(workers)
    monitor = bool(monitor)
    job_name = str(job_name)
    max_concurrent = int(max_concurrent)
    max_tokens = int(max_tokens)
    model_name = str(model_name)
    temperature = float(temperature)
    output_filename = str(output_filename)

    # Validate input directory
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
    logger.info(f"Output: {output_dir}")
    logger.info(f"Tasks: {tasks}, Workers: {workers}")

    logger.info(f"Building page-level pipeline with OCR server at {server_url}")
    pipeline_blocks = build_pipeline(
        str(output_dir),
        server_url,
        model_name=model_name,
        temperature=temperature,
        max_concurrent=max_concurrent,
        max_tokens=max_tokens,
        output_filename=output_filename,
    )

    # Create reader
    reader = LocalPDFReader(input_dir=str(input_dir), limit=limit)

    # Full pipeline
    pipeline = [reader] + pipeline_blocks

    logger.info("=" * 80)

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = f"./logs/{job_name}_run_{run_id}"

    def run_monitor():
        monitor_pipeline = [
            OCRInferenceProgressMonitor(
                output_dir=str(output_dir),
                input_dir=str(input_dir),
                page_level=True,
                port=8040,
                update_interval=5,
                stats_path=f"{run_dir}/stats.json",
            )
        ]

        monitor_executor = LocalPipelineExecutor(
            pipeline=monitor_pipeline,
            tasks=1,
            workers=1,
            logging_dir=f"{run_dir}_monitor",
        )
        monitor_executor.run()
        logger.info("✓ OCR progress monitor stopped.")

    # Execute with LocalPipelineExecutor
    try:
        monitor_thread = None
        if monitor:
            monitor_thread = threading.Thread(target=run_monitor, name="ocr-monitor")
            monitor_thread.start()

        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=tasks,
            workers=workers,
            logging_dir=run_dir,
        )

        executor.run()

        logger.info("=" * 80)
        logger.info("✓ Pipeline completed successfully!")
        logger.info(f"✓ Output: {output_dir}/output_*.jsonl.gz")
        logger.info("=" * 80)

        if monitor_thread is not None:
            monitor_thread.join()

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
