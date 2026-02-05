#!/usr/bin/env python3
"""
Fetch + OCR Pipeline

This pipeline fetches PDFs from Azure Blob Storage and extracts text using OCR.

Pipeline: Azure Blob → Fetch → OCR → Index

Usage:
    python scripts/run_fetch_ocr.py --config configs/stages/fetch_ocr.yaml
"""

import argparse
import logging

from datatrove.executor.local import LocalPipelineExecutor

from iwe_pipeline.server.manager import ServerManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load pipeline configuration from YAML file."""
    pass


def build_pipeline(config: dict) -> list:
    """
    Build the fetch + OCR pipeline from configuration.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        List of pipeline steps
    """
    pass


def main():
    """Main entry point for fetch + OCR pipeline."""
    parser = argparse.ArgumentParser(description="Run Fetch + OCR pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stages/fetch_ocr.yaml",
        help="Path to pipeline configuration file",
    )
    parser.add_argument(
        "--start-server", action="store_true", help="Start OCR inference server automatically"
    )
    parser.add_argument(
        "--tasks", type=int, default=None, help="Number of tasks (overrides config)"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of workers per task (overrides config)"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Iwe-Pipeline: Fetch + OCR")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)

    # Start server if requested
    server = None
    if args.start_server:
        logger.info("Starting OCR inference server...")
        server = ServerManager(
            server_type=config.get("ocr", {}).get("server_type", "vllm"),
            model_name_or_path=config.get("ocr", {}).get("model", "taresco/KarantaOCR"),
            port=8000,
        )
        server.start()

    # Build pipeline
    pipeline = build_pipeline(config)

    # Execute
    tasks = args.tasks or config.get("executor", {}).get("tasks", 1)
    workers = args.workers or config.get("executor", {}).get("workers", 1)

    logger.info(f"Running pipeline with {tasks} tasks, {workers} workers")

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=tasks,
        workers=workers,
        logging_dir=config.get("executor", {}).get("logging_dir", "./logs/fetch_ocr"),
    )

    executor.run()

    # Cleanup
    if server:
        logger.info("Stopping OCR inference server...")
        server.stop()

    logger.info("=" * 80)
    logger.info("Fetch + OCR pipeline completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
