#!/usr/bin/env python3
"""
Local Testing Script - Uses Real ìtèwé Components

Test the actual ìtèwé blocks on local PDFs.

Usage:
    python tewe.py \
        reader.backend=azure \
        reader.azure.container_path=az://mycontainer \
        reader.input_dir="mypdfdir" \
        ocr.server_url=http://127.0.0.1:8080 \
        ocr.model_name=mradermacher/KarantaOCR-GGUF \
        output.output_dir=local_output_test
"""

import logging
import os
from datetime import UTC, datetime
from functools import partial

import hydra
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.run_inference import (
    InferenceConfig,
    InferenceRunner,
)
from datatrove.pipeline.writers import JsonlWriter
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from itewe.readers.pdf import PDFReader
from itewe.utils.rollout_utils import rollout_postprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_reader(cfg: DictConfig) -> PDFReader:
    reader = partial(
        PDFReader,
        pdf_to_ppm=cfg.reader.pdf_to_ppm,
        yield_pages_as_documents=cfg.reader.yield_pages_as_documents,
        glob_pattern=cfg.reader.glob_pattern,
        recursive=cfg.reader.recursive,
        limit=cfg.reader.limit,
    )

    if cfg.reader.backend == "local":
        if not os.path.exists(cfg.reader.input_dir):
            raise ValueError(f"Input directory does not exist: {cfg.reader.input_dir}")

        return reader(data_folder=cfg.reader.input_dir)

    elif cfg.reader.backend == "azure":
        from adlfs import AzureBlobFileSystem

        fs = AzureBlobFileSystem()
        data_folder = ("/".join([cfg.reader.azure.container_path, cfg.reader.input_dir]), fs)

        if not fs.exists(data_folder[0]):
            raise ValueError(f"Input directory does not exist: {data_folder[0]}")

        return reader(data_folder=data_folder)

    else:
        raise ValueError(f"Unknown reader backend: {cfg.reader.backend}")


def build_pipeline(cfg: DictConfig) -> list[PipelineStep]:
    """Build full pipeline including OCR (requires inference server)."""
    output_folder = cfg.output.output_dir
    if cfg.output.backend == "azure":
        from adlfs import AzureBlobFileSystem

        fs = AzureBlobFileSystem()
        output_folder = ("/".join([cfg.reader.azure.container_path, output_folder]), fs)

    return [
        InferenceRunner(
            rollout_fn=rollout_postprocess,
            config=InferenceConfig(
                model_name_or_path=cfg.ocr.model_name,
                default_generation_params={"temperature": cfg.ocr.temperature},
                max_concurrent_generations=cfg.ocr.max_concurrent,
                server_type="endpoint",
                api_key=cfg.ocr.server_api_key,
                metric_interval=100,
                endpoint_url=cfg.ocr.server_url,
            ),
            output_writer=JsonlWriter(
                output_folder=output_folder,
                output_filename=cfg.output.output_filename,
            ),
            shared_context={
                "model_name_or_path": cfg.ocr.model_name,
                "max_tokens": cfg.ocr.max_tokens,
            },
            checkpoints_local_dir=cfg.ocr.checkpoints_local_dir,
        ),
    ]


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="tewe",
)
def main(cfg: DictConfig) -> int:
    logger.info("=" * 80)
    logger.info("Iwe-Pipeline: Local Test with Real Components")
    logger.info("=" * 80)

    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    logger.info(f"Output: {cfg.output.output_dir}")
    logger.info(f"Tasks: {cfg.executor.tasks}, Workers: {cfg.executor.workers}")

    reader = build_reader(cfg)
    pipeline_blocks = build_pipeline(cfg)
    pipeline = [reader] + pipeline_blocks

    hydra_run_dir = HydraConfig.get().run.dir

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = f"{hydra_run_dir}/{cfg.job_name}_run_{run_id}"

    logger.info(f"Run dir: {run_dir}")

    try:
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=cfg.executor.tasks,
            workers=cfg.executor.workers,
            logging_dir=run_dir,
        )

        executor.run()

        logger.info("=" * 80)
        logger.info("✓ Pipeline completed successfully!")

        logger.info(f"✓ Output: {cfg.output.output_dir}/output_*.jsonl.gz")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
