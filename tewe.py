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
from functools import partial

import hydra
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import DataFolder, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.inference.run_inference import (
    InferenceConfig,
    InferenceRunner,
)
from datatrove.pipeline.media.media_writers.zstd import ZstdWriter
from datatrove.pipeline.writers import JsonlWriter
from fsspec import AbstractFileSystem
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from itewe.classifier.ocr.ocr_classifier import OCRClassifier
from itewe.dedup import AzureContentMD5DedupFilter
from itewe.docling_extractor import DoclingExtractor
from itewe.readers.pdf import PDFReader
from itewe.utils.rollout_utils import rollout_postprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------
def _get_datafolder(output_path: str, fs: AbstractFileSystem | None = None) -> DataFolder:
    if fs is None:
        return get_datafolder(output_path)
    return get_datafolder((output_path, fs))


def _filter_ocr(x: Document) -> bool:
    meta = x.media[0].metadata or {}
    # See the training notebook why we decided for this threshold
    # TODO: @theyorubayesian Maybe investigate this?
    requires_ocr = meta.get("ocr_prob", 0) >= 0.2 or meta.get("garbled_text_ratio", 0) > 0.0
    return not requires_ocr
# --------------------------------------------


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

        storage_options = OmegaConf.to_container(cfg.azure, resolve=True)
        fs = AzureBlobFileSystem(**storage_options)
        data_folder = ("/".join([cfg.reader.azure.container_path, cfg.reader.input_dir]), fs)

        if not fs.exists(data_folder[0]):
            raise ValueError(f"Input directory does not exist: {data_folder[0]}")

        return reader(data_folder=data_folder)

    else:
        raise ValueError(f"Unknown reader backend: {cfg.reader.backend}")


def build_dedup_ocr_pipeline(cfg: DictConfig) -> list[PipelineStep]:
    """Build full pipeline including OCR (requires inference server)."""
    output_folder = cfg.output.output_dir
    fs = None

    if cfg.output.backend == "azure":
        from adlfs import AzureBlobFileSystem

        storage_options = OmegaConf.to_container(cfg.azure, resolve=True)
        fs = AzureBlobFileSystem(**storage_options)
        output_folder = os.path.join(cfg.output.azure.container_path, output_folder)

    partial_datafolder = partial(_get_datafolder, fs=fs)

    return [
        AzureContentMD5DedupFilter(
            exclusion_writer=JsonlWriter(
                output_folder=partial_datafolder(
                    output_path=os.path.join(output_folder, "content_md5_dedup", "removed")
                )
            )
        ),
        ZstdWriter(
            max_file_size=5 * 1024 * 1024 * 1024,
            output_folder=partial_datafolder(output_path=os.path.join(output_folder, "pdfs")),
            output_filename="pdfs_${rank}.zstd",
        ),
        OCRClassifier(
            path_to_model=cfg.classifiers.ocr_classifier_model_path,
            exclusion_writer=JsonlWriter(
                output_folder=partial_datafolder(
                    output_path=os.path.join(output_folder, "failed_ocr_pred")
                ),
            ),
            exclude_failed=True,
        ),
        LambdaFilter(
            _filter_ocr,
            exclusion_writer=JsonlWriter(
                output_folder=partial_datafolder(
                    output_path=os.path.join(output_folder, "ocr_required")
                ),
            ),
        ),
        JsonlWriter(
            output_folder=partial_datafolder(
                output_path=os.path.join(output_folder, "nocr_required")
            )
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
    pipeline_blocks = build_dedup_ocr_pipeline(cfg)
    pipeline = [reader] + pipeline_blocks

    run_dir = HydraConfig.get().run.dir

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
