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

import docling_ibm_models.layoutmodel.layout_predictor as _lp_module
import hydra
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import DataFolder, get_datafolder
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.inference.run_inference import (
    InferenceConfig,
    InferenceRunner,
)
from datatrove.pipeline.media.media_readers.zstd import ZstdReader
from datatrove.pipeline.media.media_writers.zstd import ZstdWriter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from fsspec import AbstractFileSystem
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from itewe.classifier.ocr.ocr_classifier import OCRClassifier
from itewe.dedup import AzureContentMD5DedupFilter
from itewe.onnx_layout_predictor import LayoutPredictor
from itewe.readers.pdf import PDFReader
from itewe.utils.rollout_utils import rollout_postprocess

# Patch LayoutPredictor to use ONNXRuntime instead of OpenVino
# This circumvents the issue where we cannot run on aarch64
# because the layout model requires openvino==2025.3.0
# but that version does not support GridSample op on aarch64
# Patch must be done before docling modules are imported
if os.getenv("USE_ONNX_LAYOUT_PREDICTOR", "").lower() in ("1", "true", "yes"):
    _lp_module.LayoutPredictor = LayoutPredictor

from itewe.docling_extractor import DoclingExtractor  # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = {}


def register_pipeline(name: str, depends_on: list[str] | None = None):
    def decorator(fn):
        PIPELINE_REGISTRY[name] = {
            "fn": fn,
            "depends_on": depends_on or [],
        }
        return fn

    return decorator


def build_pipeline(name: str, cfg: DictConfig) -> LocalPipelineExecutor:
    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unknown pipeline: '{name}'. Available: {list(PIPELINE_REGISTRY)}")

    entry = PIPELINE_REGISTRY[name]
    return entry["fn"](cfg)


# --------------------------------------------
def _get_datafolder(output_path: str, fs: AbstractFileSystem | None = None) -> DataFolder:
    if fs is None:
        return get_datafolder(output_path)
    return get_datafolder((output_path, fs))


def init_datafolder(
    folder: str, fs: str, fs_args: dict, container_path: str | None = None
) -> tuple[callable, str]:
    _fs = None

    if fs == "azure":
        from adlfs import AzureBlobFileSystem

        _fs = AzureBlobFileSystem(**fs_args)
        if container_path:
            folder = os.path.join(container_path, folder)
    elif fs != "local":
        raise ValueError(f"Unknown reader backend: {fs}")

    partial_datafolder = partial(_get_datafolder, fs=_fs)
    return partial_datafolder, folder


def _filter_ocr(x: Document) -> bool:
    meta = x.media[0].metadata or {}
    # See the training notebook why we decided for this threshold
    # TODO: @theyorubayesian Maybe investigate this?
    requires_ocr = meta.get("ocr_prob", 0) >= 0.2 or meta.get("garbled_text_ratio", 0) > 0.0
    return not requires_ocr


# --------------------------------------------
def build_pdf_reader(cfg: DictConfig) -> PDFReader:
    reader = partial(
        PDFReader,
        pdf_to_ppm=cfg.reader.pdf_to_ppm,
        yield_pages_as_documents=cfg.reader.yield_pages_as_documents,
        glob_pattern=cfg.reader.glob_pattern,
        recursive=cfg.reader.recursive,
        limit=cfg.reader.limit,
    )

    partial_datafolder, input_dir = init_datafolder(
        cfg.reader.input_dir,
        cfg.reader.backend,
        OmegaConf.to_container(cfg.azure, resolve=True),
        cfg.reader.azure.container_path,
    )

    return reader(data_folder=partial_datafolder(input_dir))


@register_pipeline("dedup_ocr_classifier")
def run_dedup_ocr_classifier_pipeline(cfg: DictConfig) -> LocalPipelineExecutor:
    partial_datafolder, output_folder = init_datafolder(
        cfg.output.output_dir,
        cfg.output.backend,
        OmegaConf.to_container(cfg.azure, resolve=True),
        cfg.output.azure.container_path,
    )

    pipeline = [
        build_pdf_reader(cfg),
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

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=cfg.executor.tasks,
        workers=cfg.executor.workers,
        logging_dir=os.path.join(HydraConfig.get().run.dir, "dedup_ocr_classifier"),
        local_tasks=cfg.executor.local_tasks,
        local_rank_offset=cfg.executor.local_rank_offset,
    )

    return executor


@register_pipeline("nocr_extraction")
def run_nocr_extraction_pipeline(cfg: DictConfig) -> LocalPipelineExecutor:
    partial_datafolder, output_folder = init_datafolder(
        cfg.output.output_dir,
        cfg.output.backend,
        OmegaConf.to_container(cfg.azure, resolve=True),
        cfg.output.azure.container_path,
    )

    pipeline_docling = [
        JsonlReader(
            data_folder=partial_datafolder(
                output_path=os.path.join(output_folder, "nocr_required")
            ),
            glob_pattern="**/*.jsonl.gz",
            doc_progress=True,
        ),
        ZstdReader(
            data_folder=partial_datafolder(output_path=os.path.join(output_folder, "pdfs")),
            workers=4,
            preserve_order=True,
        ),
        DoclingExtractor(
            timeout=10 * 60,
            exclusion_writer=JsonlWriter(
                output_folder=os.path.join(output_folder, "nocr_extraction_failed"),
            ),
        ),
        JsonlWriter(output_folder=os.path.join(output_folder, "nocr_extracted")),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline_docling,
        tasks=cfg.executor.tasks,
        workers=cfg.executor.workers,
        logging_dir=os.path.join(HydraConfig.get().run.dir, "nocr_extraction"),
        local_tasks=cfg.executor.local_tasks,
        local_rank_offset=cfg.executor.local_rank_offset,
    )

    return executor


@register_pipeline("ocr_extraction")
def run_ocr_extraction_pipeline(cfg: DictConfig) -> LocalPipelineExecutor:
    output_folder = cfg.output.output_dir
    fs = None

    if cfg.output.backend == "azure":
        from adlfs import AzureBlobFileSystem

        storage_options = OmegaConf.to_container(cfg.azure, resolve=True)
        fs = AzureBlobFileSystem(**storage_options)
        output_folder = os.path.join(cfg.output.azure.container_path, output_folder)

    partial_datafolder = partial(_get_datafolder, fs=fs)

    ocr_pipeline = [
        JsonlReader(
            data_folder=partial_datafolder(output_path=os.path.join(output_folder, "ocr_required")),
            glob_pattern="**/*.jsonl.gz",
            doc_progress=True,
        ),
        ZstdReader(
            data_folder=partial_datafolder(output_path=os.path.join(output_folder, "pdfs")),
            workers=4,
            preserve_order=True,
        ),
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
            output_writer=JsonlWriter(output_folder=os.path.join(output_folder, "ocr_extracted"))
        ),
    ]

    executor = LocalPipelineExecutor(
        pipeline=ocr_pipeline,
        tasks=cfg.executor.tasks,
        workers=cfg.executor.workers,
        logging_dir=os.path.join(HydraConfig.get().run.dir, "ocr_extraction"),
        local_tasks=cfg.executor.local_tasks,
        local_rank_offset=cfg.executor.local_rank_offset,
    )

    return executor


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

    for step in cfg.pipeline_steps:
        executor = build_pipeline(step, cfg)
        executor.run()


if __name__ == "__main__":
    raise SystemExit(main())
