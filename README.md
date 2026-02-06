# Iwe-Pipeline

DataTrove-native OCR pipeline for processing documents with local or server-backed vision models.

## Overview

Iwe-Pipeline is a modular OCR pipeline built around DataTrove pipelines and inference runners. The current codebase focuses on:

- Azure manifest-driven ingestion
- Fetching PDFs from Azure Blob Storage
- Optional page splitting for page-level OCR
- OCR via a server-backed inference endpoint (vLLM-compatible)
- Postprocessing, table cleanup, and quality scoring
- Writing intermediate OCR index and final datasets

Some components are scaffolding and are marked as TODO in code (e.g., pipeline builders and server lifecycle management). The local test runner demonstrates the current end-to-end flow against a running inference server.

## Quick Start (Local Test)

### Install

```bash
uv pip install -e .
```

### Run a local test

1. Start your OCR inference server (OpenAI-compatible chat/completions endpoint).
2. Run the local pipeline on PDFs.

```bash
python test_local.py --input-dir ./test_pdfs --output-dir ./test_output --server-url http://localhost:8000
```

This runs:

- `LocalPDFReader`
- `SplitPages`
- `InferenceRunner` with `rollout_postprocess`
- `JsonlWriter`

Outputs are written to `test_output/`.

## Pipeline Flow (Current)

Local test pipeline:

```
Local PDFs
  -> SplitPages (PDF -> per-page images)
  -> InferenceRunner (server-backed OCR)
  -> JsonlWriter (OCR output)
```

Configured multi-stage pipeline (scaffolding in `scripts/`):

```
Azure Manifest
  -> AzureFetchPDF
  -> (optional) SplitPages
  -> InferenceRunner (server-backed OCR)
  -> OCRIndexWriter
  -> Postprocess + Quality blocks
  -> FinalWriter
```

## Project Structure

```
iwe-pipeline/
├── configs/
│   ├── local.yaml
│   ├── stages/
│   │   ├── fetch_ocr.yaml
│   │   └── postprocess_quality.yaml
│   └── hf_dataset.yaml
│
├── scripts/
│   ├── make_manifest.py
│   ├── run_fetch_ocr.py
│   ├── run_postprocess.py
│   └── run_publish.py
│
├── iwe_pipeline/
│   ├── blocks/
│   │   ├── assemble/
│   │   │   └── group_pages.py
│   │   ├── fetch/
│   │   │   └── azure_fetch_pdf.py
│   │   ├── ocr/
│   │   │   └── split_pages.py
│   │   ├── postprocess/
│   │   │   ├── boilerplate.py
│   │   │   ├── language_tag.py
│   │   │   ├── normalize.py
│   │   │   └── tables.py
│   │   └── quality/
│   │       └── bert_score.py
│   │
│   ├── datamodel/
│   │   ├── doc_schema.py
│   │   └── ids.py
│   │
│   ├── monitoring/
│   │   └── tracker.py
│   │
│   ├── readers/
│   │   └── azure_manifest_reader.py
│   │
│   ├── server/
│   │   └── manager.py
│   │
│   ├── utils/
│   │   ├── io.py
│   │   ├── retry.py
│   │   ├── text.py
│   │   └── utils.py
│   │
│   └── writers/
│       ├── final_writer.py
│       └── ocr_index_writer.py
│
├── test_local.py
├── tests/
└── data/ (gitignored)
```

## Notable Components

Readers:

- `AzureManifestReader` reads JSONL/CSV manifests for reproducible ingestion.

Blocks:

- `AzureFetchPDF` downloads PDFs and stores local paths.
- `SplitPages` converts a PDF into page-level images for OCR.
- Postprocess blocks handle language tagging, normalization, boilerplate removal, and table cleanup.
- `BertQualityScore` computes a quality score.
- `GroupPages` merges page-level outputs into document-level output.

Writers:

- `OCRIndexWriter` writes intermediate OCR results.
- `FinalWriter` writes final datasets (Parquet writer wrapper).

Inference:

- `rollout_postprocess` (in `iwe_pipeline/utils/utils.py`) builds OpenAI-compatible multimodal messages for inference.
- `InferenceRunner` (DataTrove) sends requests to the configured endpoint.

## Scripts and Status

The scripts in `scripts/` define the intended multi-stage pipeline, but some functions are currently placeholders:

- `load_config` and `build_pipeline` in `run_fetch_ocr.py` and `run_postprocess.py` are TODO.
- `ServerManager` in `iwe_pipeline/server/manager.py` is a stub.

Use `test_local.py` as the working reference for an end-to-end run.

## Environment

Copy `.env.example` and set credentials as needed:

```bash
cp .env.example .env
```

## Tests

```bash
pytest
```

## License

TBD
