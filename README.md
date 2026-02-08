# Iwe-Pipeline

DataTrove-native OCR pipeline for processing PDF documents with a server-backed vision model.

## Overview

This repository focuses on a working local OCR pipeline and a monitoring UI, with scaffolding for Azure ingestion and postprocessing stages.

**Working pieces:**
- Local PDF reader + page splitting (`SplitPages`) producing one document per page
- OCR inference via DataTrove `InferenceRunner` (server-backed endpoint)
- JSONL output writer
- Progress + record monitor UI (`OCRInferenceProgressMonitor`)

**Scaffolded / placeholder (not implemented yet):**
- Azure ingestion (`AzureManifestReader`, `AzureFetchPDF`)
- Page regrouping (`GroupPages`)
- Postprocess blocks (`LanguageTag`, `BoilerplateRemover`)
- Quality scoring (`BertQualityScore`)
- Stage scripts in `scripts/` (config loading and pipeline builders are TODO)

## Quick Start (Local OCR)

1. Install dependencies:

```bash
uv pip install -e .
```

2. Start your OCR server (OpenAI-compatible chat/completions endpoint).

3. Run the local pipeline with a config file:

```bash
python run_iwe_pipeline.py --config configs/run_iwe_sample.yaml
```

### Sample Config

A ready-to-edit sample lives here:
- `configs/run_iwe_sample.yaml`

Required keys for `run_iwe_pipeline.py`:
- `input_dir` (or `data.fetched`)
- `output_dir` (or `data.ocr_extracted`)
- `limit`
- `monitor`
- `job_name`
- `ocr.server_url`
- `ocr.model_name`
- `ocr.temperature`
- `ocr.max_concurrent`
- `ocr.max_tokens`
- `output_filename`
- `executor.tasks`
- `executor.workers`

## Monitoring UI

Enable monitoring in the config:

```yaml
monitor: true
```

The monitor runs by default on:
- `http://127.0.0.1:8040`

It exposes:
- `/` list view
- `/record?record_id=...` detail view

The UI is served from:
- `iwe_pipeline/monitoring/ui/index.html`
- `iwe_pipeline/monitoring/ui/record.html`

## Project Structure

```
configs/
  run_iwe_sample.yaml
  local.yaml
  hf_dataset.yaml
  stages/
    fetch_ocr.yaml
    postprocess_quality.yaml

iwe_pipeline/
  blocks/
    assemble/
      group_pages.py           # placeholder
    fetch/
      azure_fetch_pdf.py       # placeholder
    ocr/
      split_pages.py           # active
    postprocess/
      boilerplate.py           # placeholder
      language_tag.py          # placeholder
    quality/
      bert_score.py            # placeholder
  monitoring/
    tracker.py                 # active (UI + progress API)
    ui/
      index.html
      record.html
  readers/
    azure_manifest_reader.py   # placeholder
  ids.py                       # active (ID helpers)
  utils.py                     # active (PDF + rollout helpers)

run_iwe_pipeline.py            # config-driven local runner
scripts/                       # stage scripts (scaffolded)
```

## Notable Modules

- `iwe_pipeline/blocks/ocr/split_pages.py`:
  - Splits a PDF into one document per page and appends page number to `id`.
  - Skips already-processed page IDs by scanning outputs.

- `iwe_pipeline/utils.py`:
  - PDF rendering + request building for OCR.
  - `rollout_postprocess` creates OpenAI-style multimodal payloads.

- `iwe_pipeline/monitoring/tracker.py`:
  - Progress tracking + web UI with per-page request/output inspection.

- `iwe_pipeline/ids.py`:
  - Stable doc/page ID helpers.

## Tests

```bash
pytest
```

## License

TBD
