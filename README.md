# Ìtèwé

DataTrove-native OCR pipeline for processing PDF documents with a server-backed vision model.

## Overview

This repository focuses on a working local OCR pipeline and a monitoring UI, with scaffolding for Azure ingestion and postprocessing stages.

**Working pieces:**

- PDF reader
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
  python tewe.py reader.backend=azure \
    reader.azure.container_path=az://mycontainer \
    reader.input_dir="mypdfdir" \
    ocr.server_url=http://127.0.0.1:8080 \
    ocr.model_name=mradermacher/KarantaOCR-GGUF \
    output.output_dir=outputs
  ```

### Sample Config

A ready-to-edit sample lives here:

- `configs/tewe.yaml`

Required keys for `tewe.py`:

- `limit`
- `monitor`
- `job_name`
- `ocr.server_url`
- `ocr.model_name`
- `output.output_dir`
- `output.output_filename`
- `reader.input_dir`

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

- `itewe/monitoring/ui/index.html`
- `itewe/monitoring/ui/record.html`

## Project Structure

```yaml
configs/
  run_iwe_sample.yaml
  local.yaml
  hf_dataset.yaml
  stages/
    fetch_ocr.yaml
    postprocess_quality.yaml

itewe/
  blocks/
    assemble/
      group_pages.py           # placeholder
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
    pdf.py                     # active
  ids.py                       # active (ID helpers)
  utils.py                     # active (PDF + rollout helpers)

tewe.py                        # config-driven local runner
scripts/                       # stage scripts (scaffolded)
```

## Notable Modules

- `itewe/utils.py`:
  - PDF rendering + request building for OCR.
  - `rollout_postprocess` creates OpenAI-style multimodal payloads.

- `itewe/monitoring/tracker.py`:
  - Progress tracking + web UI with per-page request/output inspection.

- `itewe/ids.py`:
  - Stable doc/page ID helpers.

## Tests

```bash
pytest
```

## License

TBD
