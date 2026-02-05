# Local Testing Guide

Test the Iwe-Pipeline using **actual components** from `iwe_pipeline` on local PDFs.

## Quick Start

### 1. Setup Test Directory

```bash
# Create directory and add PDFs
mkdir -p test_pdfs
cp /path/to/your/*.pdf test_pdfs/
```

### 2. Test Without OCR Server (Minimal Mode)

```bash
# Tests postprocessing blocks only (no OCR)
python test_local.py --input-dir test_pdfs --mode minimal
```

### 3. Test With OCR Server (Full Mode)

```bash
# First, start your OCR server
# (See server setup instructions below)

# Then run full pipeline
python test_local.py \
    --input-dir test_pdfs \
    --mode full \
    --server-url http://localhost:8000
```

## Pipeline Modes

### Minimal Mode (No OCR Server Needed)

Tests postprocessing and quality blocks:

```bash
python test_local.py --input-dir test_pdfs --mode minimal
```

**Blocks tested:**
- ✅ LanguageTag
- ✅ Normalize
- ✅ BoilerplateRemover
- ✅ TableCleaner
- ✅ BertQualityScore
- ✅ JsonlWriter

**Note:** This mode expects documents to already have text. Useful for testing postprocessing logic.

### Full Mode (Document-Level OCR)

Full pipeline with document-level OCR:

```bash
python test_local.py \
    --input-dir test_pdfs \
    --mode full \
    --server-url http://localhost:8000
```

**Blocks tested:**
- ✅ KarantaVLLMOCR (document mode)
- ✅ All postprocessing blocks
- ✅ Quality scoring
- ✅ Writer

### Page-Level Mode

Page-level OCR with split/group:

```bash
python test_local.py \
    --input-dir test_pdfs \
    --mode page-level \
    --server-url http://localhost:8000
```

**Blocks tested:**
- ✅ SplitPages
- ✅ KarantaVLLMOCR (page mode)
- ✅ GroupPages
- ✅ All postprocessing blocks
- ✅ Quality scoring
- ✅ Writer

## Components Used

This test uses **real components** from `iwe_pipeline`:

```python
# From iwe_pipeline/blocks/
from iwe_pipeline.blocks.ocr.split_pages import SplitPages
from iwe_pipeline.blocks.ocr.karanta_vllm_ocr import KarantaVLLMOCR
from iwe_pipeline.blocks.postprocess.language_tag import LanguageTag
from iwe_pipeline.blocks.postprocess.normalize import Normalize
from iwe_pipeline.blocks.postprocess.boilerplate import BoilerplateRemover
from iwe_pipeline.blocks.postprocess.tables import TableCleaner
from iwe_pipeline.blocks.quality.bert_score import BertQualityScore
from iwe_pipeline.blocks.assemble.group_pages import GroupPages

# From iwe_pipeline/datamodel/
from iwe_pipeline.datamodel.ids import generate_doc_id

# DataTrove components
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.writers.jsonl import JsonlWriter
```

## Advanced Options

### Limit Number of PDFs

```bash
python test_local.py --input-dir test_pdfs --mode minimal --limit 5
```

### Parallel Processing

```bash
python test_local.py \
    --input-dir test_pdfs \
    --mode full \
    --tasks 4 \
    --workers 2
```

### Custom Output Directory

```bash
python test_local.py \
    --input-dir test_pdfs \
    --output-dir ./my_results \
    --mode minimal
```

## Check Results

```bash
# List output files
ls -lh test_output/

# View output (decompress gzip first)
gunzip -c test_output/output_0000.jsonl.gz | jq .

# Count documents processed
gunzip -c test_output/output_*.jsonl.gz | wc -l
```

## Setting Up OCR Server

For `full` or `page-level` modes, you need an inference server.

### Option 1: vLLM Server

```bash
# Install vLLM
pip install vllm

# Start server
vllm serve nvidia/KarantaOCR --port 8000
```

### Option 2: Using Server Manager

```python
from iwe_pipeline.server.manager import ServerManager

# Start server
server = ServerManager(
    server_type="vllm",
    model_name_or_path="nvidia/KarantaOCR",
    port=8000
)
server.start()

# Test endpoint
curl http://localhost:8000/health
```

## What This Tests

✅ **Real Components** - Uses actual blocks from iwe_pipeline
✅ **DataTrove Integration** - Uses LocalPipelineExecutor
✅ **Document Flow** - Tests Document passing between blocks
✅ **Metadata Schema** - Validates real metadata structure
✅ **Pipeline Composition** - Tests block composition
✅ **File I/O** - Real reading and writing
✅ **Error Handling** - Real error handling logic

## Troubleshooting

### No PDFs Found

```bash
# Add PDFs to directory
cp /path/to/pdfs/*.pdf test_pdfs/
ls test_pdfs/
```

### Import Errors

```bash
# Install package in editable mode
pip install -e .

# Or with uv
uv pip install -e .
```

### OCR Server Not Responding

```bash
# Check server is running
curl http://localhost:8000/health

# Check logs
tail -f logs/test_local/*/worker_*.log
```

### Module Not Found

```bash
# Ensure you're in the project root
cd /path/to/Iwe-Pipeline

# Install dependencies
uv pip install -e ".[dev]"
```

## Example Output

```bash
$ python test_local.py --input-dir test_pdfs --mode minimal

2026-02-04 22:00:00 - __main__ - INFO - ================================
2026-02-04 22:00:00 - __main__ - INFO - Iwe-Pipeline: Local Test with Real Components
2026-02-04 22:00:00 - __main__ - INFO - ================================
2026-02-04 22:00:00 - __main__ - INFO - Mode: minimal
2026-02-04 22:00:00 - __main__ - INFO - Input: test_pdfs (3 PDFs)
2026-02-04 22:00:00 - __main__ - INFO - Output: ./test_output
2026-02-04 22:00:00 - __main__ - INFO - Tasks: 1, Workers: 1
2026-02-04 22:00:00 - __main__ - INFO - Building minimal pipeline (no OCR)
...
2026-02-04 22:00:10 - __main__ - INFO - ✓ Pipeline completed successfully!
2026-02-04 22:00:10 - __main__ - INFO - ✓ Output: ./test_output/output_*.jsonl.gz
```

## Next Steps

1. **Implement Block Logic** - Add real implementation to skeleton blocks
2. **Add Tests** - Create unit tests in `tests/`
3. **Setup Azure** - Use `scripts/run_fetch_ocr.py` for Azure integration
4. **Scale Up** - Increase tasks/workers for parallel processing
5. **Production** - Deploy with proper configs in `configs/`

---

This test validates your pipeline structure with **real components**! 🚀
