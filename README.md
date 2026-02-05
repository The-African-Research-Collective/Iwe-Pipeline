# Iwe-Pipeline

DataTrove-native OCR pipeline for processing documents using consumer hardware.

## Overview

Iwe-Pipeline is a modular, scalable OCR processing pipeline built on [DataTrove](https://github.com/huggingface/datatrove). It fetches PDFs from Azure Blob Storage, extracts text using vision-language models (KarantaOCR), performs intelligent postprocessing, scores quality, and publishes to HuggingFace datasets.

### Key Features

- **DataTrove-Native**: Built as composable pipeline blocks
- **Multi-Backend OCR**: Supports vLLM, Exo, Ollama, and MLX
- **Manifest-First**: Reproducible, shardable processing
- **Consumer Hardware**: Optimized for GPUs and Apple Silicon
- **Quality-Focused**: BERT-based quality scoring and filtering
- **Production-Ready**: Caching, retry logic, monitoring

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Iwe-Pipeline

# Install with uv
uv pip install -e .

# Install inference backend (choose one)
uv pip install -e ".[vllm]"    # GPU (NVIDIA)
uv pip install -e ".[mlx]"     # Apple Silicon
uv pip install -e ".[ollama]"  # Local
uv pip install -e ".[exo]"     # Distributed
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
export AZURE_CONNECTION_STRING="..."
export HF_TOKEN="hf_..."
```

### Running the Pipeline

#### Step 1: Create Manifest

```bash
python scripts/make_manifest.py \
    --container mycontainer \
    --prefix pdfs/ \
    --output manifests/azure_blobs.jsonl
```

#### Step 2: Fetch + OCR

```bash
python scripts/run_fetch_ocr.py \
    --config configs/stages/fetch_ocr.yaml \
    --start-server
```

#### Step 3: Postprocess + Quality

```bash
python scripts/run_postprocess.py \
    --config configs/stages/postprocess_quality.yaml
```

#### Step 4: Publish to Hub

```bash
python scripts/run_publish.py \
    --config configs/hf_dataset.yaml \
    --data-dir data/final
```

## Architecture

### Pipeline Flow

```
Azure Blob Storage
    ↓
Manifest (JSONL)
    ↓
[Pipeline 1: Fetch + OCR]
    ├─ AzureManifestReader
    ├─ AzureFetchPDF → ./data/fetched/
    ├─ (Optional) SplitPages
    ├─ KarantaVLLMOCR
    └─ OCRIndexWriter → ./data/ocr_index/
    ↓
[Pipeline 2: Postprocess + Quality]
    ├─ JsonlReader
    ├─ (Optional) GroupPages
    ├─ LanguageTag
    ├─ Normalize
    ├─ BoilerplateRemover
    ├─ TableCleaner
    ├─ BertQualityScore
    └─ FinalWriter → ./data/final/
    ↓
[Pipeline 3: Publish]
    └─ HuggingFace Hub
```

### Project Structure

```
iwe-pipeline/
├── configs/                     # YAML configurations
│   ├── local.yaml              # Machine settings
│   ├── stages/                 # Pipeline configs
│   │   ├── fetch_ocr.yaml
│   │   └── postprocess_quality.yaml
│   └── hf_dataset.yaml         # Hub dataset config
│
├── scripts/                     # Pipeline entrypoints
│   ├── make_manifest.py        # Create blob manifest
│   ├── run_fetch_ocr.py        # Fetch + OCR pipeline
│   ├── run_postprocess.py      # Postprocess + quality
│   └── run_publish.py          # Publish to hub
│
├── iwe_pipeline/                # Core library
│   ├── datamodel/              # Schemas and conventions
│   │   ├── doc_schema.py       # Metadata field definitions
│   │   └── ids.py              # ID generation utilities
│   │
│   ├── readers/                # DataTrove readers
│   │   ├── azure_manifest_reader.py
│   │   └── azure_blob.py
│   │
│   ├── blocks/                 # DataTrove pipeline blocks
│   │   ├── fetch/
│   │   │   └── azure_fetch_pdf.py
│   │   ├── ocr/
│   │   │   ├── split_pages.py
│   │   │   └── karanta_vllm_ocr.py
│   │   ├── postprocess/
│   │   │   ├── language_tag.py
│   │   │   ├── normalize.py
│   │   │   ├── boilerplate.py
│   │   │   └── tables.py
│   │   ├── quality/
│   │   │   └── bert_score.py
│   │   └── assemble/
│   │       └── group_pages.py
│   │
│   ├── writers/                # DataTrove writers
│   │   ├── ocr_index_writer.py
│   │   └── final_writer.py
│   │
│   ├── server/                 # Inference server managers
│   │   ├── manager.py
│   │   ├── vllm_server.py
│   │   └── exo_server.py
│   │
│   ├── monitoring/             # Metrics & tracking
│   │   └── tracker.py
│   │
│   └── utils/                  # Shared utilities
│       ├── io.py
│       ├── retry.py
│       └── text.py
│
├── tests/                      # Test suite
│   ├── test_blocks_fetch.py
│   ├── test_blocks_ocr.py
│   ├── test_blocks_postprocess.py
│   ├── test_blocks_quality.py
│   └── test_pipelines_smoke.py
│
└── data/                       # Data directories (gitignored)
    ├── fetched/
    ├── ocr_extracted/
    ├── ocr_index/             # Intermediate contract
    ├── postprocessed/
    ├── quality_scored/
    └── final/
```

## DataTrove Integration

All pipeline components extend DataTrove base classes:

- **Readers**: `BaseReader` - Yield `Document` objects from sources
- **Blocks**: `PipelineStep` - Transform/filter/enrich documents
- **Writers**: `JsonlWriter`, `ParquetWriter` - Persist results
- **Executor**: `LocalPipelineExecutor` - Run with parallelism

### Document Schema

Documents flow through the pipeline as DataTrove `Document` objects:

```python
Document(
    id="abc123",                 # Unique identifier
    text="extracted text...",    # Main content
    metadata={                   # Structured metadata
        "source": {...},
        "ocr": {...},
        "language": "eng_Latn",
        "quality_score": 0.85,
    },
    media=[...]                  # PDF bytes, images, etc.
)
```

## Pipeline Blocks

### Fetch

- **AzureFetchPDF**: Downloads PDFs from Azure Blob Storage
  - Caches based on etag
  - Adds local file path to metadata
  - Retry logic for transient failures

### OCR

- **SplitPages**: Converts PDF into page-level documents (optional)
- **KarantaVLLMOCR**: Extracts text using KarantaOCR via vLLM
  - Supports document-level and page-level modes
  - Concurrent request handling
  - Timeout management

### Postprocess

- **LanguageTag**: Identifies document language (GlotLID/fastText)
- **Normalize**: Unicode fixes, whitespace cleanup, ligature expansion
- **BoilerplateRemover**: Removes headers, footers, page numbers
- **TableCleaner**: Detects and formats tables

### Quality

- **BertQualityScore**: Scores document quality using BERT classifiers
  - Chunks documents for processing
  - Aggregates scores
  - Optional low-quality filtering

### Assemble

- **GroupPages**: Reassembles pages back into documents

## Configuration

### Local Settings (`configs/local.yaml`)

Machine-specific settings: paths, workers, Azure credentials, etc.

### Pipeline Configs (`configs/stages/`)

Pipeline-specific settings:
- Input/output paths
- Block parameters
- Executor configuration

### Dataset Config (`configs/hf_dataset.yaml`)

HuggingFace Hub dataset metadata and schema.

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=iwe_pipeline

# Specific module
pytest tests/test_blocks_ocr.py -v
```

### Adding New Blocks

1. Create file in `iwe_pipeline/blocks/<category>/`
2. Extend `PipelineStep`
3. Implement `run()` method
4. Add tests
5. Update pipeline configs

Example:

```python
from datatrove.pipeline.base import PipelineStep
from datatrove.data import Document

class MyBlock(PipelineStep):
    name = "✨ My Block"
    type = "🔨 PROCESSOR"
    
    def run(self, data: Document, rank=0, world_size=1):
        # Transform document
        data.text = self.process(data.text)
        yield data
```

### Code Style

```bash
# Lint
ruff check .

# Format
ruff format .

# Fix
ruff check --fix .
```

## Deployment

### Single Machine

Use `LocalPipelineExecutor` with multiple tasks/workers:

```python
executor = LocalPipelineExecutor(
    pipeline=pipeline,
    tasks=8,      # Number of parallel tasks
    workers=4,    # Workers per task
)
```

### Distributed (Future)

DataTrove supports Ray and Slurm executors for cluster deployment.

## Monitoring

Track pipeline progress:

- DataTrove's built-in `doc_progress`
- Custom `MetricsTracker` for OCR-specific metrics
- Logs in `./logs/`

## Troubleshooting

### OCR Server Not Responding

```bash
# Check server health
curl http://localhost:8000/health

# Restart server
python -c "from iwe_pipeline.server.manager import ServerManager; \
           s = ServerManager(); s.start()"
```

### Out of Memory

- Reduce `batch_size` in OCR config
- Reduce `tasks` or `workers` in executor
- Enable page-level processing (`split_pages: true`)

### Failed Downloads

- Check Azure credentials
- Increase `max_retries` in fetch config
- Check blob permissions

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

## Acknowledgments

- [DataTrove](https://github.com/huggingface/datatrove) - Pipeline framework
- [FinePDFs](https://github.com/huggingface/finepdfs) - Reference implementation
- [KarantaOCR](https://huggingface.co/taresco/KarantaOCR) - OCR model

---

Built with ❤️ for the OCR community
