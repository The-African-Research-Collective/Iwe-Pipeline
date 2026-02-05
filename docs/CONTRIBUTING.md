# Contributing to Iwe-Pipeline

Welcome to Iwe-Pipeline! This document provides guidelines and information for contributors who want to help build this OCR processing pipeline.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Adding New Features](#adding-new-features)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Project Overview

Iwe-Pipeline is an OCR (Optical Character Recognition) processing pipeline designed to run on consumer hardware. It processes PDF documents from Azure Blob Storage, extracts text using vision-language models, performs post-processing cleanup, scores document quality, and publishes results to HuggingFace datasets.

### Key Technologies

- **Datatrove**: Framework for data processing pipelines
- **KarantaOCR**: Vision-language model for OCR
- **vLLM/Exo/Ollama**: Inference servers for running models
- **Azure Blob Storage**: Document storage
- **HuggingFace Hub**: Dataset hosting

### Design Philosophy

1. **Modular**: Each stage is independent and can be tested/run separately
2. **Flexible**: Support multiple inference backends (vLLM, Exo, Ollama, MLX)
3. **Observable**: Built-in monitoring and metrics tracking
4. **Scalable**: Designed for distributed processing with datatrove
5. **Consumer-friendly**: Optimized to run on consumer GPUs and Apple Silicon

## Architecture

### Pipeline Flow

```
Azure Blob Storage (PDFs)
    ↓
[Stage 1] Fetch PDFs → ./data/fetched/
    ↓
[Stage 2] OCR Extraction (KarantaOCR) → ./data/ocr_extracted/
    ↓
[Stage 3] Postprocessing
    ├─ Language Tagging
    ├─ Normalization
    ├─ Boilerplate Removal
    └─ Table Cleaning → ./data/postprocessed/
    ↓
[Stage 4] Quality Scoring (BERT) → ./data/quality_scored/
    ↓
[Stage 5] Push to HuggingFace Hub
```

### Datatrove Integration

All pipeline components extend Datatrove base classes:

- **Readers**: `BaseReader` - Fetch data from sources
- **Processors**: `PipelineStep` - Transform documents
- **Writers**: `JsonlWriter`, `HuggingFaceDatasetWriter` - Save outputs
- **Execution**: `LocalPipelineExecutor` - Run pipelines with parallelism

Documents flow through the pipeline as `Document` objects:
```python
Document(
    text="...",           # Extracted text
    id="...",            # Unique identifier
    metadata={...},      # Custom metadata
    media=[...],         # PDF bytes, images, etc.
)
```

## Development Setup

### Prerequisites

- Python 3.12+
- uv package manager
- Git
- (Optional) CUDA-capable GPU or Apple Silicon Mac

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Iwe-Pipeline

# Install base dependencies
uv pip install -e .

# Install dev dependencies
uv pip install -e ".[dev]"

# Install inference backend (choose one or more)
uv pip install -e ".[vllm]"   # For NVIDIA GPUs
uv pip install -e ".[mlx]"    # For Apple Silicon
uv pip install -e ".[ollama]" # For local inference
uv pip install -e ".[exo]"    # For distributed inference
```

### Environment Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
AZURE_CONTAINER_NAME=your-container
AZURE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
HF_TOKEN=hf_...
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=iwe_pipeline --cov-report=html

# Run specific test file
pytest tests/test_ocr.py -v
```

## Project Structure

```
Iwe-Pipeline/
├── run_iwe_pipeline.py          # Main orchestration script
├── pyproject.toml               # Project dependencies & config
├── README.md                    # User documentation
├── CONTRIBUTING.md              # This file
├── .env.example                 # Environment template
│
├── iwe_pipeline/                # Main package
│   ├── readers/                 # Data fetching modules
│   │   └── azure_blob.py       # Azure Blob Storage reader
│   │
│   ├── ocr/                     # OCR extraction
│   │   └── karanta_ocr.py      # KarantaOCR extractor
│   │
│   ├── postprocessing/          # Text cleanup & enhancement
│   │   ├── language.py         # Language identification
│   │   ├── normalize.py        # Text normalization
│   │   ├── boilerplate.py      # Boilerplate removal
│   │   └── tables.py           # Table cleaning
│   │
│   ├── server/                  # Inference server management
│   │   ├── manager.py          # Unified server manager
│   │   ├── vllm_server.py      # vLLM implementation
│   │   └── exo_server.py       # Exo implementation
│   │
│   ├── monitoring/              # Metrics & observability
│   │   └── tracker.py          # MetricsTracker class
│   │
│   └── utils/                   # Helper utilities
│       └── helpers.py          # Pipeline utilities
│
├── tests/                       # Test suite
│   ├── test_pipeline.py        # Integration tests
│   ├── test_ocr.py             # OCR module tests
│   └── test_postprocessing.py  # Postprocessing tests
│
└── data/                        # Data directories (gitignored)
    ├── input/                   # Input PDFs
    └── output/                  # Processed outputs
```

## Pipeline Stages

### Stage 1: Fetch PDFs (`iwe_pipeline/readers/azure_blob.py`)

**Purpose**: Download PDF documents from Azure Blob Storage

**Key Classes**:
- `AzureBlobReader(BaseReader)`: Streams PDFs from Azure

**Implementation Notes**:
- Should handle pagination for large blob lists
- Support blob prefix filtering
- Implement retry logic for failed downloads
- Stream PDFs without loading all into memory
- Return `Document` objects with PDF in `media` field

**Example**:
```python
from azure.storage.blob import BlobServiceClient

def run(self, data=None, rank=0, world_size=1):
    client = BlobServiceClient.from_connection_string(self.connection_string)
    container = client.get_container_client(self.container_name)
    
    for blob in container.list_blobs(name_starts_with=self.blob_prefix):
        blob_client = container.get_blob_client(blob.name)
        pdf_bytes = blob_client.download_blob().readall()
        
        yield Document(
            id=blob.name,
            media=[{"media_bytes": pdf_bytes, "media_type": "application/pdf"}],
            metadata={"source": "azure_blob", "blob_name": blob.name}
        )
```

### Stage 2: OCR Extraction (`iwe_pipeline/ocr/karanta_ocr.py`)

**Purpose**: Extract text from PDF pages using KarantaOCR vision-language model

**Key Classes**:
- `KarantaOCRExtractor(PipelineStep)`: Processes PDFs through OCR

**Implementation Notes**:
- Convert PDF pages to images
- Use `InferenceRunner` from datatrove for model inference
- Support batch processing for efficiency
- Handle multi-page PDFs
- Preserve page structure in output
- Add OCR metadata (confidence scores, detected language, etc.)

**Integration with Servers**:
```python
# Use InferenceRunner with custom rollout function
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceConfig

async def rollout_ocr(document, generate, **kwargs):
    # Convert PDF to images
    pages = convert_pdf_to_images(document.media[0].media_bytes)
    
    # Process each page
    results = []
    for page_img in pages:
        result = await generate({"image": page_img, "prompt": "Extract all text"})
        results.append(result.text)
    
    document.text = "\n\n".join(results)
    return document
```

### Stage 3: Postprocessing

#### Language Tagging (`postprocessing/language.py`)

**Purpose**: Identify document language using GlotLID or similar

**Implementation Notes**:
- Use GlotLID or fastText for language detection
- Add language code to metadata (e.g., "eng_Latn", "fra_Latn")
- Support multi-language documents
- Add confidence scores

#### Normalization (`postprocessing/normalize.py`)

**Purpose**: Clean and standardize text

**Implementation Notes**:
- Fix unicode encoding issues
- Normalize whitespace (spaces, tabs, newlines)
- Remove or fix control characters
- Standardize quotes, dashes, ellipses
- Handle ligatures (fi, fl → f+i, f+l)

#### Boilerplate Removal (`postprocessing/boilerplate.py`)

**Purpose**: Remove repetitive headers, footers, page numbers

**Implementation Notes**:
- Detect repeated patterns across pages
- Identify page numbers and remove them
- Remove common header/footer text
- Preserve document body content
- Use heuristics: position, frequency, length

#### Table Cleaning (`postprocessing/tables.py`)

**Purpose**: Detect and properly format tables

**Implementation Notes**:
- Detect table structures in OCR output
- Convert malformed tables to markdown or CSV format
- Remove tables that are too corrupted
- Preserve table semantics

### Stage 4: Quality Scoring

**Purpose**: Score document quality using BERT classifiers

**Implementation Notes**:
- Chunk documents into BERT-sized segments (512 tokens)
- Run classifier on each chunk
- Aggregate scores (mean, max, etc.)
- Add quality scores to metadata
- Support multiple quality metrics (educational value, DCLM quality, etc.)

### Stage 5: Push to Hub

**Purpose**: Upload processed documents to HuggingFace Hub

**Implementation Notes**:
- Use `HuggingFaceDatasetWriter` from datatrove
- Support private repositories
- Add dataset card metadata
- Handle authentication with HF_TOKEN
- Support incremental updates

## Adding New Features

### Adding a New Reader

1. Create file in `iwe_pipeline/readers/`
2. Extend `BaseReader` from datatrove
3. Implement `run()` method to yield `Document` objects
4. Add tests in `tests/`

```python
from datatrove.pipeline.readers.base import BaseReader
from datatrove.data import Document

class MyReader(BaseReader):
    name = "🔖 My Reader"
    
    def __init__(self, source_path: str, **kwargs):
        super().__init__(**kwargs)
        self.source_path = source_path
    
    def run(self, data=None, rank=0, world_size=1):
        # Implement your reading logic
        for item in self.get_items():
            yield Document(
                id=item.id,
                text=item.text,
                metadata={"source": "my_reader"}
            )
```

### Adding a New Postprocessor

1. Create file in `iwe_pipeline/postprocessing/`
2. Extend `PipelineStep` from datatrove
3. Implement `run()` method that transforms and yields documents
4. Add configuration parameters in `__init__`
5. Add tests

```python
from datatrove.pipeline.base import PipelineStep
from datatrove.data import Document

class MyProcessor(PipelineStep):
    name = "✨ My Processor"
    type = "🔨 PROCESSOR"
    
    def __init__(self, param1: str = "default", **kwargs):
        super().__init__()
        self.param1 = param1
    
    def run(self, data: Document, rank=0, world_size=1):
        # Process document
        data.text = self.process_text(data.text)
        yield data
```

### Adding a New Server Backend

1. Create implementation in `iwe_pipeline/server/`
2. Update `ServerManager` to support new backend
3. Implement server lifecycle methods: `start()`, `stop()`, `health_check()`
4. Add to pyproject.toml optional dependencies

```python
class MyServerManager:
    def __init__(self, model_path: str, port: int = 8000):
        self.model_path = model_path
        self.port = port
        self.process = None
    
    def start(self):
        # Start inference server
        pass
    
    def stop(self):
        # Stop server gracefully
        pass
    
    def health_check(self) -> bool:
        # Check if server is responding
        return True
```

## Code Style

### Python Style Guide

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters (configured in ruff)
- Use docstrings (Google style) for all public functions/classes

### Linting

```bash
# Run ruff for linting
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Docstring Format

```python
def process_document(text: str, language: str = "en") -> str:
    """
    Process document text with language-specific rules.
    
    Args:
        text: Raw document text to process
        language: ISO language code (default: "en")
    
    Returns:
        Processed text with normalized formatting
        
    Raises:
        ValueError: If language is not supported
    """
    pass
```

### Naming Conventions

- Classes: `PascalCase` (e.g., `KarantaOCRExtractor`)
- Functions/methods: `snake_case` (e.g., `run_ocr_extraction`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `KARANTA_OCR_MODEL`)
- Private methods: `_leading_underscore` (e.g., `_process_page`)

## Testing

### Testing Strategy

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test pipeline stages end-to-end
3. **Mocking**: Mock external services (Azure, HuggingFace, model inference)

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from iwe_pipeline.ocr.karanta_ocr import KarantaOCRExtractor

def test_karanta_ocr_initialization():
    """Test OCR extractor can be initialized with correct parameters."""
    extractor = KarantaOCRExtractor(
        model_name_or_path="taresco/KarantaOCR",
        server_type="vllm",
        device="cuda"
    )
    assert extractor.model_name_or_path == "taresco/KarantaOCR"
    assert extractor.server_type == "vllm"

@patch('iwe_pipeline.ocr.karanta_ocr.convert_pdf_to_images')
def test_ocr_extraction(mock_convert):
    """Test OCR extraction processes PDF correctly."""
    mock_convert.return_value = [Mock()] * 3  # 3 pages
    
    extractor = KarantaOCRExtractor()
    document = Mock(media=[Mock(media_bytes=b"fake_pdf")])
    
    # Test extraction
    result = next(extractor.run(document))
    assert result is not None
```

### Test Coverage

Aim for >80% code coverage. Check coverage with:

```bash
pytest --cov=iwe_pipeline --cov-report=term-missing
```

## Submitting Changes

### Workflow

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Make changes** following the code style guidelines
4. **Add tests** for new functionality
5. **Run tests** to ensure everything passes
6. **Commit changes** with descriptive messages
7. **Push** to your fork: `git push origin feature/my-feature`
8. **Open a Pull Request** with a clear description

### Commit Messages

Follow conventional commits format:

```
feat: add support for Ollama inference server
fix: handle empty PDF pages in OCR extraction
docs: update contributing guide with testing section
test: add integration tests for pipeline stages
refactor: simplify language detection logic
```

### Pull Request Guidelines

- **Title**: Clear, concise description of changes
- **Description**: 
  - What does this PR do?
  - Why is this change needed?
  - How was it tested?
  - Any breaking changes?
- **Tests**: All tests must pass
- **Documentation**: Update relevant docs
- **Review**: Request review from maintainers

### Pull Request Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
How were the changes tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] All tests pass
```

## Development Tips

### Debugging

```python
# Enable datatrove logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test single pipeline stage
from iwe_pipeline.ocr.karanta_ocr import KarantaOCRExtractor

extractor = KarantaOCRExtractor()
test_doc = Document(text="test", media=[...])
result = next(extractor.run(test_doc))
```

### Performance Profiling

```python
# Profile pipeline execution
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run pipeline
run_ocr_extraction()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Local Testing Without Azure

```python
# Use local files instead of Azure Blob
from datatrove.pipeline.readers.jsonl import JsonlReader

# Replace AzureBlobReader with JsonlReader for local testing
reader = JsonlReader(data_folder="./test_data", glob_pattern="*.jsonl")
```

## Getting Help

- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check README.md for user documentation

## License

This project is licensed under the MIT License. By contributing, you agree to license your contributions under the same license.

## Acknowledgments

This pipeline is inspired by HuggingFace's [FinePDFs](https://github.com/huggingface/finepdfs) project and built on the excellent [Datatrove](https://github.com/huggingface/datatrove) framework.
