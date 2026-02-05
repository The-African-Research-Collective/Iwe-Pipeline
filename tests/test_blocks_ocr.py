"""
Tests for OCR blocks.
"""

from iwe_pipeline.blocks.ocr.karanta_vllm_ocr import KarantaVLLMOCR
from iwe_pipeline.blocks.ocr.split_pages import SplitPages


def test_karanta_ocr_init():
    """Test KarantaOCR block initialization."""
    block = KarantaVLLMOCR(server_url="http://localhost:8000")
    assert block.server_url == "http://localhost:8000"
    assert block.mode == "document"


def test_split_pages_init():
    """Test SplitPages block initialization."""
    block = SplitPages()
    assert block is not None


def test_ocr_extraction():
    """Test OCR text extraction."""
    pass
