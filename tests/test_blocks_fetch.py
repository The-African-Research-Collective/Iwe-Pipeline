"""
Tests for fetch blocks.
"""

from iwe_pipeline.blocks.fetch.azure_fetch_pdf import AzureFetchPDF


def test_azure_fetch_init():
    """Test Azure fetch block initialization."""
    block = AzureFetchPDF(download_dir="./data/test")
    assert block.download_dir.name == "test"
    assert block.cache_enabled is True


def test_azure_fetch_download():
    """Test PDF download."""
    pass
