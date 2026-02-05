"""
Tests for quality scoring blocks.
"""

from iwe_pipeline.blocks.quality.bert_score import BertQualityScore


def test_bert_score_init():
    """Test BERT quality scorer initialization."""
    block = BertQualityScore()
    assert block.chunk_size == 512
    assert block.threshold == 0.5
    assert block.filter_low_quality is False


def test_quality_scoring():
    """Test quality scoring."""
    pass
