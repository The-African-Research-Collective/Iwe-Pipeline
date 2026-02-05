"""
Tests for postprocessing blocks.
"""

from iwe_pipeline.blocks.postprocess.boilerplate import BoilerplateRemover
from iwe_pipeline.blocks.postprocess.language_tag import LanguageTag
from iwe_pipeline.blocks.postprocess.normalize import Normalize
from iwe_pipeline.blocks.postprocess.tables import TableCleaner


def test_language_tag_init():
    """Test language tagger initialization."""
    block = LanguageTag(backend="glotlid")
    assert block.backend == "glotlid"
    assert block.threshold == 0.5


def test_normalize_init():
    """Test normalizer initialization."""
    block = Normalize()
    assert block.fix_unicode is True
    assert block.normalize_whitespace is True


def test_boilerplate_init():
    """Test boilerplate remover initialization."""
    block = BoilerplateRemover()
    assert block.remove_headers is True
    assert block.min_text_length == 100


def test_table_cleaner_init():
    """Test table cleaner initialization."""
    block = TableCleaner()
    assert block.detect is True
    assert block.format is True
