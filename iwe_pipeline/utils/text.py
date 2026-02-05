"""
Shared text processing utilities.

Common text operations used across multiple blocks.
"""

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    - Converts multiple spaces to single space
    - Normalizes newlines
    - Strips leading/trailing whitespace

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Multiple newlines → double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Multiple spaces → single space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip each line
    lines = [line.strip() for line in text.split("\n")]

    return "\n".join(lines).strip()


def fix_unicode(text: str) -> str:
    """
    Fix common unicode encoding issues.

    Args:
        text: Input text

    Returns:
        Fixed text
    """
    # Normalize to NFC form
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width characters
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)

    return text


def remove_control_chars(text: str) -> str:
    """
    Remove control characters except newlines and tabs.

    Args:
        text: Input text

    Returns:
        Text without control characters
    """
    return "".join(
        char
        for char in text
        if char == "\n" or char == "\t" or not unicodedata.category(char).startswith("C")
    )


def expand_ligatures(text: str) -> str:
    """
    Expand common ligatures to separate characters.

    Args:
        text: Input text

    Returns:
        Text with expanded ligatures
    """
    ligature_map = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬆ": "st",
        "Ꜳ": "AA",
        "ꜳ": "aa",
    }

    for ligature, replacement in ligature_map.items():
        text = text.replace(ligature, replacement)

    return text
