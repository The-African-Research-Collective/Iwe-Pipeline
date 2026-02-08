"""
Stable document and page ID generation utilities.

Ensures consistent ID generation across pipeline stages for reproducibility
and deduplication.
"""

import hashlib


def generate_doc_id(blob_url: str) -> str:
    """
    Generate stable document ID from blob URL.

    Args:
        blob_url: Azure blob URL

    Returns:
        Stable document ID (hex hash)
    """
    return hashlib.sha256(blob_url.encode()).hexdigest()[:16]


def generate_page_id(doc_id: str, page_num: int) -> str:
    """
    Generate stable page ID from document ID and page number.

    Args:
        doc_id: Parent document ID
        page_num: Page number (0-indexed)

    Returns:
        Page ID in format "doc_id::pXXXX"
    """
    return f"{doc_id}::p{page_num:04d}"


def parse_page_id(page_id: str) -> tuple[str, int]:
    """
    Parse page ID back into document ID and page number.

    Args:
        page_id: Page ID in format "doc_id::pXXXX"

    Returns:
        Tuple of (doc_id, page_num)
    """
    doc_id, page_str = page_id.split("::")
    page_num = int(page_str[1:])  # Remove 'p' prefix
    return doc_id, page_num


def is_page_id(doc_id: str) -> bool:
    """
    Check if an ID is a page ID (vs document ID).

    Args:
        doc_id: Document or page ID

    Returns:
        True if page ID, False otherwise
    """
    return "::" in doc_id
