"""
I/O utilities for file system operations.

Helpers for paths, caching, and fsspec integration.
"""

import hashlib
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists, create if needed.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_cache_path(key: str, cache_dir: str = "./cache") -> Path:
    """
    Get cache file path for a given key.

    Args:
        key: Cache key (will be hashed)
        cache_dir: Cache directory

    Returns:
        Path to cache file
    """
    cache_dir = ensure_dir(cache_dir)
    key_hash = hashlib.md5(key.encode()).hexdigest()
    return cache_dir / f"{key_hash}.cache"


def is_cached(key: str, cache_dir: str = "./cache") -> bool:
    """
    Check if key exists in cache.

    Args:
        key: Cache key
        cache_dir: Cache directory

    Returns:
        True if cached, False otherwise
    """
    return get_cache_path(key, cache_dir).exists()
