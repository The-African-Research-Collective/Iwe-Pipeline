#!/usr/bin/env python3
"""
Create Azure Blob Manifest

Lists all blobs in an Azure container and creates a manifest file
for reproducible, shardable processing.

Usage:
    python scripts/make_manifest.py \
        --container mycontainer \
        --prefix pdfs/ \
        --output manifests/azure_blobs.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def list_blobs(container_name: str, connection_string: str, prefix: str = "") -> list:
    """
    List all blobs in an Azure container.

    Args:
        container_name: Name of the container
        connection_string: Azure connection string
        prefix: Optional prefix to filter blobs

    Returns:
        List of blob metadata dictionaries
    """
    pass


def main():
    """Main entry point for manifest creation."""
    parser = argparse.ArgumentParser(description="Create Azure Blob manifest")
    parser.add_argument("--container", type=str, required=True, help="Azure Blob container name")
    parser.add_argument(
        "--connection-string",
        type=str,
        help="Azure connection string (or use AZURE_CONNECTION_STRING env var)",
    )
    parser.add_argument("--prefix", type=str, default="", help="Blob prefix to filter")
    parser.add_argument(
        "--output",
        type=str,
        default="manifests/azure_blobs.jsonl",
        help="Output manifest file path",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Creating Azure Blob Manifest")
    logger.info("=" * 80)

    # List blobs
    logger.info(f"Listing blobs from container: {args.container}")
    blobs = list_blobs(args.container, args.connection_string, args.prefix)

    logger.info(f"Found {len(blobs)} blobs")

    # Write manifest
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for blob in blobs:
            f.write(json.dumps(blob) + "\n")

    logger.info(f"✓ Manifest written to {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
