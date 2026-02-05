"""
OCR Index Writer

Wrapper around JsonlWriter with OCR-specific conventions.
"""

from datatrove.pipeline.writers.jsonl import JsonlWriter


class OCRIndexWriter(JsonlWriter):
    """
    Writer for OCR index (intermediate contract).

    Standardizes output format and location for OCR results.
    """

    def __init__(
        self,
        output_folder: str = "./data/ocr_index",
        compression: str = "gz",
        output_filename: str = "${rank}.jsonl.gz",
        **kwargs,
    ):
        """
        Initialize OCR index writer.

        Args:
            output_folder: Output directory
            compression: Compression format (gz, zst, or None)
            output_filename: Output filename pattern
        """
        super().__init__(
            output_folder=output_folder,
            compression=compression,
            output_filename=output_filename,
            **kwargs,
        )
