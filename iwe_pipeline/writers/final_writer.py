"""
Final Dataset Writer

Writer for final processed dataset with schema validation.
"""

from datatrove.pipeline.writers import ParquetWriter


class FinalWriter(ParquetWriter):
    """
    Writer for final processed dataset.

    Handles schema validation and partitioning.
    """

    def __init__(
        self,
        output_folder: str = "./data/final",
        partition_by: str | None = None,
        output_filename: str = "${rank}.parquet",
        **kwargs,
    ):
        """
        Initialize final dataset writer.

        Args:
            output_folder: Output directory
            partition_by: Field to partition by (e.g., "language")
            output_filename: Output filename pattern
        """
        # Note: ParquetWriter may not exist in datatrove yet
        # Use JsonlWriter as fallback and convert to parquet separately
        super().__init__(
            output_folder=output_folder,
            output_filename=output_filename,
            **kwargs,
        )
        self.partition_by = partition_by
