import tempfile
from collections.abc import Callable
from io import BytesIO

from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader
from pypdf import PdfReader, PdfWriter

from iwe_pipeline.utils import pdftoppm_exists, render_pdf_to_base64png

try:
    from adlfs import AzureBlobFileSystem
except ImportError:
    AzureBlobFileSystem = None


class PDFReader(BaseDiskReader):
    """Read PDF files from local or remote FS.
        Will read each page as a separate document.

    Parameters
    ----------
    pdf_to_ppm : bool
        If True, each page of the PDF will be converted to a PNG file.
    target_longest_image_dim : int
        Desired length of the longest side of the output PNG image, in pixels.
        The DPI passed to pdftoppm is calculated from this and the PDF page size.

    See BaseDiskReader for remaining params
    """

    name = "📖 PDF"

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        pdf_to_ppm: bool = False,
        target_longest_image_dim: int = 2048,
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )

        if self.pdf_to_ppm and not pdftoppm_exists:
            raise RuntimeError(
                "pdf_to_ppm=True requires poppler-utils (pdftoppm). "
                "Install it via your system package manager."
            )
        self.pdf_to_ppm = pdf_to_ppm
        self.target_longest_image_dim = target_longest_image_dim

    @property
    def has_azure_fs(self):
        if AzureBlobFileSystem is None:
            return False
        return isinstance(self.data_folder.fs, AzureBlobFileSystem)

    def get_azure_document_metadata(self, filepath):
        full_metadata = self.data_folder.fs.info(f"{self.data_folder.path}/{filepath}")
        return {
            "etag": full_metadata["etag"],
            "content_md5": full_metadata["content_settings"]["content_md5"],
            **full_metadata["metadata"],
        }

    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "rb") as f:
            reader = PdfReader(f)

            source_document_metadata = {
                "num_pages": len(reader.pages),
            }
            if self.has_azure_fs:
                source_document_metadata = self.get_azure_document_metadata(filepath)

            for idx, page in enumerate(reader.pages):
                writer = PdfWriter()
                writer.add_page(page)

                buffer = BytesIO()
                writer.write(buffer.getvalue())
                data = {
                    "text": " ",
                    "metadata": {"source": source_document_metadata, "page": idx},
                }

                if self.pdf_to_ppm:
                    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                        tmp_pdf.write(buffer)
                        tmp_pdf.flush()

                        png_bytes = render_pdf_to_base64png(
                            tmp_pdf.name,
                            target_longest_image_dim=self.target_longest_image_dim,
                            as_str=False,
                        )

                        data["media"] = [{"media_bytes": png_bytes, "media_type": "image/png"}]
                else:
                    data["media"] = [{"media_bytes": buffer, "media_type": "application/pdf"}]

                with self.track_time():
                    # NOTE: document.id will be filepath/page_idx
                    yield self.get_document_from_dict(data, filepath, idx)
