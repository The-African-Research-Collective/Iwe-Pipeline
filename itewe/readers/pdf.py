import base64
import os
import tempfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import Literal, NotRequired, TypedDict

import pikepdf
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.utils.logging import logger

from itewe.ids import generate_doc_id
from itewe.utils.pdf_utils import pdftoppm_exists, render_pdf_to_base64png

try:
    from adlfs import AzureBlobFileSystem
except ImportError:
    AzureBlobFileSystem = None


class Media(TypedDict, total=False):
    class MediaMetadata(TypedDict):
        page: int

    id: str
    type: Literal["application/pdf", "image/png"]
    url: str
    media_bytes: str
    metadata: MediaMetadata


class DocumentMetadata(TypedDict, total=False):
    content_md5: NotRequired[str | None]
    etag: NotRequired[str | None]
    num_pages: int
    document_id: NotRequired[str]


class PDFPage(TypedDict):
    class PageMetadata(TypedDict, total=False):
        source: DocumentMetadata
        page: int

    id: str
    media: list[Media]  # len(media) = 1
    metadata: PageMetadata
    text: str


class PDFDocument(TypedDict):
    id: str
    text: str
    media: list[Media]
    metadata: DocumentMetadata


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
    yield_pages_as_documents : bool
        If True, each page of the PDF is yielded as a separate document
    skip_render: bool
        If True, only metadata is yielded. The media field of each document isn't populated.

    See BaseDiskReader for remaining params
    """

    name = "📖 PDF"

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        pdf_to_ppm: bool = False,
        skip_render: bool = False,
        target_longest_image_dim: int = 2048,
        yield_pages_as_documents: bool = False,
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

        if pdf_to_ppm and not pdftoppm_exists():
            raise RuntimeError(
                "pdf_to_ppm=True requires poppler-utils (pdftoppm). "
                "Install it via your system package manager."
            )
        self.pdf_to_ppm = pdf_to_ppm
        self.target_longest_image_dim = target_longest_image_dim
        self.yield_pages_as_documents = yield_pages_as_documents
        self.skip_render = skip_render

    @property
    def has_azure_fs(self):
        if AzureBlobFileSystem is None:
            return False
        return isinstance(self.data_folder.fs, AzureBlobFileSystem)

    def get_azure_document_metadata(self, filepath):
        full_metadata = self.data_folder.fs.info(f"{self.data_folder.path}/{filepath}")

        content_md5_bytes = full_metadata["content_settings"]["content_md5"]
        content_md5_str = (
            base64.b64encode(content_md5_bytes).decode("utf-8") if content_md5_bytes else None
        )

        return {
            "etag": full_metadata["etag"],
            "content_md5": content_md5_str,
            **full_metadata["metadata"],
        }

    def read_file(self, filepath: str) -> Iterable[PDFPage | PDFDocument]:
        with self.data_folder.open(filepath, "rb") as f:
            pdf_bytes = f.read()

            source_document_metadata = {}
            if self.has_azure_fs:
                source_document_metadata = (
                    source_document_metadata | self.get_azure_document_metadata(filepath)
                )

            document_id = (
                source_document_metadata.get("etag", generate_doc_id(filepath))
                if self.has_azure_fs
                else generate_doc_id(filepath)
            )
            if "etag" not in source_document_metadata:
                source_document_metadata["document_id"] = document_id

            try:
                source_document_metadata["num_pages"] = len(pikepdf.open(BytesIO(pdf_bytes)).pages)
            except Exception as e:
                logger.warning(f"Failed to open PDF {filepath}: {e}. Yielding empty document.")
                self.get_document_from_dict(
                    {"text": " ", "metadata": {"source": source_document_metadata}},
                    source_file=filepath,
                    id_in_file=document_id,
                )
                return

            if self.skip_render:
                data = {"text": " ", "metadata": {"source": source_document_metadata}}
                yield self.get_document_from_dict(
                    data, source_file=filepath, id_in_file=document_id
                )
            elif self.yield_pages_as_documents:
                yield from self._read_file_by_pages(filepath, pdf_bytes, source_document_metadata)
            else:
                yield from self._read_file_whole(filepath, pdf_bytes, source_document_metadata)

    def _default_adapter(
        self, data: dict, path: str, id_in_file: int | str
    ) -> PDFPage | PDFDocument:
        """ """
        metadata = data.pop("metadata", {})
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                pass
        if not isinstance(metadata, dict):
            metadata = {"metadata": metadata}
        return {
            "text": data.pop(self.text_key, ""),
            "id": id_in_file,
            "media": data.pop("media", []),
            "metadata": metadata | data,  # remaining data goes into metadata
        }

    def _render_pages(
        self, filepath: str, pdf_bytes: bytes, num_pages: int
    ) -> tuple[str, list[bytes]]:
        if self.pdf_to_ppm:
            tmp_pdf_file = None
            try:
                if self.has_azure_fs:
                    tmp_pdf_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                    tmp_pdf_file.write(pdf_bytes)
                    tmp_pdf_file.flush()
                    pdf_path = tmp_pdf_file.name
                else:
                    pdf_path = filepath

                render_page = partial(
                    render_pdf_to_base64png,
                    local_pdf_path=pdf_path,
                    target_longest_image_dim=self.target_longest_image_dim,
                    as_str=False,
                )
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(render_page, page_num=idx + 1) for idx in range(num_pages)
                    ]
                    return "image/png", [f.result() for f in futures]
            finally:
                if tmp_pdf_file is not None:
                    tmp_pdf_file.close()
                    os.unlink(tmp_pdf_file.name)
        else:

            def extract_page(idx: int) -> bytes:
                pdf = pikepdf.open(BytesIO(pdf_bytes))
                out = pikepdf.Pdf.new()
                out.pages.append(pdf.pages[idx])
                buf = BytesIO()
                out.save(buf)
                page_bytes = buf.getvalue()
                buf.close()
                return page_bytes

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(extract_page, idx) for idx in range(num_pages)]
                return "application/pdf", [f.result() for f in futures]

    def _read_file_whole(
        self, filepath: str, pdf_bytes: bytes, metadata: dict
    ) -> Iterable[PDFDocument]:
        with self.track_time():
            media_type, media_bytes_per_page = self._render_pages(
                filepath, pdf_bytes, metadata["num_pages"]
            )

        data = {
            "text": " ",
            "metadata": {"source": metadata},
            "media": [
                {
                    "id": generate_doc_id(f"{filepath}/{idx}"),
                    "type": media_type,
                    "url": f"{filepath}/{idx}",
                    "media_bytes": media_bytes_per_page[idx],
                    "metadata": {"page": idx},
                }
                for idx in range(metadata["num_pages"])
            ],
        }

        with self.track_time():
            yield self.get_document_from_dict(
                data,
                source_file=filepath,
                id_in_file=metadata.get("document_id", metadata.get("etag")),
            )

    def _read_file_by_pages(
        self, filepath: str, pdf_bytes: bytes, metadata: dict
    ) -> Iterable[PDFPage]:
        with self.track_time():
            media_type, media_bytes_per_page = self._render_pages(
                filepath, pdf_bytes, metadata["num_pages"]
            )

        for idx in range(metadata["num_pages"]):
            page_id = generate_doc_id(f"{filepath}/{idx}")
            data = {
                "text": " ",
                "metadata": {"source": metadata, "page": idx},
                "media": [
                    {
                        "id": page_id,
                        "type": media_type,
                        "url": f"{filepath}/{idx}",
                        "media_bytes": media_bytes_per_page[idx],
                        "metadata": {"page": idx},
                    }
                ],
            }
            with self.track_time():
                yield self.get_document_from_dict(data, source_file=filepath, id_in_file=page_id)
