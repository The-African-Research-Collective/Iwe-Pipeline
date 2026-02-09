from io import BytesIO

from datatrove.pipeline.readers.base import BaseDiskReader
from pypdf import PdfReader, PdfWriter


class PDFReader(BaseDiskReader):
    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "rb") as f:
            reader = PdfReader(f)

            for idx, page in enumerate(reader.pages):
                writer = PdfWriter()
                writer.add_page(page)

                buffer = BytesIO()
                writer.write(buffer)
                data = {"pdf_bytes": buffer.getvalue(), "text": " "}

                with self.track_time():
                    # NOTE: This function puts pdf_bytes in document.metadata
                    # document.id will be filepath/page_idx
                    yield self.get_document_from_dict(data, filepath, idx)
