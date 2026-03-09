
import pypdfium2 as pdfium
from IPython.display import display


def display_pdf(pdf_bytes: bytes, scale: float = 0.5):
    pdf = pdfium.PdfDocument(pdf_bytes)

    for page in pdf:
        bitmap = page.render(scale=scale)
        display(bitmap.to_pil())
