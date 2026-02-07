import base64
import json
import subprocess
from typing import Any

from datatrove.data import Document
from pypdf import PdfReader

SYSTEM_PROMPT = """Below is the image of one page of a PDF document.
Just return the plain text representation of this document as if you were reading it naturally.
Turn equations into a LaTeX representation, and tables into markdown format. Remove the
headers and footers, but keep references and footnotes.
Read any natural handwriting.
This is likely one page out of several in the document, so be sure to preserve any sentences
that come from the previous page, or continue onto the next page, exactly as they are.
If there is no text at all that you think you should read, you can output null.
if the document contains diacritics, please include them in the output.
Do not hallucinate.
"""


def get_pdf_num_pages(local_pdf_path: str) -> int:
    reader = PdfReader(local_pdf_path)
    return len(reader.pages)


def get_pdf_media_box_width_height(local_pdf_path: str, page_num: int) -> tuple[float, float]:
    """
    Get the MediaBox dimensions for a specific page in a PDF file using the pdfinfo command.

    :param pdf_file: Path to the PDF file
    :param page_num: The page number for which to extract MediaBox dimensions
    :return: A dictionary containing MediaBox dimensions or None if not found
    """
    # Construct the pdfinfo command to extract info for the specific page
    command = [
        "pdfinfo",
        "-f",
        str(page_num),
        "-l",
        str(page_num),
        "-box",
        "-enc",
        "UTF-8",
        local_pdf_path,
    ]

    # Run the command using subprocess
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if there is any error in executing the command
    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")

    # Parse the output to find MediaBox
    output = result.stdout

    for line in output.splitlines():
        if "MediaBox" in line:
            media_box_str: list[str] = line.split(":")[1].strip().split()
            media_box: list[float] = [float(x) for x in media_box_str]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])

    raise ValueError("MediaBox not found in the PDF info.")


def render_pdf_to_base64png(
    local_pdf_path: str, page_num: int, target_longest_image_dim: int = 2048
) -> str:
    longest_dim = max(get_pdf_media_box_width_height(local_pdf_path, page_num))

    # Convert PDF page to PNG using pdftoppm
    pdftoppm_result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-r",
            str(
                target_longest_image_dim * 72 / longest_dim
            ),  # 72 pixels per point is the conversion factor
            local_pdf_path,
        ],
        timeout=120,
        capture_output=True,
    )
    assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
    return base64.b64encode(pdftoppm_result.stdout).decode("utf-8")


def get_png_dimensions_from_base64(base64_data) -> tuple[int, int]:
    """
    Returns the (width, height) of a PNG image given its base64-encoded data,
    without base64-decoding the entire data or loading the PNG itself

    Should be really fast to support filtering

    Parameters:
    - base64_data (str): Base64-encoded PNG image data.

    Returns:
    - tuple: (width, height) of the image.

    Raises:
    - ValueError: If the data is not a valid PNG image or the required bytes are not found.
    """
    # PNG signature is 8 bytes
    png_signature_base64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
    if not base64_data.startswith(png_signature_base64[:8]):
        raise ValueError("Not a valid PNG file")

    # Positions in the binary data where width and height are stored
    width_start = 16  # Byte position where width starts (0-based indexing)
    _width_end = 20  # Byte position where width ends (exclusive)
    _height_start = 20
    height_end = 24

    # Compute the byte range needed (from width_start to height_end)
    start_byte = width_start
    end_byte = height_end

    # Calculate base64 character positions
    # Each group of 3 bytes corresponds to 4 base64 characters
    base64_start = (start_byte // 3) * 4
    base64_end = ((end_byte + 2) // 3) * 4  # Add 2 to ensure we cover partial groups

    # Extract the necessary base64 substring
    base64_substring = base64_data[base64_start:base64_end]

    # Decode only the necessary bytes
    decoded_bytes = base64.b64decode(base64_substring)

    # Compute the offset within the decoded bytes
    offset = start_byte % 3

    # Extract width and height bytes
    width_bytes = decoded_bytes[offset : offset + 4]
    height_bytes = decoded_bytes[offset + 4 : offset + 8]

    if len(width_bytes) < 4 or len(height_bytes) < 4:
        raise ValueError("Insufficient data to extract dimensions")

    # Convert bytes to integers
    width = int.from_bytes(width_bytes, "big")
    height = int.from_bytes(height_bytes, "big")

    return width, height


def build_message(image_base64: bytes, system_prompt: str = SYSTEM_PROMPT):
    """Format messages in OpenAI-compatible multimodal chat format."""

    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        }
    ]

    return prompt


def prepare_requests_postprocess(
    document: Document, model_name: str, max_tokens: int
) -> list[tuple[dict[str, Any], int]]:
    from loguru import logger as _logger  # use local logger if available

    requests: list[tuple[dict[str, Any], int]] = []

    for page_index, page_bytes in enumerate(document.media):
        try:
            requests.append(
                (
                    {
                        "messages": build_message(page_bytes["media_bytes"]),
                        "model": model_name,
                        "temperature": 0.0,
                        "max_tokens": max_tokens,
                    },
                    page_index,
                )
            )

        except Exception as e:
            _logger.error(f"Error preparing page request: {e}")

    return requests


async def rollout_postprocess(document: Document, generate: Any, **kwargs) -> Any:
    # Use the existing logic to prepare requests
    import asyncio
    import atexit
    from concurrent.futures import ProcessPoolExecutor

    model_name = kwargs.get("model_name_or_path", "taresco/KarantaOCR")
    max_tokens = kwargs.get("max_tokens", 8192)

    if not hasattr(rollout_postprocess, "process_pool"):
        rollout_postprocess.process_pool = ProcessPoolExecutor(max_workers=4)
        atexit.register(rollout_postprocess.process_pool.shutdown)

    from .utils import prepare_requests_postprocess as _prep

    requests_tuple = await asyncio.get_event_loop().run_in_executor(
        rollout_postprocess.process_pool, _prep, document, model_name, max_tokens
    )
    request_ids = [i for _, i in requests_tuple]
    document.metadata["request_page_indices"] = request_ids
    request_payload_max_chars = kwargs.get("request_payload_max_chars", 1200)
    request_previews: list[str] = []
    for request, _ in requests_tuple:
        try:
            preview = json.dumps(request, ensure_ascii=False)
        except TypeError:
            preview = str(request)
        if request_payload_max_chars and len(preview) > request_payload_max_chars:
            preview = preview[:request_payload_max_chars] + f"... (truncated, {len(preview)} chars)"
        request_previews.append(preview)
    document.metadata["request_payloads"] = request_previews

    # Run inference for all relevant pages
    tasks = [generate(request) for request, _ in requests_tuple]
    results = await asyncio.gather(*tasks)

    # Store results in metadata for postprocess_postprocess
    document.metadata["inference_results"] = results

    # Run post-processing
    # postprocess_postprocess(document)

    return results
