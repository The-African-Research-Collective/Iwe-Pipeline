import base64
import logging
import re
import subprocess
from collections import deque

import pymupdf

from itewe.utils.loggers import LoggerStream

pymupdf_error_regexes = [
    r"format error: object out of range",
    r"syntax error: no XObject subtype specified",
    r"syntax error: syntax error in content stream",
    r"object is not a stream",
    r"syntax error: syntax error in array",
    r"format error: cannot load page tree",
    r"syntax error: cannot parse indirect object",
]

corrupted_logs_regex = re.compile("|".join(pymupdf_error_regexes))


def check_is_corrupted_or_encrypted(pymupdf_doc: pymupdf.Document) -> bool:
    if pymupdf_doc.is_encrypted or pymupdf_doc.needs_pass:
        return True

    # Capture logs
    logger_stream = LoggerStream(logging.getLogger("pymupdf"))
    with logger_stream as log_output:
        # Read all pages to trigger errors
        error_logs = ""
        try:
            deque((page.get_text() for page in pymupdf_doc.pages()), maxlen=0)
        except Exception as e:
            error_logs += str(e)
        logs = log_output.value().encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        if error_logs:
            logs += f"\n\nError logs: {error_logs}"

    if corrupted_logs_regex.search(logs):
        return True

    return False


def get_pdf_num_pages(local_pdf_path: str) -> int:
    doc = pymupdf.open(filename=local_pdf_path, filetype="pdf")
    return len(doc)


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
    local_pdf_path: str, page_num: int, target_longest_image_dim: int = 2048, as_str: bool = True
) -> bytes | str:
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
    png_bytes = base64.b64encode(pdftoppm_result.stdout)

    if as_str:
        return png_bytes.decode("utf-8")

    return png_bytes


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


def pdftoppm_exists():
    proc = subprocess.Popen(
        "pdftoppm --help", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return "pdftoppm" in str(proc.communicate())
