import json
import os
import tempfile
from typing import Any

from datatrove.data import Document
from datatrove.pipeline.inference.types import InferenceResult

from itewe.utils.pdf_utils import render_pdf_to_base64png

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


def build_message(image_base64: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
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

    def _get_base64_png(media: Any) -> str:
        media_type = getattr(media, "type", None)
        if media_type is None and isinstance(getattr(media, "metadata", None), dict):
            media_type = media.metadata.get("type")
        media_bytes = media.media_bytes

        if media_type == "application/pdf":
            tmp_pdf_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            try:
                if isinstance(media_bytes, str):
                    tmp_pdf_file.write(media_bytes.encode("utf-8"))
                else:
                    tmp_pdf_file.write(media_bytes)
                tmp_pdf_file.flush()
                return str(render_pdf_to_base64png(tmp_pdf_file.name, page_num=1, as_str=True))
            finally:
                tmp_pdf_file.close()
                os.unlink(tmp_pdf_file.name)

        if isinstance(media_bytes, bytes):
            return media_bytes.decode("utf-8")
        return str(media_bytes)

    requests: list[tuple[dict[str, Any], int]] = []

    for media in document.media:
        try:
            image_base64 = _get_base64_png(media)
            requests.append(
                (
                    {
                        "messages": build_message(image_base64),
                        "model": model_name,
                        "temperature": 0.0,
                        "max_tokens": max_tokens,
                    },
                    media.metadata["page"],
                )
            )

        except Exception as e:
            _logger.error(f"Error preparing page request: {e}")

    return requests


def _extract_natural_text(result: Any) -> str:
    if result is None:
        return ""

    if isinstance(result, str):
        parsed: Any = result
        try:
            parsed = json.loads(result)
        except Exception:
            return result
        return _extract_natural_text(parsed)

    if isinstance(result, list):
        texts = [_extract_natural_text(item) for item in result]
        return "\n".join([text for text in texts if text]).strip()

    if isinstance(result, dict):
        natural_text = result.get("natural_text")
        if isinstance(natural_text, str):
            return natural_text
        if natural_text is None:
            # fallback paths for other possible response schemas
            text = result.get("text")
            if isinstance(text, str):
                return text
            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        return _extract_natural_text(content)
        return ""

    return str(result)


class RepetitionChecker:
    def __init__(self, max_consecutive_chars: int = 200):
        self.max_consecutive_chars = max_consecutive_chars
        self._last_char: str | None = None
        self._run_length = 0

    def add_char(self, char: str) -> str | None:
        if char == self._last_char:
            self._run_length += 1
        else:
            self._last_char = char
            self._run_length = 1
        if self._run_length >= self.max_consecutive_chars:
            return "repetition"
        return None


def postprocess_extract(document: Document) -> Document | None:
    page_results = document.metadata.get("inference_results", [])
    if len(page_results) == 0:
        return None

    total_pages = int(document.metadata.get("num_pages", 0))
    request_page_indices = document.metadata.get("request_page_indices", [])
    if total_pages <= 0:
        total_pages = max(len(page_results), max(request_page_indices, default=-1) + 1)

    page_dict = {index: "<--- failed_to_process_page --->" for index in range(total_pages)}
    for i, page_result in enumerate(page_results):
        page_index = request_page_indices[i] if i < len(request_page_indices) else i

        if not isinstance(page_result, InferenceResult):
            continue

        stop_reason = page_result.finish_reason
        content_text = _extract_natural_text(page_result.text).strip() or page_result.text

        checker = RepetitionChecker()
        for char in content_text:
            repetition_type = checker.add_char(char)
            if repetition_type is not None:
                stop_reason = repetition_type
                break

        if stop_reason == "stop":
            content = content_text
        else:
            content = f"<--- stop_reason_{stop_reason} --->"
        page_dict[int(page_index)] = content

    offsets = [0]
    running = 0
    for page in page_dict.values():
        running += len(page)
        offsets.append(running)
    offsets = offsets[1:]
    document.text = "\n".join(page_dict.values())

    extraction_metadata = {
        "page_offsets": offsets,
        "extracted_pages": len(page_results),
        "total_pages": len(page_dict),
    }
    if len(document.media) > 0 and getattr(document.media[0], "metadata", None) is not None:
        document.media[0].metadata = document.media[0].metadata | extraction_metadata
    if "inference_results" in document.metadata:
        del document.metadata["inference_results"]
    return document


async def rollout_postprocess(document: Document, generate: Any, **kwargs) -> Any:
    # Use the existing logic to prepare requests
    import asyncio
    import atexit
    from concurrent.futures import ProcessPoolExecutor

    from loguru import logger as _logger  # use local logger if available

    model_name = kwargs.get("model_name_or_path", "taresco/KarantaOCR")
    max_tokens = kwargs.get("max_tokens", 8192)

    if not hasattr(rollout_postprocess, "process_pool"):
        rollout_postprocess.process_pool = ProcessPoolExecutor(max_workers=4)
        atexit.register(rollout_postprocess.process_pool.shutdown)

    from itewe.utils.rollout_utils import prepare_requests_postprocess as _prep

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
    # Allow individual request failures without crashing the whole pipeline.
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for result in raw_results:
        if isinstance(result, Exception):
            _logger.error(f"Inference request failed: {result}")
            results.append({"text": "", "error": str(result)})
        else:
            results.append(result)

    # Run post-processing
    document.metadata["inference_results"] = results
    postprocess_extract(document)

    return results
