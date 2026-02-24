import json
from typing import Any

from datatrove.data import Document

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

    requests: list[tuple[dict[str, Any], int]] = []

    for media in document.media:
        try:
            requests.append(
                (
                    {
                        "messages": build_message(media.media_bytes.decode("utf-8")),
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
    # postprocess_postprocess(document)

    return results
