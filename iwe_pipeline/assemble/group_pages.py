"""
Group Pages Block

Reassembles page-level documents back into document-level documents.
Reverses the split_pages operation.
"""

from collections import defaultdict
from collections.abc import Generator
from typing import Any

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep

from iwe_pipeline.ids import generate_doc_id, is_page_id, parse_page_id


class GroupPages(PipelineStep):
    """
    Group page documents back into parent documents.

    Takes Documents with IDs like "doc_id::p0000" and groups them
    back into a single Document per parent doc_id.
    """

    name = "📑 Group Pages"
    type = "🔨 PROCESSOR"

    def __init__(
        self,
        join_separator: str = "\n\n",
        inference_metadata_keys: tuple[str, ...] = ("rollout_results", "inference_results"),
        **kwargs,
    ):
        """
        Initialize page grouper.

        Args:
            join_separator: Separator for joining page texts
            inference_metadata_keys: Metadata keys to check for inference output
        """
        super().__init__()
        self.join_separator = join_separator
        self.inference_metadata_keys = inference_metadata_keys

    def _media_url_to_source_page(self, media: Any) -> tuple[str | None, int | None]:
        """
        Parse source filepath and page index from media URL "{filepath}/{idx}".
        """
        if isinstance(media, dict):
            url = media.get("url")
            media_metadata = media.get("metadata") or {}
        else:
            url = getattr(media, "url", None)
            media_metadata = getattr(media, "metadata", {}) or {}

        if isinstance(url, str) and "/" in url:
            source_path, page_str = url.rsplit("/", 1)
            if page_str.isdigit():
                return source_path, int(page_str)

        page_index = media_metadata.get("page")
        if isinstance(page_index, int):
            return url, page_index

        return None, None

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            text = value.get("text") or value.get("output_text") or value.get("completion")
            if text is None:
                return ""
            return str(text).strip()
        text_attr = getattr(value, "text", None)
        if text_attr is not None:
            return str(text_attr).strip()
        if isinstance(value, int | float | bool):
            return str(value).strip()
        return ""

    def _extract_page_texts(self, document: Document) -> dict[int, str]:
        """
        Extract per-page text from inference metadata.
        """
        metadata = document.metadata or {}
        raw_results = None
        for key in self.inference_metadata_keys:
            if key in metadata:
                raw_results = metadata.get(key)
                break

        if raw_results is None:
            return {}

        rollouts = raw_results if isinstance(raw_results, list) else [raw_results]
        request_page_indices = metadata.get("request_page_indices")

        page_texts: dict[int, str] = {}
        for rollout in rollouts:
            if isinstance(rollout, list):
                if (
                    isinstance(request_page_indices, list)
                    and len(request_page_indices) == len(rollout)
                    and all(isinstance(page, int) for page in request_page_indices)
                ):
                    page_result_pairs = zip(request_page_indices, rollout)
                else:
                    page_result_pairs = enumerate(rollout)

                for page_idx, result in page_result_pairs:
                    text = self._normalize_text(result)
                    if text and page_idx not in page_texts:
                        page_texts[int(page_idx)] = text
                continue

            page_idx = 0
            if (
                isinstance(request_page_indices, list)
                and len(request_page_indices) == 1
                and isinstance(request_page_indices[0], int)
            ):
                page_idx = request_page_indices[0]

            text = self._normalize_text(rollout)
            if text and page_idx not in page_texts:
                page_texts[page_idx] = text

        return page_texts

    def _resolve_group_doc_id(self, document: Document, source_path: str) -> str:
        metadata = document.metadata or {}
        source_metadata = metadata.get("source")
        if isinstance(source_metadata, dict):
            if isinstance(source_metadata.get("document_id"), str):
                return source_metadata["document_id"]
            if isinstance(source_metadata.get("etag"), str):
                return source_metadata["etag"]

        if is_page_id(document.id):
            try:
                doc_id, _ = parse_page_id(document.id)
                return doc_id
            except ValueError:
                pass

        return generate_doc_id(source_path)

    def _base_group_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        grouped_metadata = dict(metadata or {})
        grouped_metadata.pop("page", None)
        grouped_metadata.pop("request_page_indices", None)
        grouped_metadata.pop("request_payloads", None)
        for key in self.inference_metadata_keys:
            grouped_metadata.pop(key, None)
        return grouped_metadata

    def run(
        self,
        data: Generator[Document, None, None] | None,
        rank: int = 0,
        world_size: int = 1,
    ) -> Generator[Document, None, None]:
        """
        Group pages into documents.

        Expects page-level inference output and groups pages by the source
        filepath extracted from media URL values ("{filepath}/{idx}").

        Args:
            data: Input page-level documents from inference output
            rank: Current process rank
            world_size: Total number of processes

        Yields:
            Grouped document-level output
        """
        if data is None:
            return

        grouped_texts: dict[str, dict[int, str]] = defaultdict(dict)
        grouped_media: dict[str, dict[int, list[Any]]] = defaultdict(dict)
        grouped_metadata: dict[str, dict[str, Any]] = {}
        grouped_ids: dict[str, str] = {}

        for document in data:
            page_texts = self._extract_page_texts(document)
            page_entries: list[tuple[str, int, list[Any]]] = []

            for media in document.media:
                source_path, page_idx = self._media_url_to_source_page(media)
                if source_path is None or page_idx is None:
                    continue
                page_entries.append((source_path, page_idx, [media]))

            # Fallback for ID-based page documents without usable media URL.
            if not page_entries and is_page_id(document.id):
                try:
                    source_path, page_idx = parse_page_id(document.id)
                    page_entries.append((source_path, page_idx, []))
                except ValueError:
                    pass

            # Not groupable; keep the document in the stream.
            if not page_entries:
                if page_texts:
                    ordered = [page_texts[idx] for idx in sorted(page_texts) if page_texts[idx]]
                    if ordered:
                        document.text = self.join_separator.join(ordered)
                yield document
                continue

            for source_path, page_idx, media_items in page_entries:
                text = page_texts.get(page_idx, "")
                if not text and len(page_texts) == 1:
                    text = next(iter(page_texts.values()))
                if not text:
                    text = document.text.strip()

                existing_text = grouped_texts[source_path].get(page_idx, "")
                if text and not existing_text:
                    grouped_texts[source_path][page_idx] = text
                else:
                    grouped_texts[source_path].setdefault(page_idx, existing_text)

                if page_idx not in grouped_media[source_path]:
                    grouped_media[source_path][page_idx] = media_items

                if source_path not in grouped_metadata:
                    grouped_metadata[source_path] = self._base_group_metadata(document.metadata)
                if source_path not in grouped_ids:
                    grouped_ids[source_path] = self._resolve_group_doc_id(document, source_path)

        for source_path in sorted(grouped_texts):
            page_map = grouped_texts[source_path]
            ordered_pages = sorted(page_map)
            ordered_texts = [page_map[page] for page in ordered_pages if page_map[page]]
            combined_text = self.join_separator.join(ordered_texts)

            combined_media: list[Any] = []
            for page in ordered_pages:
                combined_media.extend(grouped_media[source_path].get(page, []))

            metadata = dict(grouped_metadata.get(source_path, {}))
            metadata["source_path"] = source_path
            metadata["grouped_page_count"] = len(ordered_pages)
            metadata["grouped_page_indices"] = ordered_pages

            grouped_document = Document(
                text=combined_text,
                id=grouped_ids[source_path],
                media=combined_media,
                metadata=metadata,
            )
            self.stat_update("grouped_documents")
            self.stat_update("grouped_pages", value=len(ordered_pages), unit="document")
            self.update_doc_stats(grouped_document)
            yield grouped_document
