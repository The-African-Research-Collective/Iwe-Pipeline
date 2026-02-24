"""
Modified from OCR predictor block in finepdfs repo:
https://github.com/huggingface/finepdfs/blob/main/blocks/predictor/ocr_predictor.py
"""

import io
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import chain

import numpy as np
import pymupdf


@dataclass
class PageFeatures:
    unique_font_count: int
    char_count: int
    text_box_count: int
    avg_text_box_length: float
    text_area_ratio: float
    # hidden text features
    hidden_char_count: int
    hidden_text_box_count: int
    hidden_avg_text_box_length: float
    hidden_text_area_ratio: float
    # image features
    image_count: int
    non_junk_image_count: int
    bitmap_proportion: float
    max_merged_strip_area: float
    # drawing and vector graphics features
    drawing_strokes_count: int
    vector_graphics_obj_count: int


@dataclass
class PDFChunkFeatures:
    is_form: bool
    creator_or_producer_is_known_scanner: bool
    garbled_text_ratio: float
    global_garbled_text_ratio: float
    num_junk_image_xrefs: int
    num_unique_image_xrefs: int
    sampled_page_indices: list[int]
    num_pages_requested_for_sampling: int
    num_pages_successfully_sampled: int

    sampled_page_features: list[PageFeatures] = field(default_factory=list)

    def get_page_features(self) -> dict[str, list]:
        assert self.sampled_page_features is not None
        features = [asdict(page) for page in self.sampled_page_features]

        return {f"page_level_{k}s": [f[k] for f in features] for k in features[0].keys()}

    def get_flattened_features(self, resample: bool = False) -> dict[str, float | int]:
        doc_level_features = [
            "creator_or_producer_is_known_scanner",
            "garbled_text_ratio",
            "is_form",
            "num_pages_successfully_sampled",
            # Due to a bug, this last two features are not passed to the XGB Predictor model
            # TODO: @theyorubayesian - Explore retraining the model with these features included
            # "num_junk_image_xrefs",
            # "num_unique_image_xrefs",
        ]
        page_level_features = [
            "page_level_unique_font_counts",
            "page_level_char_counts",
            "page_level_text_box_counts",
            "page_level_avg_text_box_lengths",
            "page_level_text_area_ratios",
            "page_level_hidden_char_counts",
            "page_level_hidden_text_box_counts",
            "page_level_hidden_avg_text_box_lengths",
            "page_level_hidden_text_area_ratios",
            "page_level_image_counts",
            "page_level_non_junk_image_counts",
            "page_level_bitmap_proportions",
            "page_level_max_merged_strip_areas",
            # These last two features aren't pluralized. Nit.
            "page_level_drawing_strokes_counts",
            "page_level_vector_graphics_obj_counts",
        ]

        doc_features = asdict(self)  # Use asdict instead of model_dump
        sampled_page_features = self.get_page_features()

        assert self.num_pages_successfully_sampled == len(
            sampled_page_features["page_level_char_counts"]
        )

        page_idxs = list(range(self.num_pages_successfully_sampled))

        if resample and (
            self.num_pages_successfully_sampled < self.num_pages_requested_for_sampling
        ):
            page_idxs += np.random.choice(
                self.num_pages_requested_for_sampling,
                self.num_pages_requested_for_sampling - self.num_pages_successfully_sampled,
                replace=True,
            ).tolist()

        flattened_features = {k: v for k, v in doc_features.items() if k in doc_level_features}
        used_keys = set()

        for key in page_level_features:
            _new_key = key
            if key in [
                "page_level_drawing_strokes_counts",
                "page_level_vector_graphics_obj_counts",
            ]:
                # Nit: For some weird reason these two features weren't pluralized
                # Can be fixed when we retrain the model
                _new_key = key[:-1]

            flattened_features.update(
                {f"{_new_key}_page{i + 1}": sampled_page_features[key][i] for i in page_idxs}
            )
            used_keys.add(key)

        return flattened_features


class PDFFeatureExtractor:
    UNICODE_REPLACEMENT_CHAR = chr(0xFFFD)
    JUNK_IMAGE_MAX_PAGE_THRESHOLD_RATIO = 0.3
    JUNK_IMAGE_THRESHOLD_MAX_PAGES = 3
    KNOWN_SCANNER_CREATORS_AND_PRODUCERS = {
        "scanner",
        "scan",
        "epson",
        "hp scanjet",
        "canon",
        "fujitsu",
        "kodak",
        "brother",
        "xerox",
        "lexmark",
        "kmc",
        "kofax",
        "ricoh",
        "iris",
        "capturedocument",
        "paperport",
        "readiris",
        "simpleocr",
    }
    MERGE_MAX_OFFSET = 5
    MERGE_MAX_GAP = 2

    def __init__(self, num_pages_to_sample: int = 5, num_chunks: int = 1):
        self.num_pages_to_sample = num_pages_to_sample
        self.num_chunks = num_chunks

    def _creator_or_producer_is_known_scanner(self, doc: pymupdf.Document) -> bool:
        creator = doc.metadata.get("creator", "").lower()
        producer = doc.metadata.get("producer", "").lower()
        for kw in self.KNOWN_SCANNER_CREATORS_AND_PRODUCERS:
            if kw in creator or kw in producer:
                return True
        return False

    @staticmethod
    def _get_bbox_area(bbox: tuple[int]) -> int:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _heuristic_merge_image_strips_on_page(
        self, single_page_image_list: list, page_width: float, page_height: float
    ) -> list:
        if not single_page_image_list:
            return []

        page_images_deduped_by_bbox = []
        dedup_bboxes = set()
        for img_data in single_page_image_list:
            bbox_tuple = (img_data[0], img_data[1], img_data[2], img_data[3])
            if bbox_tuple not in dedup_bboxes:
                dedup_bboxes.add(bbox_tuple)
                page_images_deduped_by_bbox.append(img_data)

        if not page_images_deduped_by_bbox:
            return []

        page_images_deduped_by_bbox.sort(key=lambda img: (img[1], img[0]))

        if not page_images_deduped_by_bbox:
            return []
        current_page_merged_list = [page_images_deduped_by_bbox[0]]

        for img_to_potentially_merge in page_images_deduped_by_bbox[1:]:
            x0_curr, y0_curr, x1_curr, y1_curr, imgid_curr = img_to_potentially_merge
            last_merged_img = current_page_merged_list[-1]
            x0_last, y0_last, x1_last, y1_last, _ = last_merged_img

            img_curr_width = abs(x1_curr - x0_curr)
            img_curr_height = abs(y1_curr - y0_curr)

            full_width_curr = page_width > 0 and (img_curr_width >= page_width * 0.9)
            full_height_curr = page_height > 0 and (img_curr_height >= page_height * 0.9)

            can_merge = False
            if full_width_curr:
                is_vertically_aligned_and_close = (
                    (abs(x0_last - x0_curr) <= self.MERGE_MAX_OFFSET)
                    and (abs(x1_last - x1_curr) <= self.MERGE_MAX_OFFSET)
                    and (abs(y0_curr - y1_last) <= self.MERGE_MAX_GAP)
                )
                if is_vertically_aligned_and_close:
                    can_merge = True

            if not can_merge and full_height_curr:
                is_horizontally_aligned_and_close = (
                    (abs(y0_last - y0_curr) <= self.MERGE_MAX_OFFSET)
                    and (abs(y1_last - y1_curr) <= self.MERGE_MAX_OFFSET)
                    and (abs(x0_curr - x1_last) <= self.MERGE_MAX_GAP)
                )
                if is_horizontally_aligned_and_close:
                    can_merge = True

            if can_merge:
                current_page_merged_list[-1] = [
                    min(x0_curr, x0_last),
                    min(y0_curr, y0_last),
                    max(x1_curr, x1_last),
                    max(y1_curr, y1_last),
                    imgid_curr,
                ]
            else:
                current_page_merged_list.append(img_to_potentially_merge)

        return current_page_merged_list

    def _get_junk_image_stats_from_sampled_pages(
        self, doc: pymupdf.Document, sampled_page_indices: list[int]
    ) -> dict[str, int | list[int]]:
        doc_stats = {"junk_image_xrefs_list": []}

        if not sampled_page_indices:
            return doc_stats

        img_xrefs_all_instances_sampled = []
        page_unique_xrefs_map_sampled = defaultdict(set)

        for page_idx in sampled_page_indices:
            page_unique_xrefs_map_sampled[page_idx] = set()

            try:
                page = doc.load_page(page_idx)
                image_definitions_on_page = page.get_images(full=False)

                for img_def_tuple in image_definitions_on_page:
                    xref = img_def_tuple[0]
                    if xref == 0:
                        # Maybe a widget, stencil, transparency mask or inline image,
                        # or embedded in a form XObject
                        continue

                    page_unique_xrefs_map_sampled[page_idx].add(xref)
                    img_xrefs_all_instances_sampled.append(xref)
            except Exception as e:
                print(f"Error processing page {page_idx} with error: {e}")

        if not img_xrefs_all_instances_sampled:
            return doc_stats

        doc_stats["num_unique_image_xrefs"] = len(set(img_xrefs_all_instances_sampled))

        xrefs_counts_across_sampled_pages = Counter(
            chain.from_iterable(page_unique_xrefs_map_sampled.values())
        )

        # choose the min between total number of pages sampled & the calculated max occurence
        max_page_occurence_threshold = min(
            len(sampled_page_indices),
            max(
                self.JUNK_IMAGE_MAX_PAGE_THRESHOLD_RATIO * len(sampled_page_indices),
                self.JUNK_IMAGE_THRESHOLD_MAX_PAGES,
            ),
        )

        if len(sampled_page_indices) < self.JUNK_IMAGE_THRESHOLD_MAX_PAGES:
            current_junk_xrefs_list = []
        else:
            current_junk_xrefs_list = [
                xref
                for xref, xref_page_count in xrefs_counts_across_sampled_pages.items()
                if xref_page_count >= max_page_occurence_threshold
            ]

        doc_stats["num_junk_image_xrefs"] = len(current_junk_xrefs_list)
        doc_stats["junk_image_xrefs_list"] = current_junk_xrefs_list

        return doc_stats

    def _get_document_text_info(self, doc: pymupdf.Document) -> tuple[list[int], list[int]]:
        text_length_for_all_pages = []
        num_replacement_char_per_page = []

        for page in doc:
            page_text = page.get_text(
                "text",
                flags=pymupdf.TEXT_MEDIABOX_CLIP | pymupdf.TEXT_PRESERVE_WHITESPACE,
            )
            text_length_for_all_pages.append(len(page_text))
            num_replacement_char_per_page.append(page_text.count(self.UNICODE_REPLACEMENT_CHAR))

        return text_length_for_all_pages, num_replacement_char_per_page

    def _get_page_num_fonts(self, page: pymupdf.Page) -> int:
        all_fonts = set()

        try:
            for fi in page.get_fonts(full=True):
                if len(fi) > 3 and fi[3]:
                    all_fonts.add(fi[3])
        except Exception as e:
            print(f"Error getting num fonts on page: {e}")

        return len(all_fonts)

    def _get_page_text_info(self, page: pymupdf.Page) -> dict:
        page_char_count = page_hidden_char_count = 0
        page_text_total_area = page_hidden_total_area = 0
        page_text_box_count = page_hidden_box_count = 0

        page_area = max(1.0, page.rect.width * page.rect.height)

        for tr in page.get_texttrace():
            n = len(tr.get("chars", []))
            area = self._get_bbox_area(tr.get("bbox"))

            if tr.get("type") == 3 or tr.get("opacity", 1.0) == 0:
                page_hidden_char_count += n
                page_hidden_total_area += area
                page_hidden_box_count += 1
            else:
                page_char_count += n
                page_text_total_area += area
                page_text_box_count += 1

        return {
            "avg_text_box_length": page_text_total_area / page_text_box_count
            if page_text_box_count > 0
            else 0,
            "char_count": page_char_count,
            "text_area_ratio": page_text_total_area / page_area if page_area > 0 else 0,
            "text_box_count": page_text_box_count,
            # hidden text features
            "hidden_char_count": page_hidden_char_count,
            "hidden_text_box_count": page_hidden_box_count,
            "hidden_avg_text_box_length": page_hidden_total_area / page_hidden_box_count
            if page_hidden_box_count > 0
            else 0,
            "hidden_text_area_ratio": page_hidden_total_area / page_area if page_area > 0 else 0,
        }

    def _sample_pages_into_chunks(self, doc: pymupdf) -> list[list[int]]:
        total_pages = len(doc)
        if total_pages == 0 or self.num_pages_to_sample <= 0:
            return []

        available_idxs = list(range(total_pages))
        chunks = []

        num_chunks = -1
        if self.num_chunks == -1:
            num_chunks = total_pages // self.num_pages_to_sample + 1

        for _ in range(num_chunks):
            if len(available_idxs) == 0:
                break

            chunk_size = min(self.num_pages_to_sample, len(available_idxs))
            chunk = random.sample(available_idxs, chunk_size)

            for idx in chunk:
                available_idxs.remove(idx)

            chunks.append(chunk)

        return chunks

    def _get_page_drawings_and_vg_info(self, page: pymupdf.Page) -> dict[str, int]:
        drawings_stroke_count = vg_obj_count = 0

        try:
            drawings = page.get_cdrawings()
            vg_obj_count = len(drawings)

            for path in drawings:
                # 'items' contains sequence of path construction operators
                # like ('l', x, y) or ('c', ...)
                # A simple heuristic: count line segments or curves as strokes.
                # This is a simplification.
                for item in path.get("items", []):
                    if item[0] in ["l", "c", "q"]:  # line, curve, quadratic
                        drawings_stroke_count += 1

                if path.get("rect") or path.get("quad"):
                    if path.get("stroke_opacity", 1) > 0 and path.get(
                        "color"
                    ):  # stroked, not filled or transparent
                        drawings_stroke_count += 1
        except Exception as e:
            print(f"Error getting drawing and vector graphics info on page: {e}")

        return {
            "drawing_strokes_count": drawings_stroke_count,
            "vector_graphics_obj_count": vg_obj_count,
        }

    def _get_page_image_features(self, page: pymupdf.Page, junk_image_xrefs_sampled: list):
        total_image_instances = non_junk_image_instances = 0
        non_junk_rects_for_strip_merge = []

        try:
            image_definitions_on_page = page.get_images(full=False)
            for img_def_tuple in image_definitions_on_page:
                xref = img_def_tuple[0]

                if xref == 0:
                    continue

                image_rects_on_page = page.get_image_rects(xref, transform=False)
                total_image_instances += len(image_rects_on_page)

                if xref not in junk_image_xrefs_sampled:
                    non_junk_image_instances += len(image_rects_on_page)
                    for rect_obj in image_rects_on_page:
                        if rect_obj.is_empty or rect_obj.is_infinite:
                            continue
                        bbox_list = [rect_obj.x0, rect_obj.y0, rect_obj.x1, rect_obj.y1]
                        non_junk_rects_for_strip_merge.append(bbox_list + [xref])
        except Exception as e:
            print(f"Error getting image features on page: {e}")

        merged_strip_bboxes_original = self._heuristic_merge_image_strips_on_page(
            non_junk_rects_for_strip_merge, page.rect.width, page.rect.height
        )
        merged_strip_areas_original = [
            abs(b[2] - b[0]) * abs(b[3] - b[1]) for b in merged_strip_bboxes_original
        ]
        page_area = max(1.0, page.rect.width * page.rect.height)

        return {
            "image_count": total_image_instances,
            "non_junk_image_count": non_junk_image_instances,
            "max_merged_strip_area": max(merged_strip_areas_original) / page_area
            if merged_strip_areas_original and page_area > 0
            else 0.0,
            "bitmap_proportion": sum(merged_strip_areas_original) / page_area
            if merged_strip_areas_original and page_area > 0
            else 0.0,
        }

    def _get_sampled_page_indices(self, doc: pymupdf.Document) -> list[list[int]]:
        total_pages = len(doc)
        if total_pages == 0 or self.num_pages_to_sample <= 0:
            return []

        available_indices = list(range(total_pages))
        sampled_indices = []

        if self.num_chunks == -1:
            num_chunks = len(available_indices) // self.num_pages_to_sample + 1
        else:
            num_chunks = self.num_chunks

        for _ in range(num_chunks):
            if len(available_indices) == 0:
                break

            chunk_size = min(self.num_pages_to_sample, len(available_indices))
            chunk = random.sample(available_indices, chunk_size)

            for idx in chunk:
                available_indices.remove(idx)

            sampled_indices.append(sorted(chunk))

        return sampled_indices

    def compute_features_for_chunk(
        self, doc: pymupdf.Document, chunk: list[int]
    ) -> PDFChunkFeatures:
        text_length_for_all_pages, num_replacement_char_per_page = self._get_document_text_info(doc)

        text_lengths_for_pages_in_chunk, num_replacement_char_per_page_in_chunk = (
            [text_length_for_all_pages[idx] for idx in chunk],
            [num_replacement_char_per_page[idx] for idx in chunk],
        )

        features = PDFChunkFeatures(
            is_form=bool(doc.is_form_pdf is True),
            creator_or_producer_is_known_scanner=self._creator_or_producer_is_known_scanner(doc),
            global_garbled_text_ratio=0.0
            if sum(text_length_for_all_pages) == 0
            else sum(num_replacement_char_per_page) / sum(text_length_for_all_pages),
            garbled_text_ratio=0.0
            if sum(text_lengths_for_pages_in_chunk) == 0
            else sum(num_replacement_char_per_page_in_chunk) / sum(text_lengths_for_pages_in_chunk),
            num_pages_requested_for_sampling=len(chunk),
            # These are updated below
            num_junk_image_xrefs=0,
            num_unique_image_xrefs=0,
            num_pages_successfully_sampled=0,
            sampled_page_indices=[],
        )

        if features.num_pages_requested_for_sampling == 0:
            features.sampled_page_features = []
            return features

        # image xref stats
        junk_image_info = self._get_junk_image_stats_from_sampled_pages(doc, chunk)
        features.num_junk_image_xrefs = junk_image_info["num_junk_image_xrefs"]
        features.num_unique_image_xrefs = junk_image_info["num_unique_image_xrefs"]

        for page_idx in chunk:
            try:
                page = doc.load_page(page_idx)
                features.num_pages_successfully_sampled += 1
                features.sampled_page_indices.append(page_idx)
            except Exception as e:
                print(f"Error processing page {page_idx} with error: {e}")
                continue

            page_text_info = self._get_page_text_info(page)
            page_image_info = self._get_page_image_features(
                page, junk_image_info["junk_image_xrefs_list"]
            )
            page_drawings_and_vg_info = self._get_page_drawings_and_vg_info(page)

            page_info = PageFeatures(
                unique_font_count=self._get_page_num_fonts(page),
                char_count=page_text_info["char_count"],
                text_box_count=page_text_info["text_box_count"],
                avg_text_box_length=page_text_info["avg_text_box_length"],
                text_area_ratio=page_text_info["text_area_ratio"],
                # hidden text features
                hidden_char_count=page_text_info["hidden_char_count"],
                hidden_text_box_count=page_text_info["hidden_text_box_count"],
                hidden_avg_text_box_length=page_text_info["hidden_avg_text_box_length"],
                hidden_text_area_ratio=page_text_info["hidden_text_area_ratio"],
                # image features
                image_count=page_image_info["image_count"],
                non_junk_image_count=page_image_info["non_junk_image_count"],
                bitmap_proportion=page_image_info["bitmap_proportion"],
                max_merged_strip_area=page_image_info["max_merged_strip_area"],
                # drawing and vector graphics features
                drawing_strokes_count=page_drawings_and_vg_info["drawing_strokes_count"],
                vector_graphics_obj_count=page_drawings_and_vg_info["vector_graphics_obj_count"],
            )

            features.sampled_page_features.append(page_info)

        return features

    def extract_all_features(self, doc: pymupdf.Document) -> PDFChunkFeatures:
        sampled_page_indices_to_try = self._get_sampled_page_indices(doc)
        return [
            self.compute_features_for_chunk(doc, chunk) for chunk in sampled_page_indices_to_try
        ]

    def run(self, doc_bytes: bytes) -> tuple[PDFChunkFeatures, int]:
        pymupdf_doc = None

        try:
            pymupdf_doc = pymupdf.open(stream=io.BytesIO(doc_bytes), filetype="pdf")
            main_features = self.extract_all_features(pymupdf_doc)
            n_pages = len(pymupdf_doc)
            return main_features, n_pages
        finally:
            if pymupdf_doc:
                pymupdf_doc.close()
