"""
Modified from OCR predictor block in finepdfs repo:
https://github.com/huggingface/finepdfs/blob/main/blocks/predictor/ocr_predictor.py
"""

import io

import pandas as pd
import pymupdf
from datatrove.pipeline.writers.disk_base import DiskWriter
from xgboost import XGBClassifier

from itewe.classifier.base import BaseMediaExtractor
from itewe.classifier.ocr.feature_extractor import PDFFeatureExtractor
from itewe.utils.pdf_utils import check_is_corrupted_or_encrypted


class OCRClassifier(BaseMediaExtractor):
    name = "📄 OCRClassifier"

    def __init__(
        self,
        path_to_model: str,
        exclusion_writer: DiskWriter,
        exclude_failed: bool = True,
        num_pages_to_sample: int = 8,
        timeout: int = 60,
        **kwargs,
    ):
        super().__init__(
            timeout=timeout, exclusion_writer=exclusion_writer, exclude_failed=exclude_failed
        )
        self.feature_extractor = PDFFeatureExtractor(
            num_chunks=1, num_pages_to_sample=num_pages_to_sample
        )
        self.path_to_model = path_to_model
        self.num_pages_to_sample = num_pages_to_sample
        self._model = None
        self._model_feature_names = None

    @property
    def model(self):
        if self._model is None:
            self._model = XGBClassifier()
            self._model.load_model(self.path_to_model)
            self._model_feature_names = self._model.get_booster().feature_names
        return self._model

    def extract(self, media_bytes: bytes | None, document_metadata: dict):
        if not media_bytes:
            return "<no content>", {"extraction_error": "Media bytes are None"}

        try:
            pymupdf_doc = pymupdf.open(stream=io.BytesIO(media_bytes), filetype="pdf")

            if check_is_corrupted_or_encrypted(pymupdf_doc):
                return "<no content>", {"extraction_error": "Document is corrupted or encrypted"}

            features = self.feature_extractor.extract_all_features(
                pymupdf_doc, flatten=True, resample=True
            )

            ocr_prob = self.model.predict_proba(
                pd.DataFrame.from_dict(features)[self._model_feature_names]
            )[0][1]

            return "<no content>", {
                "ocr_prob": float(ocr_prob),
                "is_form": features[0]["is_form"],
                "garbled_text_ratio": features[0]["garbled_text_ratio"],
                "is_encrypted": bool(pymupdf_doc.is_encrypted),
                "needs_password": bool(pymupdf_doc.needs_pass),
                "num_pages": int(pymupdf_doc.page_count),
            }
        except Exception as e:
            return "<no content>", {"extraction_error": str(e)}
