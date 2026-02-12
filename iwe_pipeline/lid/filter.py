from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

from iwe_pipeline.lid.fasttext import LanguageStrategy, LocalLID, OpenLID, OpenLIDv2


class AfricanLanguageFilter(BaseFilter):
    name = "african_language_filter"
    backend_map: dict[str, type[OpenLID]] = {
        "openlid": OpenLID,
        "openlidv2": OpenLIDv2,
        "locallid": LocalLID,
    }

    def __init__(
        self,
        lid_model: Literal["locallid", "openlid", "openlidv2"] = "openlidv2",
        lid_model_path: str | None = None,
        languages: list[str] | str | None = None,
        strategy: LanguageStrategy = LanguageStrategy.ALAP,
        language_threshold: float = 0.85,
        keep_top_predictions_threshold: float = -1,
        batch_size: int = 1,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer, batch_size=batch_size)

        self.backend = lid_model
        self.language_threshold = language_threshold
        self.keep_top_predictions_threshold = keep_top_predictions_threshold
        self.strategy = strategy

        if isinstance(languages, str):
            languages = [languages]

        self.languages = set(languages) if languages else languages
        self.model = self.backend_map[lid_model](
            model_path=lid_model_path, languages=languages, k=5, strategy=strategy
        )

        if self.languages is None:
            # model.languages may be updated depending on strategy so we sync
            self.languages = self.model.languages

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter: True if a sample should be KEPT, false if it should be REMOVED.
        """
        top_prediction, predictions = self.model.predict(doc)

        doc.metadata["language"] = top_prediction

        if self.keep_top_predictions_threshold != -1:
            doc.metadata["top_language_pairs"] = sorted(
                [
                    pred
                    for pred in predictions
                    if pred["score"] > self.keep_top_predictions_threshold
                ],
                key=lambda x: x["score"],
                reverse=True,
            )

        return (
            # no language list and top_prediction is above threshold
            (self.languages is None and top_prediction["score"] > self.language_threshold)
            # any prediction is in language list and above threshold
            or (
                self.languages
                and any(pred["score"] > self.language_threshold for pred in predictions)
            )
        )
