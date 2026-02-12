import gzip
import re
import shutil
from collections.abc import Callable
from enum import Enum
from typing import Final, TypedDict

from datatrove.data import Document
from datatrove.io import cached_asset_path_or_download, safely_create_file
from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.lid import FastTextLID

from iwe_pipeline.lid.cleaning_utils import get_nonprintable_char_handler

PIVOT_LANGUAGES: Final = frozenset(["eng_Latn", "fra_Latn", "por_Latn"])


class LanguageMetadata(TypedDict):
    language: str
    lid: str
    score: float


class LanguageStrategy(Enum):
    CUSTOM = "custom"
    AL = "african_languages"
    ALAP = "african_languages_and_pivot"


class OpenLID(FastTextLID):
    NAME = "openlid"
    MODEL_SUBFOLDER = "openlid"
    MODEL_URL = "https://data.statmt.org/lid/lid201-model.bin.gz"
    WHITESPACE_REGEX = re.compile(r"\s+")
    SUPPORTED_AFRICAN_LANGUAGES: Final = frozenset(
        [
            "afr_Latn",
            "amh_Ethi",
            "bam_Latn",
            "bem_Latn",
            "cjk_Latn",
            "dik_Latn",
            "ewe_Latn",
            "fon_Latn",
            "fuv_Latn",
            "gaz_Latn",
            "hau_Latn",
            "ibo_Latn",
            "kab_Latn",
            "kik_Latn",
            "kin_Latn",
            "kmb_Latn",
            "knc_Latn",
            "kon_Latn",
            "lua_Latn",
            "luo_Latn",
            "lug_Latn",
            "mos_Latn",
            "nso_Latn",
            "nya_Latn",
            "plt_Latn",
            "run_Latn",
            "sna_Latn",
            "som_Latn",
            "sot_Latn",
            "ssw_Latn",
            "swa_Latn",
            "taq_Latn",
            "taq_Tfng",
            "tir_Ethi",
            "tsn_Latn",
            "tso_Latn",
            "twi_Latn",
            "umb_Latn",
            "wol_Latn",
            "xho_Latn",
            "yor_Latn",
            "zul_Latn",
        ]
    )

    def __init__(
        self,
        languages: list[str] | None = None,
        k: int = -1,
        strategy: LanguageStrategy = LanguageStrategy.ALAP,
        **kwargs,
    ) -> None:
        """
        Args:
            languages (list[str]): Languages to predict
            k (int, optional): Number of top-k languages to consider
                            all languages outside of k will be considered as being predicted with 0.0
            strategy (LanguageStrategy): If languages is None, this sets languages using the strategy passed
        """
        if languages is None:
            if strategy == LanguageStrategy.AL:
                languages = self.SUPPORTED_AFRICAN_LANGUAGES
            elif strategy == LanguageStrategy.ALAP:
                languages = self.SUPPORTED_AFRICAN_LANGUAGES | PIVOT_LANGUAGES
        super().__init__(languages, k)

    def _initialize_model(self) -> None:
        check_required_dependencies("lid", [("fasttext", "fasttext-numpy2-wheel")])
        from fasttext.FastText import _FastText

        model_file = cached_asset_path_or_download(
            self.MODEL_URL,
            namespace="lid",
            subfolder=self.MODEL_SUBFOLDER,
            desc=f"fast-text language identifier model {self.NAME}",
        )

        if model_file.endswith(".gz"):
            output_path = model_file.rstrip(".gz")

            def decompress():
                with gzip.open(model_file, "rb") as f_in:
                    with open(output_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

            safely_create_file(output_path, decompress)
            self._model = _FastText(output_path)
        else:
            self._model = _FastText(model_file)

    @property
    def model(self):
        if self._model is None:
            self._initialize_model()
        return self._model

    @staticmethod
    def clean_text(text: str) -> str:
        return OpenLID.WHITESPACE_REGEX.sub(
            " ",
            text.replace("\n", " ")
            .replace("|", " ")
            .replace("--", "")
            .replace("=", "")
            .replace("- -", "")
            .replace("#", "")
            .replace("*", ""),
        )

    def predict(
        self, doc: Document
    ) -> tuple[LanguageMetadata, list[LanguageMetadata]] | list[LanguageMetadata]:
        languages, scores = self.model.predict(self.clean_text(doc.text), k=self.k)
        scores_dict = {lang.split("__")[2]: score.item() for lang, score in zip(languages, scores)}

        top_prediction = max(scores_dict.items(), key=lambda x: x[1])
        top_prediction = LanguageMetadata(
            language=top_prediction[0], lid=self.NAME, score=top_prediction[1]
        )

        if self.languages:
            predictions = [
                LanguageMetadata(
                    language=language, lid=self.NAME, score=scores_dict.get(language, 0.0)
                )
                for language in self.languages
            ]
            return top_prediction, predictions

        predictions = [
            LanguageMetadata(language=language, lid=self.NAME, score=score)
            for language, score in scores_dict.items()
        ]
        return predictions


class OpenLIDv2(OpenLID):
    MODEL_URL = "https://huggingface.co/laurievb/OpenLID-v2/resolve/main/model.bin?download=true"
    MODEL_SUBFOLDER = "openlid_v2"
    SUPPORTED_AFRICAN_LANGUAGES: Final = frozenset(
        [
            "afr_Latn",
            "amh_Ethi",
            "bam_Latn",
            "bem_Latn",
            "cjk_Latn",
            "dik_Latn",
            "ewe_Latn",
            "fon_Latn",
            "fuv_Latn",
            "gaz_Latn",
            "hau_Latn",
            "ibo_Latn",
            "kab_Latn",
            "kik_Latn",
            "kin_Latn",
            "kmb_Latn",
            "knc_Latn",
            "kon_Latn",
            "lua_Latn",
            "luo_Latn",
            "lug_Latn",
            "mos_Latn",
            "nso_Latn",
            "nya_Latn",
            "plt_Latn",
            "run_Latn",
            "sna_Latn",
            "som_Latn",
            "sot_Latn",
            "ssw_Latn",
            "swh_Latn",
            "taq_Latn",
            "taq_Tfng",
            "tir_Ethi",
            "tsn_Latn",
            "tso_Latn",
            "twi_Latn",
            "umb_Latn",
            "wol_Latn",
            "xho_Latn",
            "yor_Latn",
            "zul_Latn",
        ]
    )

    def __init__(
        self,
        languages: list[str] | None = None,
        k: int = -1,
        strategy: LanguageStrategy = LanguageStrategy.ALAP,
        **kwargs,
    ):
        check_required_dependencies("OpenLIDV2", ["cleantext", "emoji"])
        super().__init__(languages, k, strategy)

        global replace_emoji
        global cleantext_func

        from cleantext import clean as cleantext_func
        from emoji import replace_emoji

        self.npc_handler = get_nonprintable_char_handler()

    @staticmethod
    def clean_text(text: str) -> str:
        demojized_text = replace_emoji(text, replace="")
        return cleantext_func(
            demojized_text,
            clean_all=True,
            numbers=True,
            extra_spaces=True,
            stemming=False,
            stopwords=False,
            punct=True,
            lowercase=False,
        )


class LocalLID:
    def __init__(
        self,
        model_path: str,
        cleaning_func: Callable[[str], str] = OpenLIDv2.clean_text,
        k: int = -1,
        languages: list[str] | None = None,
        **kwargs,
    ):
        self.model_path = model_path
        self.clean_text = cleaning_func
        self.languages = languages
        self.k = k
        self._model = None

    @property
    def model(self):
        if self._model is None:
            check_required_dependencies("lid", [("fasttext", "fasttext-numpy2-wheel")])
            from fasttext.FastText import _FastText

            self._model = _FastText(self.model_path)
        return self._model

    def predict(
        self, doc: Document
    ) -> tuple[LanguageMetadata, list[LanguageMetadata]] | list[LanguageMetadata]:
        return OpenLIDv2.predict(self, doc)
