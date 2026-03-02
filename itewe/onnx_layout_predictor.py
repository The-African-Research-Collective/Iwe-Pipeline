import logging
import os
from collections.abc import Iterable
from typing import Union

import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)


def post_process_object_detection_onnx(scores, labels, boxes, threshold):
    results = []
    for score, label, box in zip(scores, labels, boxes):
        results.append(
            {
                "scores": score[score > threshold],
                "labels": label[score > threshold],
                "boxes": box[score > threshold],
            }
        )
    return results


class LayoutPredictor:
    def __init__(
        self,
        artifact_path: str,
        device: str = "cpu",
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: set[str] = set(),
    ):
        self._classes_map = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title",
            11: "Document Index",
            12: "Code",
            13: "Checkbox-Selected",
            14: "Checkbox-Unselected",
            15: "Form",
            16: "Key-Value Region",
        }
        self._black_classes = blacklist_classes
        self._threshold = base_threshold
        self._image_size = 640
        self._num_threads = num_threads

        path = os.environ.get("LAYOUT_ONNX_PATH", "models/heron/heron.onnx")
        self._init_onnx_model(path)

        _log.debug(f"LayoutPredictor settings: {self.info()}")

    def info(self) -> dict:
        return {
            "device": "cpu",
            "num_threads": self._num_threads,
            "image_size": self._image_size,
            "threshold": self._threshold,
        }

    def _init_onnx_model(self, model_path: str):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self._num_threads

        self._session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )
        _log.debug("ONNX model loaded from %s", model_path)
        _log.debug("Inputs: %s", [i.name for i in self._session.get_inputs()])
        _log.debug("Outputs: %s", [o.name for o in self._session.get_outputs()])

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Resize, normalize, and convert to NCHW float32."""
        orig_size = np.array([image.size[::-1]], dtype=np.int64)  # [H, W]
        image = image.resize((self._image_size, self._image_size), Image.BILINEAR)
        pixel_values = np.array(image, dtype=np.float32) / 255.0  # HWC
        pixel_values = np.transpose(pixel_values, (2, 0, 1))[np.newaxis, ...]  # NCHW
        return pixel_values, orig_size

    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        elif isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img).convert("RGB")
        else:
            raise TypeError("Not supported input image format")

        pixel_values, target_sizes = self._preprocess(page_img)

        ort_inputs = {
            "images": pixel_values,
            "orig_target_sizes": target_sizes,
        }

        outputs = self._session.run(None, ort_inputs)
        # Output order is: labels[0], boxes[1], scores[2]
        results = post_process_object_detection_onnx(
            outputs[2],  # scores
            outputs[0],  # labels
            outputs[1],  # boxes
            self._threshold
        )

        w, h = page_img.size
        result = results[0]
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = float(score)
            label_id = int(label_id)
            label_str = self._classes_map[label_id]

            if label_str in self._black_classes:
                continue

            bbox_float = [float(b) for b in box]
            yield {
                "l": min(w, max(0, bbox_float[0])),
                "t": min(h, max(0, bbox_float[1])),
                "r": min(w, max(0, bbox_float[2])),
                "b": min(h, max(0, bbox_float[3])),
                "label": label_str,
                "confidence": score,
            }
