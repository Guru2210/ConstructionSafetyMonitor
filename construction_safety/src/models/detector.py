"""
src/models/detector.py — YOLOv8 worker detection wrapper.

WorkerDetector wraps Ultralytics YOLOv8 for pedestrian/worker detection.
Failures are caught and logged; never raised to caller.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class Detection:
    """Single detected worker bounding box."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coords
    confidence: float
    class_id: int = 0
    class_name: str = "person"
    track_id: int | None = None  # populated by tracker


class WorkerDetector:
    """
    Wraps YOLOv8 (or similar) for worker/person detection.

    If the model file is absent, falls back to mock mode which returns
    a single dummy detection centred in the frame so the pipeline can
    still run end-to-end.
    """

    PERSON_CLASS_IDS = {0}  # COCO person = 0; extend for domain-fine-tuned models

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._mock_mode = False
        self._model: Any = None

        model_path = Path(model_path) if model_path else None

        if model_path:
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(model_path))
                person_ids = [k for k, v in self._model.names.items() if v.lower() in ("person", "worker")]
                if person_ids:
                    self.PERSON_CLASS_IDS = set(person_ids)
                logger.info(f"WorkerDetector loaded model from {model_path} with person ids {self.PERSON_CLASS_IDS}")
            except Exception:
                logger.warning(
                    f"Failed to load WorkerDetector model from {model_path}:\n"
                    + traceback.format_exc()
                )
                self._mock_mode = True
        else:
            logger.warning(
                f"WorkerDetector: model file not found at {model_path!r} — "
                "running in MOCK mode"
            )
            self._mock_mode = True

    @property
    def is_mock(self) -> bool:
        return self._mock_mode

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a frame (H×W×C BGR numpy array).

        Returns list of Detection objects (no track_id yet — tracker assigns those).
        """
        if self._mock_mode:
            return self._mock_detect(frame)
        try:
            return self._real_detect(frame)
        except Exception:
            logger.warning(
                "WorkerDetector.detect() failed, returning empty:\n"
                + traceback.format_exc()
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _real_detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=list(self.PERSON_CLASS_IDS),
            verbose=False,
            device=self.device,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                detections.append(
                    Detection(
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=result.names.get(cls_id, "person"),
                    )
                )
        return detections

    def _mock_detect(self, frame: np.ndarray) -> list[Detection]:
        """Return 1-2 plausible mock detections centred in the frame."""
        import random

        h, w = frame.shape[:2]
        detections = []
        n = random.randint(1, 2)
        for i in range(n):
            cx = random.uniform(0.2, 0.8) * w
            cy = random.uniform(0.2, 0.8) * h
            bw = random.uniform(0.08, 0.15) * w
            bh = random.uniform(0.25, 0.45) * h
            detections.append(
                Detection(
                    bbox=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                    confidence=round(random.uniform(0.55, 0.90), 3),
                    class_id=0,
                    class_name="person",
                )
            )
        return detections
