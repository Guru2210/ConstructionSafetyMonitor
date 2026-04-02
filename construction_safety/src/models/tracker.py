"""
src/models/tracker.py — ByteTracker wrapper using supervision library.

Thin interface: update(detections, frame) -> detections with track_id populated.
"""
from __future__ import annotations

import traceback
from typing import Any

import numpy as np
from loguru import logger

from src.models.detector import Detection


class WorkerTracker:
    """
    Wraps supervision's ByteTracker to assign persistent track IDs
    to detections across frames.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        self._tracker: Any = None
        self._mock_mode = False

        try:
            import supervision as sv

            if hasattr(sv, 'ByteTrack'):
                self._tracker = sv.ByteTrack(
                    track_activation_threshold=track_activation_threshold,
                    lost_track_buffer=lost_track_buffer,
                    minimum_matching_threshold=minimum_matching_threshold,
                    frame_rate=frame_rate,
                )
            else:
                self._tracker = sv.ByteTracker(
                    track_activation_threshold=track_activation_threshold,
                    lost_track_buffer=lost_track_buffer,
                    minimum_matching_threshold=minimum_matching_threshold,
                    frame_rate=frame_rate,
                )
            logger.info("WorkerTracker: ByteTracker initialised")
        except Exception:
            logger.warning(
                "WorkerTracker: supervision ByteTracker init failed — MOCK mode:\n"
                + traceback.format_exc()
            )
            self._mock_mode = True
            self._next_id = 1

    @property
    def is_mock(self) -> bool:
        return self._mock_mode

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> list[Detection]:
        """
        Assign/update track IDs for each detection.

        Returns updated Detection list (same objects, track_id populated).
        """
        if not detections:
            return []

        if self._mock_mode:
            return self._mock_update(detections)

        try:
            return self._real_update(detections, frame)
        except Exception:
            logger.warning(
                "WorkerTracker.update() failed, keeping None track_ids:\n"
                + traceback.format_exc()
            )
            return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _real_update(
        self, detections: list[Detection], frame: np.ndarray
    ) -> list[Detection]:
        import supervision as sv
        import numpy as np_

        # Build sv.Detections from our Detection list
        xyxy = np_.array([d.bbox for d in detections], dtype=np_.float32)
        confs = np_.array([d.confidence for d in detections], dtype=np_.float32)
        class_ids = np_.array([d.class_id for d in detections], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=class_ids,
        )

        # Run tracker
        tracked = self._tracker.update_with_detections(sv_dets)

        # Map tracked results back. tracker returns sv.Detections with tracker_id
        updated: list[Detection] = []
        if tracked.tracker_id is None:
            return detections

        for i, tid in enumerate(tracked.tracker_id):
            if i < len(detections):
                det = detections[i]
                det.track_id = int(tid)
                updated.append(det)

        return updated

    def _mock_update(self, detections: list[Detection]) -> list[Detection]:
        """Assign sequential stable track IDs (mock mode)."""
        for det in detections:
            if det.track_id is None:
                det.track_id = self._next_id
                self._next_id += 1
        return detections
