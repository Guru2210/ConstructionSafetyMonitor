"""
src/pipeline/inference.py — Main inference orchestrator.

InferencePipeline.process_frame() drives the full model cascade:
  1. Worker detection (YOLOv8)
  2. Pose estimation per worker + ROI derivation (YOLOv8-pose)
  3. PPE classification against each ROI (ResNet-18 heads)
  4. ByteTracker assignment
  → FrameResult
"""
from __future__ import annotations

import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.config import settings
from src.models.detector import Detection, WorkerDetector
from src.models.pose import PPERegions, PoseEstimator, WorkerPose
from src.models.ppe_heads import PPECheckResult, PPEClassifier
from src.models.tracker import WorkerTracker


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WorkerResult:
    """Full analysis result for a single tracked worker in one frame."""

    track_id: int | None
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float

    pose: WorkerPose | None = None
    ppe_regions: PPERegions | None = None
    pose_quality: str = "unknown"  # "good" | "partial" | "poor" | "unknown"

    helmet: PPECheckResult | None = None
    vest: PPECheckResult | None = None
    harness: PPECheckResult | None = None
    gloves: PPECheckResult | None = None
    boots: PPECheckResult | None = None
    goggles: PPECheckResult | None = None

    zone_name: str | None = None
    zone_type: str | None = None
    violations: list[Any] = field(default_factory=list)  # populated by ComplianceEngine


@dataclass
class FrameResult:
    """Output of processing one video frame."""

    frame_id: str
    site_id: str
    workers: list[WorkerResult]
    processing_ms: float
    frame_shape: tuple[int, int] = (0, 0)  # (height, width)


# ---------------------------------------------------------------------------
# Helper: crop a bounding box from a frame
# ---------------------------------------------------------------------------

def _crop_roi(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Safe crop — returns empty array for degenerate bboxes."""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2].copy()


def _crop_band_poly(
    frame: np.ndarray,
    polygon: list[tuple[float, float]],
) -> np.ndarray:
    """
    Crop the bounding box of a diagonal band polygon from frame.
    We use the AABB of the polygon for simplicity, as the ResNet head
    is invariant to exact masking.
    """
    if not polygon:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return _crop_roi(frame, (min(xs), min(ys), max(xs), max(ys)))


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """
    Orchestrates the full model cascade for one frame.
    """

    def __init__(self) -> None:
        self._detector = WorkerDetector(
            model_path=settings.detector_model_path,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
        )
        self._pose_estimator = PoseEstimator(
            model_path=settings.pose_model_path,
            keypoint_conf_threshold=settings.keypoint_conf_threshold,
        )
        self._ppe_classifier = PPEClassifier(
            helmet_model_path=settings.helmet_model_path,
            vest_model_path=settings.vest_model_path,
            harness_model_path=settings.harness_model_path,
            gloves_model_path=settings.gloves_model_path,
            boots_model_path=settings.boots_model_path,
            goggles_model_path=settings.goggles_model_path,
            helmet_conf_threshold=settings.helmet_conf_threshold,
            vest_conf_threshold=settings.vest_conf_threshold,
            harness_conf_threshold=settings.harness_conf_threshold,
            gloves_conf_threshold=settings.gloves_conf_threshold,
            boots_conf_threshold=settings.boots_conf_threshold,
            goggles_conf_threshold=settings.goggles_conf_threshold,
            vest_coverage_threshold=settings.vest_coverage_threshold,
        )
        self._tracker = WorkerTracker()
        self._executor = ThreadPoolExecutor(max_workers=4)

        logger.info(
            "InferencePipeline initialised | "
            f"detector={'mock' if self._detector.is_mock else 'real'} | "
            f"pose={'mock' if self._pose_estimator.is_mock else 'real'} | "
            f"ppe_heads={self._ppe_classifier.models_status}"
        )

    @property
    def models_status(self) -> dict[str, Any]:
        return {
            "detector": "mock" if self._detector.is_mock else "loaded",
            "pose": "mock" if self._pose_estimator.is_mock else "loaded",
            "tracker": "mock" if self._tracker.is_mock else "loaded",
            **{f"ppe_{k}": v for k, v in self._ppe_classifier.models_status.items()},
        }

    async def process_frame(
        self,
        frame: np.ndarray,
        site_id: str,
        frame_id: str | None = None,
    ) -> FrameResult:
        """
        Run the full detection → pose → PPE cascade on one frame.

        Parameters
        ----------
        frame    : BGR numpy array (H×W×3)
        site_id  : identifier for the site (used for zone lookup)
        frame_id : optional unique identifier string for this frame

        Returns
        -------
        FrameResult with workers populated (zone/violations filled by ComplianceEngine)
        """
        t0 = time.perf_counter()
        frame_id = frame_id or str(int(t0 * 1000))
        h, w = frame.shape[:2]

        try:
            # ----------------------------------------------------------
            # Step 1: Detect workers
            # ----------------------------------------------------------
            loop = asyncio.get_running_loop()
            detections: list[Detection] = await loop.run_in_executor(
                self._executor, self._detector.detect, frame
            )

            # ----------------------------------------------------------
            # Step 2: Track (assign persistent IDs)
            # ----------------------------------------------------------
            tracked: list[Detection] = await loop.run_in_executor(
                self._executor, self._tracker.update, detections, frame
            )

            # ----------------------------------------------------------
            # Steps 3+4: Pose + PPE per worker (concurrent via gather)
            # ----------------------------------------------------------
            worker_results = []
            for det in tracked:
                wr = await self._analyse_worker(frame, det, loop)
                worker_results.append(wr)

        except Exception:
            logger.warning(
                "InferencePipeline.process_frame() failed:\n" + traceback.format_exc()
            )
            worker_results = []

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FrameResult(
            frame_id=frame_id,
            site_id=site_id,
            workers=worker_results,
            processing_ms=round(elapsed_ms, 2),
            frame_shape=(h, w),
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _analyse_worker(
        self,
        frame: np.ndarray,
        detection: Detection,
        loop: asyncio.AbstractEventLoop,
    ) -> WorkerResult:
        """Run pose + PPE for a single worker detection."""
        worker = WorkerResult(
            track_id=detection.track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
        )
        try:
            # Pose estimation (CPU-bound → executor)
            worker_crop = _crop_roi(frame, detection.bbox)
            pose: WorkerPose = await loop.run_in_executor(
                self._executor,
                self._pose_estimator.detect_pose,
                worker_crop,
                detection.bbox,
            )
            worker.pose = pose
            worker.pose_quality = pose.pose_quality

            # ROI derivation (fast, can run sync)
            regions: PPERegions = self._pose_estimator.derive_rois(
                pose, frame_shape=(frame.shape[0], frame.shape[1])
            )
            worker.ppe_regions = regions

            # PPE checks (strictly sequential to prevent OOM/cache thrashing)
            helmet_crop = _crop_roi(frame, regions.helmet_roi.to_xyxy()) if regions.helmet_roi else np.zeros((1, 1, 3), dtype=np.uint8)
            worker.helmet = await loop.run_in_executor(
                self._executor, self._ppe_classifier.check_helmet, helmet_crop
            )
            worker.vest = await loop.run_in_executor(
                self._executor, self._check_vest_wrapper, frame, regions
            )
            worker.harness = await loop.run_in_executor(
                self._executor, self._check_harness_wrapper, frame, regions
            )
            worker.gloves = await loop.run_in_executor(
                self._executor, self._check_gloves_wrapper, frame, regions
            )
            worker.boots = await loop.run_in_executor(
                self._executor, self._check_boots_wrapper, frame, regions
            )
            worker.goggles = await loop.run_in_executor(
                self._executor, self._check_goggles_wrapper, frame, regions
            )
        except Exception:
            logger.warning(
                f"_analyse_worker failed for track {detection.track_id}:\n"
                + traceback.format_exc()
            )

        return worker

    def _check_vest_wrapper(
        self, frame: np.ndarray, regions: PPERegions
    ) -> PPECheckResult:
        vest_crop = _crop_roi(frame, regions.vest_roi.to_xyxy()) if regions.vest_roi else np.zeros((1, 1, 3), dtype=np.uint8)
        left_crop = _crop_roi(frame, regions.vest_left_half.to_xyxy()) if regions.vest_left_half else None
        right_crop = _crop_roi(frame, regions.vest_right_half.to_xyxy()) if regions.vest_right_half else None
        return self._ppe_classifier.check_vest(vest_crop, left_crop, right_crop)

    def _check_harness_wrapper(
        self, frame: np.ndarray, regions: PPERegions
    ) -> PPECheckResult:
        left_crop = _crop_band_poly(frame, regions.harness_left_band) if regions.harness_left_band else None
        right_crop = _crop_band_poly(frame, regions.harness_right_band) if regions.harness_right_band else None
        return self._ppe_classifier.check_harness(left_crop, right_crop)

    def _check_gloves_wrapper(
        self, frame: np.ndarray, regions: PPERegions
    ) -> PPECheckResult:
        left_crop = _crop_roi(frame, regions.left_wrist_roi.to_xyxy()) if regions.left_wrist_roi else None
        right_crop = _crop_roi(frame, regions.right_wrist_roi.to_xyxy()) if regions.right_wrist_roi else None
        return self._ppe_classifier.check_gloves(left_crop, right_crop)

    def _check_boots_wrapper(
        self, frame: np.ndarray, regions: PPERegions
    ) -> PPECheckResult:
        left_crop = _crop_roi(frame, regions.left_boot_roi.to_xyxy()) if regions.left_boot_roi else None
        right_crop = _crop_roi(frame, regions.right_boot_roi.to_xyxy()) if regions.right_boot_roi else None
        return self._ppe_classifier.check_boots(left_crop, right_crop)

    def _check_goggles_wrapper(
        self, frame: np.ndarray, regions: PPERegions
    ) -> PPECheckResult:
        crop = _crop_roi(frame, regions.goggles_roi.to_xyxy()) if regions.goggles_roi else None
        return self._ppe_classifier.check_goggles(crop)