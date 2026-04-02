"""
src/models/pose.py — YOLOv8-pose wrapper + ROI derivation.

PoseEstimator:
  - detect_pose(worker_crop, worker_bbox) -> WorkerPose  [17 COCO kp in frame coords]
  - derive_rois(pose) -> PPERegions                      [anatomical ROIs for each PPE item]
"""
from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# COCO keypoint indices
# ---------------------------------------------------------------------------
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class WorkerPose:
    """17 COCO keypoints in original frame coordinates."""

    keypoints: list[Keypoint]  # length 17
    pose_quality: str = "good"  # "good" | "partial" | "poor"

    def kp(self, idx: int) -> Keypoint:
        return self.keypoints[idx]

    def is_valid(self, idx: int, threshold: float = 0.5) -> bool:
        return self.keypoints[idx].confidence >= threshold


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def to_xywh(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.width, self.height

    def clamp(self, w: int, h: int) -> "BoundingBox":
        return BoundingBox(
            x1=max(0.0, self.x1),
            y1=max(0.0, self.y1),
            x2=min(float(w), self.x2),
            y2=min(float(h), self.y2),
        )


@dataclass
class PPERegions:
    """All anatomical ROIs derived from pose keypoints."""

    # Head
    helmet_roi: BoundingBox | None = None

    # Torso
    vest_roi: BoundingBox | None = None
    vest_left_half: BoundingBox | None = None
    vest_right_half: BoundingBox | None = None

    # Harness diagonal bands (stored as polygon 4-points)
    harness_left_band: list[tuple[float, float]] | None = None   # shoulder→hip diagonal
    harness_right_band: list[tuple[float, float]] | None = None  # shoulder→hip diagonal

    # Wrists (for gloves)
    left_wrist_roi: BoundingBox | None = None
    right_wrist_roi: BoundingBox | None = None

    # Boots
    left_boot_roi: BoundingBox | None = None
    right_boot_roi: BoundingBox | None = None

    # Goggles
    goggles_roi: BoundingBox | None = None


class PoseEstimator:
    """
    Wraps YOLOv8-pose for keypoint detection.

    Falls back to mock mode if weights absent — mock returns plausible
    keypoints that derive to non-None ROIs so downstream code can run.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        keypoint_conf_threshold: float = 0.5,
    ) -> None:
        self.device = device
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self._mock_mode = False
        self._model: Any = None

        model_path = Path(model_path) if model_path else None
        if model_path:
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(model_path))
                logger.info(f"PoseEstimator loaded model from {model_path}")
            except Exception:
                logger.warning(
                    "PoseEstimator model load failed:\n" + traceback.format_exc()
                )
                self._mock_mode = True
        else:
            logger.warning(
                f"PoseEstimator: model not found at {model_path!r} — MOCK mode"
            )
            self._mock_mode = True

    @property
    def is_mock(self) -> bool:
        return self._mock_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_pose(
        self,
        worker_crop: np.ndarray,
        worker_bbox: tuple[float, float, float, float],
    ) -> WorkerPose:
        """
        Run pose estimation on a worker crop and translate results to
        original frame coordinates using worker_bbox offset.

        Parameters
        ----------
        worker_crop : np.ndarray  (H×W×C BGR)
        worker_bbox : (x1, y1, x2, y2) in original frame

        Returns
        -------
        WorkerPose with 17 COCO keypoints in frame coordinates
        """
        if self._mock_mode:
            return self._mock_pose(worker_bbox)
        try:
            return self._real_pose(worker_crop, worker_bbox)
        except Exception:
            logger.warning("PoseEstimator.detect_pose failed:\n" + traceback.format_exc())
            return self._mock_pose(worker_bbox)

    def derive_rois(
        self,
        pose: WorkerPose,
        frame_shape: tuple[int, int] | None = None,
    ) -> PPERegions:
        """
        Derive PPE inspection regions from COCO keypoints.

        Parameters
        ----------
        pose        : WorkerPose
        frame_shape : (height, width) of original frame — used for clamping
        """
        regions = PPERegions()
        kp = pose.keypoints
        kct = self.keypoint_conf_threshold

        # ------------------------------------------------------------------ #
        # Helmet ROI — derived from available head keypoints and shoulders    #
        # ------------------------------------------------------------------ #
        nose = kp[KP_NOSE]
        l_ear = kp[KP_LEFT_EAR]
        r_ear = kp[KP_RIGHT_EAR]
        l_eye = kp[KP_LEFT_EYE]
        r_eye = kp[KP_RIGHT_EYE]
        l_sh = kp[KP_LEFT_SHOULDER]
        r_sh = kp[KP_RIGHT_SHOULDER]

        head_kps = [k for k in (nose, l_ear, r_ear, l_eye, r_eye) if k.confidence >= kct]

        if len(head_kps) > 0:
            # Estimate a robust "head center" and "head size"
            cx = sum(k.x for k in head_kps) / len(head_kps)
            cy = sum(k.y for k in head_kps) / len(head_kps)
            
            head_size = 40.0  # default minimum padding
            
            if l_sh.confidence >= kct and r_sh.confidence >= kct:
                # Shoulders provide a very stable reference for worker scale
                shoulder_width = abs(r_sh.x - l_sh.x)
                head_size = max(30.0, shoulder_width * 0.45)
            elif len(head_kps) >= 2:
                xs = [k.x for k in head_kps]
                ys = [k.y for k in head_kps]
                dist = max(max(xs) - min(xs), max(ys) - min(ys))
                head_size = max(30.0, dist * 1.5)

            # Crop region around the head center (cx, cy)
            # We want more space ABOVE the head center (for the helmet) and less below
            x1 = cx - head_size * 1.4
            x2 = cx + head_size * 1.4
            y1 = cy - head_size * 2.2
            y2 = cy + head_size * 1.0

            roi = BoundingBox(x1, y1, x2, y2)
            if frame_shape:
                roi = roi.clamp(frame_shape[1], frame_shape[0])
            if roi.width > 5 and roi.height > 5:
                regions.helmet_roi = roi

        # ------------------------------------------------------------------ #
        # Vest ROI — full torso from shoulders to hips                        #
        # ------------------------------------------------------------------ #
        l_sh = kp[KP_LEFT_SHOULDER]
        r_sh = kp[KP_RIGHT_SHOULDER]
        l_hip = kp[KP_LEFT_HIP]
        r_hip = kp[KP_RIGHT_HIP]

        if all(k.confidence >= kct for k in [l_sh, r_sh, l_hip, r_hip]):
            pad = 10
            x1 = min(l_sh.x, r_sh.x, l_hip.x, r_hip.x) - pad
            y1 = min(l_sh.y, r_sh.y) - pad
            x2 = max(l_sh.x, r_sh.x, l_hip.x, r_hip.x) + pad
            y2 = max(l_hip.y, r_hip.y) + pad

            vest = BoundingBox(x1, y1, x2, y2)
            if frame_shape:
                vest = vest.clamp(frame_shape[1], frame_shape[0])
            if vest.width > 10 and vest.height > 10:
                regions.vest_roi = vest
                cx = vest.center_x
                regions.vest_left_half = BoundingBox(vest.x1, vest.y1, cx, vest.y2)
                regions.vest_right_half = BoundingBox(cx, vest.y1, vest.x2, vest.y2)

        # ------------------------------------------------------------------ #
        # Harness diagonal bands                                               #
        # ------------------------------------------------------------------ #
        # Left band: left_shoulder (5) → right_hip (12)  40px wide
        # Right band: right_shoulder (6) → left_hip (11)  40px wide
        if l_sh.confidence >= kct and r_hip.confidence >= kct:
            regions.harness_left_band = self._diagonal_band(
                (l_sh.x, l_sh.y), (r_hip.x, r_hip.y), width=40
            )

        if r_sh.confidence >= kct and l_hip.confidence >= kct:
            regions.harness_right_band = self._diagonal_band(
                (r_sh.x, r_sh.y), (l_hip.x, l_hip.y), width=40
            )

        # ------------------------------------------------------------------ #
        # Wrist ROIs — 30px radius circles (stored as bounding square)        #
        # ------------------------------------------------------------------ #
        l_wr = kp[KP_LEFT_WRIST]
        r_wr = kp[KP_RIGHT_WRIST]
        r = 30.0

        if l_wr.confidence >= kct:
            wr_roi = BoundingBox(l_wr.x - r, l_wr.y - r, l_wr.x + r, l_wr.y + r)
            if frame_shape:
                wr_roi = wr_roi.clamp(frame_shape[1], frame_shape[0])
            regions.left_wrist_roi = wr_roi

        if r_wr.confidence >= kct:
            wr_roi = BoundingBox(r_wr.x - r, r_wr.y - r, r_wr.x + r, r_wr.y + r)
            if frame_shape:
                wr_roi = wr_roi.clamp(frame_shape[1], frame_shape[0])
            regions.right_wrist_roi = wr_roi

        # ------------------------------------------------------------------ #
        # Boots ROIs                                                         #
        # ------------------------------------------------------------------ #
        l_ank = kp[KP_LEFT_ANKLE]
        r_ank = kp[KP_RIGHT_ANKLE]
        bw, bhu, bhd = 35.0, 45.0, 20.0

        if l_ank.confidence >= kct:
            bt_roi = BoundingBox(l_ank.x - bw, l_ank.y - bhu, l_ank.x + bw, l_ank.y + bhd)
            if frame_shape:
                bt_roi = bt_roi.clamp(frame_shape[1], frame_shape[0])
            regions.left_boot_roi = bt_roi

        if r_ank.confidence >= kct:
            bt_roi = BoundingBox(r_ank.x - bw, r_ank.y - bhu, r_ank.x + bw, r_ank.y + bhd)
            if frame_shape:
                bt_roi = bt_roi.clamp(frame_shape[1], frame_shape[0])
            regions.right_boot_roi = bt_roi

        # ------------------------------------------------------------------ #
        # Goggles ROI                                                        #
        # ------------------------------------------------------------------ #
        eyes = [k for k in (kp[KP_LEFT_EYE], kp[KP_RIGHT_EYE], kp[KP_NOSE]) if k.confidence >= kct]
        if len(eyes) >= 1:
            cxe = sum(k.x for k in eyes) / len(eyes)
            cye = sum(k.y for k in eyes) / len(eyes)
            gsz = max(15.0, 24.0)
            g_roi = BoundingBox(cxe - gsz * 1.5, cye - gsz * 0.8, cxe + gsz * 1.5, cye + gsz * 1.2)
            if frame_shape:
                g_roi = g_roi.clamp(frame_shape[1], frame_shape[0])
            regions.goggles_roi = g_roi

        return regions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _real_pose(
        self,
        worker_crop: np.ndarray,
        worker_bbox: tuple[float, float, float, float],
    ) -> WorkerPose:
        results = self._model.predict(
            worker_crop,
            verbose=False,
            device=self.device,
        )
        x_off, y_off = worker_bbox[0], worker_bbox[1]
        keypoints: list[Keypoint] = []

        if results and results[0].keypoints is not None:
            kps_data = results[0].keypoints.data  # tensor (N, 17, 3)
            if len(kps_data) > 0:
                # Take first person detected in crop
                kp_arr = kps_data[0].cpu().numpy()  # (17, 3) x, y, conf
                for kp in kp_arr:
                    keypoints.append(
                        Keypoint(
                            x=float(kp[0]) + x_off,
                            y=float(kp[1]) + y_off,
                            confidence=float(kp[2]),
                        )
                    )
        # Pad to 17 if needed
        while len(keypoints) < 17:
            keypoints.append(Keypoint(x=0.0, y=0.0, confidence=0.0))

        return WorkerPose(
            keypoints=keypoints,
            pose_quality=self._score_pose_quality(keypoints),
        )

    def _mock_pose(
        self, worker_bbox: tuple[float, float, float, float]
    ) -> WorkerPose:
        """Generate plausible mock keypoints from bbox geometry."""
        import random

        x1, y1, x2, y2 = worker_bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2

        def kp(rx: float, ry: float, conf: float | None = None) -> Keypoint:
            c = conf if conf is not None else round(random.uniform(0.6, 0.95), 3)
            return Keypoint(x=x1 + rx * w, y=y1 + ry * h, confidence=c)

        keypoints = [
            kp(0.50, 0.05),   # 0 nose
            kp(0.45, 0.04),   # 1 left_eye
            kp(0.55, 0.04),   # 2 right_eye
            kp(0.38, 0.06),   # 3 left_ear
            kp(0.62, 0.06),   # 4 right_ear
            kp(0.30, 0.20),   # 5 left_shoulder
            kp(0.70, 0.20),   # 6 right_shoulder
            kp(0.20, 0.35),   # 7 left_elbow
            kp(0.80, 0.35),   # 8 right_elbow
            kp(0.15, 0.50),   # 9 left_wrist
            kp(0.85, 0.50),   # 10 right_wrist
            kp(0.35, 0.55),   # 11 left_hip
            kp(0.65, 0.55),   # 12 right_hip
            kp(0.35, 0.75),   # 13 left_knee
            kp(0.65, 0.75),   # 14 right_knee
            kp(0.35, 0.95),   # 15 left_ankle
            kp(0.65, 0.95),   # 16 right_ankle
        ]
        return WorkerPose(keypoints=keypoints, pose_quality="good")

    @staticmethod
    def _score_pose_quality(keypoints: list[Keypoint]) -> str:
        valid = sum(1 for kp in keypoints if kp.confidence >= 0.5)
        ratio = valid / 17
        if ratio >= 0.8:
            return "good"
        elif ratio >= 0.5:
            return "partial"
        return "poor"

    @staticmethod
    def _diagonal_band(
        pt1: tuple[float, float],
        pt2: tuple[float, float],
        width: float = 40,
    ) -> list[tuple[float, float]]:
        """
        Compute 4 corners of a rectangle of `width` px centred on the
        line from pt1 to pt2. Returns list of (x, y) tuples (clockwise).
        """
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return []

        # Unit perpendicular
        px = -dy / length * (width / 2)
        py = dx / length * (width / 2)

        return [
            (pt1[0] + px, pt1[1] + py),
            (pt1[0] - px, pt1[1] - py),
            (pt2[0] - px, pt2[1] - py),
            (pt2[0] + px, pt2[1] + py),
        ]
