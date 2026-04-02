"""
src/utils/drawing.py — Frame annotation utilities.

annotate_frame(frame, report) -> np.ndarray
  Draws bounding boxes, keypoint skeletons, ROI outlines, and status labels.
  Returns annotated copy; never modifies original.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.pipeline.reporter import ViolationReport
from src.pipeline.inference import WorkerResult
from src.models.pose import WorkerPose, PPERegions


# ---------------------------------------------------------------------------
# Color palette  (BGR)
# ---------------------------------------------------------------------------
COLOR_SAFE    = (34, 197, 94)    # green
COLOR_WARNING = (234, 179, 8)    # amber
COLOR_ALERT   = (249, 115, 22)   # orange
COLOR_URGENT  = (239, 68, 68)    # red
COLOR_SKELETON= (147, 197, 253)  # light blue
COLOR_ROI     = (209, 213, 219)  # grey

# COCO skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]


def annotate_frame(
    frame: np.ndarray,
    report: ViolationReport,
) -> np.ndarray:
    """
    Annotate a frame with bounding boxes, skeleton overlays, ROI outlines,
    and status labels.

    Parameters
    ----------
    frame  : BGR numpy array (original, unmodified)
    report : ViolationReport from generate_report()

    Returns
    -------
    Annotated copy of frame
    """
    canvas = frame.copy()

    for worker in report.workers:
        color = _worker_color(worker)
        _draw_bbox(canvas, worker.bbox, color, worker)
        if worker.pose is not None:
            _draw_skeleton(canvas, worker.pose)
        if worker.ppe_regions is not None:
            _draw_roi_outlines(canvas, worker.ppe_regions)

    return canvas


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _worker_color(worker: WorkerResult) -> tuple[int, int, int]:
    if not worker.violations:
        return COLOR_SAFE
    severities = {v.severity for v in worker.violations}
    if "urgent" in severities:
        return COLOR_URGENT
    if "alert" in severities:
        return COLOR_ALERT
    return COLOR_WARNING


def _draw_bbox(
    canvas: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int],
    worker: WorkerResult,
) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)

    # Label: "W3 | no helmet | 94%"
    track_label = f"W{worker.track_id}" if worker.track_id is not None else "W?"
    if worker.violations:
        items = ", ".join(f"no {v.ppe_item}" for v in worker.violations[:2])
        conf_pct = round(max(v.confidence for v in worker.violations) * 100)
        label = f"{track_label} | {items} | {conf_pct}%"
    else:
        label = f"{track_label} | SAFE"

    # Draw label background
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y1 - 6, label_h + 4)
    cv2.rectangle(
        canvas,
        (x1, label_y - label_h - 4),
        (x1 + label_w + 4, label_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        canvas, label,
        (x1 + 2, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


def _draw_skeleton(canvas: np.ndarray, pose: WorkerPose) -> None:
    """Draw COCO skeleton connections and keypoint dots."""
    kps = pose.keypoints
    h, w = canvas.shape[:2]

    # Draw connections
    for a, b in SKELETON_CONNECTIONS:
        if a < len(kps) and b < len(kps):
            ka, kb = kps[a], kps[b]
            if ka.confidence >= 0.3 and kb.confidence >= 0.3:
                pt_a = (int(ka.x), int(ka.y))
                pt_b = (int(kb.x), int(kb.y))
                cv2.line(canvas, pt_a, pt_b, COLOR_SKELETON, 2, cv2.LINE_AA)

    # Draw keypoints
    for kp in kps:
        if kp.confidence >= 0.3:
            cx, cy = int(kp.x), int(kp.y)
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), -1)


def _draw_roi_outlines(canvas: np.ndarray, regions: PPERegions) -> None:
    """Draw dashed outlines for all non-None ROIs."""
    rois = [
        regions.helmet_roi,
        regions.vest_roi,
        regions.left_wrist_roi,
        regions.right_wrist_roi,
    ]
    for roi in rois:
        if roi is not None:
            _draw_dashed_rect(
                canvas,
                (int(roi.x1), int(roi.y1), int(roi.x2), int(roi.y2)),
                COLOR_ROI,
            )

    # Draw harness bands
    for band in [regions.harness_left_band, regions.harness_right_band]:
        if band:
            pts = np.array([(int(x), int(y)) for x, y in band], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=COLOR_ROI, thickness=1)


def _draw_dashed_rect(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    dash: int = 8,
    gap: int = 4,
) -> None:
    """Draw a dashed rectangle outline."""
    x1, y1, x2, y2 = rect

    def draw_dashed_line(pt1: tuple[int, int], pt2: tuple[int, int]) -> None:
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist = max(abs(dx), abs(dy))
        if dist == 0:
            return
        step = dash + gap
        for start in range(0, dist, step):
            sx = pt1[0] + int(dx * start / dist)
            sy = pt1[1] + int(dy * start / dist)
            ex = pt1[0] + int(dx * min(start + dash, dist) / dist)
            ey = pt1[1] + int(dy * min(start + dash, dist) / dist)
            cv2.line(canvas, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA)

    draw_dashed_line((x1, y1), (x2, y1))
    draw_dashed_line((x2, y1), (x2, y2))
    draw_dashed_line((x2, y2), (x1, y2))
    draw_dashed_line((x1, y2), (x1, y1))
