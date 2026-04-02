"""
src/models/ppe_heads.py — ResNet-18 PPE classification heads.

Four independent heads:
  - HelmetHead   : check_helmet()
  - VestHead     : check_vest()
  - HarnessHead  : check_harness()
  - GlovesHead   : check_gloves()

Each head loads from a .pth file if it exists, otherwise runs in mock mode.
PPEClassifier is the unified interface.
"""
from __future__ import annotations

import random
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PPECheckResult:
    """Result of a single PPE check."""

    status: str  # "present" | "absent" | "partial" | "unknown"
    confidence: float  # 0.0 – 1.0
    coverage_pct: float | None = None   # vest only — pixel coverage %
    symmetry_score: float | None = None  # harness only — left/right symmetry
    details: str = ""


# ---------------------------------------------------------------------------
# Helper: build a ResNet-18 head
# ---------------------------------------------------------------------------

def _build_mobilenet_v3_head(num_classes: int = 3) -> "Any":
    """Build a MobileNetV3-Small with replaced final classifier layer."""
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def _load_model(model_path: Path, num_classes: int = 3) -> "tuple[Any, bool]":
    """
    Try to load MobileNetV3 from path.
    Returns (model, is_mock).
    """
    try:
        import torch

        model = _build_mobilenet_v3_head(num_classes)
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        logger.info(f"Loaded PPE head from {model_path}")
        return model, False
    except Exception:
        logger.warning(
            f"PPE head weight load failed from {model_path}:\n" + traceback.format_exc()
        )
        return None, True


def _preprocess_crop(crop: np.ndarray) -> "Any":
    """Preprocess a numpy crop for MobileNetV3 inference."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        raise ValueError("Degenerate crop")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # OpenCV is BGR — convert to RGB
    import cv2
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if len(crop.shape) == 3 and crop.shape[2] == 3 else crop
    return transform(rgb).unsqueeze(0)


def _run_inference(model: "Any", tensor: "Any") -> "list[float]":
    """Run head inference, return softmax probabilities."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].tolist()
    return probs


# ---------------------------------------------------------------------------
# Status label mappings (index → status string)
# ---------------------------------------------------------------------------
# 0 = absent, 1 = partial, 2 = present
_STATUS_MAP = {0: "absent", 1: "partial", 2: "present"}


# ---------------------------------------------------------------------------
# PPEClassifier
# ---------------------------------------------------------------------------

class PPEClassifier:
    """
    Unified PPE classification interface.

    Each PPE type has its own ResNet-18 head. All heads fall back to a
    plausible mock if weights are absent, logging a warning.
    """

    def __init__(
        self,
        helmet_model_path: str | Path | None = None,
        vest_model_path: str | Path | None = None,
        harness_model_path: str | Path | None = None,
        gloves_model_path: str | Path | None = None,
        boots_model_path: str | Path | None = None,
        goggles_model_path: str | Path | None = None,
        helmet_conf_threshold: float = 0.70,
        vest_conf_threshold: float = 0.65,
        harness_conf_threshold: float = 0.60,
        gloves_conf_threshold: float = 0.55,
        boots_conf_threshold: float = 0.60,
        goggles_conf_threshold: float = 0.60,
        vest_coverage_threshold: float = 0.40,
    ) -> None:
        self.helmet_conf_threshold = helmet_conf_threshold
        self.vest_conf_threshold = vest_conf_threshold
        self.harness_conf_threshold = harness_conf_threshold
        self.gloves_conf_threshold = gloves_conf_threshold
        self.boots_conf_threshold = boots_conf_threshold
        self.goggles_conf_threshold = goggles_conf_threshold
        self.vest_coverage_threshold = vest_coverage_threshold

        # Load each head
        self._helmet_model, self._helmet_mock = self._init_head(
            helmet_model_path, "helmet"
        )
        self._vest_model, self._vest_mock = self._init_head(vest_model_path, "vest")
        self._harness_model, self._harness_mock = self._init_head(
            harness_model_path, "harness"
        )
        self._gloves_model, self._gloves_mock = self._init_head(
            gloves_model_path, "gloves"
        )
        self._boots_model, self._boots_mock = self._init_head(
            boots_model_path, "boots"
        )
        self._goggles_model, self._goggles_mock = self._init_head(
            goggles_model_path, "goggles"
        )

    @property
    def models_status(self) -> dict[str, str]:
        return {
            "helmet": "mock" if self._helmet_mock else "loaded",
            "vest": "mock" if self._vest_mock else "loaded",
            "harness": "mock" if self._harness_mock else "loaded",
            "gloves": "mock" if self._gloves_mock else "loaded",
            "boots": "mock" if self._boots_mock else "loaded",
            "goggles": "mock" if self._goggles_mock else "loaded",
        }

    # ------------------------------------------------------------------
    # Public check methods
    # ------------------------------------------------------------------

    def check_helmet(self, roi_crop: np.ndarray) -> PPECheckResult:
        """
        Check if a helmet is correctly worn in the head ROI crop.

        status:
          - "present"  — helmet detected, centroid within expected position
          - "partial"  — helmet bbox present but centroid displaced > 15px
          - "absent"   — no helmet detected
        """
        if self._helmet_mock:
            return self._mock_result("helmet")
        try:
            tensor = _preprocess_crop(roi_crop)
            probs = _run_inference(self._helmet_model, tensor)
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            if conf < self.helmet_conf_threshold:
                idx = 0  # force absent if below threshold
            return PPECheckResult(
                status=_STATUS_MAP.get(idx, "absent"),
                confidence=round(conf, 4),
                details=f"probs={[round(p, 3) for p in probs]}",
            )
        except ValueError:
            return PPECheckResult(status='unknown', confidence=0.0)
        except Exception:
            logger.warning("check_helmet failed:\n" + traceback.format_exc())
            return PPECheckResult(status="unknown", confidence=0.0)

    def check_vest(
        self,
        roi_crop: np.ndarray,
        left_half_crop: np.ndarray | None = None,
        right_half_crop: np.ndarray | None = None,
    ) -> PPECheckResult:
        """
        Check vest coverage in torso ROI.

        coverage_pct: estimated pixel coverage of vest class in torso (0-100)
        partial = coverage < threshold or vest missing from one half
        """
        if self._vest_mock:
            return self._mock_result("vest")
        try:
            tensor = _preprocess_crop(roi_crop)
            probs = _run_inference(self._vest_model, tensor)
            idx = int(np.argmax(probs))
            conf = float(probs[idx])

            # Estimate coverage via secondary-class score
            present_prob = probs[2] if len(probs) > 2 else conf
            coverage_pct = round(present_prob * 100, 1)

            # Check left and right halves
            left_present = right_present = True
            if left_half_crop is not None and left_half_crop.size > 0:
                lp = _run_inference(self._vest_model, _preprocess_crop(left_half_crop))
                left_present = np.argmax(lp) == 2  # "present"
            if right_half_crop is not None and right_half_crop.size > 0:
                rp = _run_inference(self._vest_model, _preprocess_crop(right_half_crop))
                right_present = np.argmax(rp) == 2

            if coverage_pct < self.vest_coverage_threshold * 100 or not (left_present and right_present):
                status = "partial"
            else:
                status = _STATUS_MAP.get(idx, "absent")

            if conf < self.vest_conf_threshold:
                status = "absent"

            return PPECheckResult(
                status=status,
                confidence=round(conf, 4),
                coverage_pct=coverage_pct,
            )
        except ValueError:
            return PPECheckResult(status='unknown', confidence=0.0)
        except Exception:
            logger.warning("check_vest failed:\n" + traceback.format_exc())
            return PPECheckResult(status="unknown", confidence=0.0)

    def check_harness(
        self,
        left_band_crop: np.ndarray | None,
        right_band_crop: np.ndarray | None,
    ) -> PPECheckResult:
        """
        Check harness by inspecting both diagonal bands.

        symmetry_score: ratio of min/max confidence across bands (0-1)
        partial = only one band detected or symmetry_score < 0.6
        """
        if self._harness_mock:
            return self._mock_result("harness")
        try:
            left_conf = 0.0
            right_conf = 0.0

            if left_band_crop is not None and left_band_crop.size > 0:
                lp = _run_inference(self._harness_model, _preprocess_crop(left_band_crop))
                left_conf = float(lp[2]) if len(lp) > 2 else 0.0

            if right_band_crop is not None and right_band_crop.size > 0:
                rp = _run_inference(self._harness_model, _preprocess_crop(right_band_crop))
                right_conf = float(rp[2]) if len(rp) > 2 else 0.0

            both_detected = (
                left_conf >= self.harness_conf_threshold
                and right_conf >= self.harness_conf_threshold
            )
            either_detected = (
                left_conf >= self.harness_conf_threshold
                or right_conf >= self.harness_conf_threshold
            )

            max_conf = max(left_conf, right_conf)
            min_conf = min(left_conf, right_conf)
            symmetry_score = round(min_conf / max_conf, 3) if max_conf > 0 else 0.0

            if both_detected and symmetry_score >= 0.6:
                status = "present"
            elif either_detected or (both_detected and symmetry_score < 0.6):
                status = "partial"
            else:
                status = "absent"

            return PPECheckResult(
                status=status,
                confidence=round((left_conf + right_conf) / 2, 4),
                symmetry_score=symmetry_score,
            )
        except ValueError:
            return PPECheckResult(status='unknown', confidence=0.0)
        except Exception:
            logger.warning("check_harness failed:\n" + traceback.format_exc())
            return PPECheckResult(status="unknown", confidence=0.0)

    def check_gloves(
        self,
        left_wrist_crop: np.ndarray | None,
        right_wrist_crop: np.ndarray | None,
    ) -> PPECheckResult:
        """Check gloves at wrist ROIs. Returns 'unknown' if model unimplemented."""
        if self._gloves_mock:
            return self._mock_result("gloves")
        try:
            left_conf = right_conf = 0.0
            if left_wrist_crop is not None and left_wrist_crop.size > 0:
                lp = _run_inference(self._gloves_model, _preprocess_crop(left_wrist_crop))
                left_conf = float(lp[2]) if len(lp) > 2 else 0.0
            if right_wrist_crop is not None and right_wrist_crop.size > 0:
                rp = _run_inference(self._gloves_model, _preprocess_crop(right_wrist_crop))
                right_conf = float(rp[2]) if len(rp) > 2 else 0.0

            avg_conf = (left_conf + right_conf) / 2
            if avg_conf >= self.gloves_conf_threshold:
                status = "present"
            elif avg_conf > 0.15:
                status = "partial"
            else:
                status = "absent"

            return PPECheckResult(status=status, confidence=round(avg_conf, 4))
        except ValueError:
            return PPECheckResult(status='unknown', confidence=0.0)
        except Exception:
            logger.warning("check_gloves failed:\n" + traceback.format_exc())
            return PPECheckResult(status="unknown", confidence=0.0)

    def check_boots(
        self,
        left_boot_crop: np.ndarray | None,
        right_boot_crop: np.ndarray | None,
    ) -> PPECheckResult:
        """Check boots at ankle ROIs."""
        if self._boots_mock:
            return self._mock_result("boots")
        try:
            left_conf = right_conf = 0.0
            if left_boot_crop is not None and left_boot_crop.size > 0:
                lp = _run_inference(self._boots_model, _preprocess_crop(left_boot_crop))
                left_conf = float(lp[2]) if len(lp) > 2 else 0.0
            if right_boot_crop is not None and right_boot_crop.size > 0:
                rp = _run_inference(self._boots_model, _preprocess_crop(right_boot_crop))
                right_conf = float(rp[2]) if len(rp) > 2 else 0.0

            avg_conf = (left_conf + right_conf) / 2
            if avg_conf >= self.boots_conf_threshold:
                status = "present"
            elif avg_conf > 0.15:
                status = "partial"
            else:
                status = "absent"

            return PPECheckResult(status=status, confidence=round(avg_conf, 4))
        except ValueError:
            return PPECheckResult(status='unknown', confidence=0.0)
        except Exception:
            logger.warning("check_boots failed:\n" + traceback.format_exc())
            return PPECheckResult(status="unknown", confidence=0.0)

    def check_goggles(self, goggles_crop: np.ndarray | None) -> PPECheckResult:
        """Check goggles at eye ROI."""
        if self._goggles_mock:
            return self._mock_result("goggles")
        try:
            if goggles_crop is None or goggles_crop.size == 0:
                return PPECheckResult(status="absent", confidence=0.0)
                
            tensor = _preprocess_crop(goggles_crop)
            probs = _run_inference(self._goggles_model, tensor)
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            
            if conf < self.goggles_conf_threshold:
                idx = 0  # force absent if below threshold
                
            return PPECheckResult(
                status=_STATUS_MAP.get(idx, "absent"),
                confidence=round(conf, 4),
                details=f"probs={[round(p, 3) for p in probs]}",
            )
        except ValueError:
            return PPECheckResult(status='unknown', confidence=0.0)
        except Exception:
            logger.warning("check_goggles failed:\n" + traceback.format_exc())
            return PPECheckResult(status="unknown", confidence=0.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_head(
        model_path: str | Path | None, name: str
    ) -> "tuple[Any, bool]":
        if model_path is None:
            logger.warning(f"PPE head '{name}': no path provided — MOCK mode")
            return None, True
        path = Path(model_path)
        if not path.exists():
            logger.warning(f"PPE head '{name}': weights not found at {path} — MOCK mode")
            return None, True
        return _load_model(path)

    @staticmethod
    def _mock_result(ppe_type: str) -> PPECheckResult:
        """Return plausible random result for demonstration."""
        statuses = ["present", "present", "present", "absent", "partial"]
        status = random.choice(statuses)
        conf = round(random.uniform(0.45, 0.95), 3)
        coverage = round(random.uniform(30, 90), 1) if ppe_type == "vest" else None
        sym = round(random.uniform(0.5, 1.0), 3) if ppe_type == "harness" else None
        return PPECheckResult(
            status=status,
            confidence=conf,
            coverage_pct=coverage,
            symmetry_score=sym,
            details="mock",
        )
