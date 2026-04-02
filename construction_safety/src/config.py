from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
WORKSPACE_ROOT = BASE_DIR.parent

class Settings:
    # Model Paths
    detector_model_path: Path = BASE_DIR / "models" / "yolov8n.pt"
    pose_model_path: Path = BASE_DIR / "models" / "yolov8n-pose.pt"
    
    helmet_model_path: Path = BASE_DIR / "models" / "helmet_head.pth"
    vest_model_path: Path = BASE_DIR / "models" / "vest_head.pth"
    harness_model_path: Path = BASE_DIR / "models" / "harness_head.pth"
    gloves_model_path: Path = BASE_DIR / "models" / "gloves_head.pth"
    boots_model_path: Path = BASE_DIR / "models" / "boots_head.pth"
    goggles_model_path: Path = BASE_DIR / "models" / "goggles_head.pth"

    # Thresholds
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    keypoint_conf_threshold: float = 0.5

    helmet_conf_threshold: float = 0.35
    vest_conf_threshold: float = 0.35
    harness_conf_threshold: float = 0.35
    gloves_conf_threshold: float = 0.35
    boots_conf_threshold: float = 0.35
    goggles_conf_threshold: float = 0.35
    vest_coverage_threshold: float = 0.3

settings = Settings()
