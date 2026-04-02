import cv2
from typing import Iterator, Tuple
import numpy as np

def extract_frames_at_fps(video_path: str, target_fps: float) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
        
    frame_interval = max(1, int(fps / target_fps))
    
    idx = 0
    yield_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if idx % frame_interval == 0:
            yield yield_idx, frame
            yield_idx += 1
            
        idx += 1
        
    cap.release()