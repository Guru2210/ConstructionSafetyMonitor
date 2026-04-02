# Construction Site Safety Monitor

> **Computer vision pipeline for construction site PPE compliance monitoring.**

Detects workers -> estimates pose -> verifies helmet/vest/harness/gloves/boots/goggles placement using keypoint-anchored ROIs -> applies zone-based rules -> generates structured violation reports.

---

## Architecture overview

\Camera Frame
     |
     v
+----------------------------------------------------------------------+
| InferencePipeline                                                    |
|                                                                      |
|  YOLOv8n (Worker Detection)                                          |
|       v                                                              |
|  ByteTracker (Persistent Track IDs)                                  |
|       v (per worker, async parallel)                                 |
|  YOLOv8n-pose (17 COCO Keypoints) -> derive_rois()                   |
|       v (per ROI, async parallel)                                    |
|  MobileNetV3-Small Heads x 6 [helmet | vest | harness | gloves | boots | goggles] |
|                                                                      |
+----------------------------------------------------------------------+
       v
ComplianceEngine (zone rules)
       v
ViolationReport (JSON) + Annotated Video/Image
\
### Key insight: Keypoint-anchored PPE verification
PPE is checked only within anatomical ROIs derived from pose keypoints:
- **Helmet**: ROI derived from nose + ear keypoints
- **Vest**: Full torso bounding box from shoulders + hips, checked as whole + halves
- **Harness**: 40px wide diagonal bands from shoulder->hip
- **Gloves**: 30px radius circles around wrist keypoints
- **Boots**: Estimated bounding boxes anchored on ankle keypoints
- **Goggles**: Region cropped around eye + nose keypoints

---

## Installation

\\ash
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
\
## Standalone Inference (CLI)

Run end-to-end evaluation on videos or images.

\\ash
cd construction_safety/

# Single image
python run_inference.py --input D:/Construction/YoloDataset/images/test/image1.jpg

# Video file
python run_inference.py --input D:/Construction/Construction.mp4 --video        

# With zone rules applied
python run_inference.py --input image.jpg --zone elevated_zone
\
---

## Models used

| Model | Purpose | Why |
|---|---|---|
| YOLOv8n | Person/worker detection | Fast, accurate, production-proven at 25 FPS+ |
| YOLOv8n-pose | 17 COCO keypoint estimation | Enables anatomical ROI derivation |
| MobileNetV3-Small x 6 | PPE classification heads | Lightweight classifier with good transfer learning characteristics |
| ByteTracker | Multi-object tracking | Maintains persistent IDs across frames |

---

## Training Pipeline

All training steps (data prep, YOLOv8 fine-tuning, and MobileNetV3 head training) are contained within the \	rain.ipynb\ Jupyter Notebook. 

1. Open \	rain.ipynb\.
2. Run the cells to process the dataset and generate Ground Truth crops.
3. Train the MobileNetV3 heads.
4. Export weights to the \models/\ directory for inference.

---

## Project structure

\\	ext
construction_safety/
|-- requirements.txt            <- Python dependencies
|-- pytest.ini                  <- Test configuration
|-- README.md                   <- This file
|-- run_inference.py            <- CLI entrypoint for running the pipeline
|-- src/
|   |-- config.py               <- Configurations, paths, and thresholds
|   |-- models/
|   |   |-- detector.py         <- YOLOv8 worker detection
|   |   |-- pose.py             <- YOLOv8-pose + ROI derivation
|   |   |-- ppe_heads.py        <- MobileNetV3-Small PPE classification heads
|   |   |-- tracker.py          <- ByteTracker wrapper
|   |-- pipeline/
|   |   |-- inference.py        <- Model cascade orchestrator
|   |   |-- compliance.py       <- Zone rules and violation detection
|   |   |-- zones.py            <- Zone manager definitions
|   |   |-- reporter.py         <- ViolationReport generator
|   |-- utils/
|       |-- drawing.py          <- Frame annotation (bboxes, skeleton, ROIs)
|       |-- video.py            <- Frame extraction reader
|-- tests/
    |-- test_pipeline.py        <- ROI derivation + inference tests
    |-- test_compliance.py      <- Violation tests
\