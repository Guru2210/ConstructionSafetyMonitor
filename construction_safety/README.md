# Construction Site Safety Monitor

> **Production-grade computer vision pipeline for construction site PPE compliance monitoring.**

Detects workers → estimates pose → verifies helmet/vest/harness placement using keypoint-anchored ROIs → applies zone-based rules → generates structured violation reports → served via FastAPI REST + WebSocket API.

---

## Architecture overview

```
Camera Frame
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ InferencePipeline                                               │
│                                                                 │
│  YOLOv8n (Worker Detection)                                     │
│       ↓                                                         │
│  ByteTracker (Persistent Track IDs)                             │
│       ↓ (per worker, async parallel)                            │
│  YOLOv8n-pose (17 COCO Keypoints) → derive_rois()              │
│       ↓ (per ROI, async parallel)                               │
│  MobileNetV3-Small Heads × 6 [helmet | vest | harness | gloves | boots | goggles]│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
       ↓
ComplianceEngine (EMA temporal smoothing, zone rules)
       ↓
ViolationReport (JSON) → FastAPI → Client
```

### Key insight: Keypoint-anchored PPE verification
PPE is checked only within anatomical ROIs derived from pose keypoints:
- **Helmet**: ROI derived from nose + ear keypoints (prevents false positives from hardhats lying on the ground)
- **Vest**: Full torso bounding box from shoulders + hips, checked as whole + halves
- **Harness**: 40px wide diagonal bands from shoulder→hip in both directions
- **Gloves**: 30px radius circles around wrist keypoints
- **Boots**: Estimated bounding boxes anchored on ankle keypoints
- **Goggles**: Region cropped around eye + nose keypoints

---

## Standalone Inference (CLI)

Run end-to-end evaluation outside of API usage. Supports detailed zone-restriction emulation.

```bash
cd construction_safety/

# Single image
python run_inference.py --input D:/Construction/YoloDataset/images/test/image1009.jpg

# Video file
python run_inference.py --input D:/Construction/Construction.mp4 --video

# With zone rules applied
python run_inference.py --input image.jpg --zone elevated_zone
```

---

## Models used

| Model | Purpose | Why |
|---|---|---|
| YOLOv8n | Person/worker detection | Fast, accurate, production-proven at 25 FPS+ |
| YOLOv8n-pose | 17 COCO keypoint estimation | Enables anatomical ROI derivation |
| MobileNetV3-Small × 6 | PPE classification heads | Lightweight classifier with good transfer learning characteristics (helmet, vest, harness, gloves, boots, goggles) |
| ByteTracker | Multi-object tracking | Maintains persistent IDs across frames for temporal analysis |

---

## Quick start (Docker Compose)

### Prerequisites
- Docker 24+
- docker-compose v2
- Trained model weights in `./models/` (or run in mock mode without weights)

### 1. Clone and configure

```bash
cd construction_safety/
cp .env.example .env
# Edit .env if needed (database URL, model paths, etc.)
```

### 2. Start all services

```bash
docker-compose up --build
```

Services started:
- **API** → http://localhost:8000
- **PostgreSQL** → localhost:5432
- **Redis** → localhost:6379

### 3. Verify health

```bash
curl http://localhost:8000/health
# {"status":"ok","models_loaded":{"detector":"mock",...},"uptime_seconds":3.2}
```

---

## Running without Docker (development)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env

# Start the API
cd construction_safety/
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API usage

### Analyse an image

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/site_image.jpg" \
  -F "site_id=site_001" \
  | python -m json.tool
```

**Response:**
```json
{
  "frame_id": "site_001:0",
  "site_id": "site_001",
  "overall_status": "UNSAFE",
  "worker_count": 2,
  "violation_count": 1,
  "summary_text": "Site status: UNSAFE. 1 violation(s) detected across 1 worker(s) (#2). Affected PPE items: helmet.",
  "processing_ms": 142.3,
  "workers": [
    {
      "track_id": 2,
      "bbox": [120.5, 45.3, 340.2, 418.7],
      "confidence": 0.87,
      "pose_quality": "good",
      "zone_name": "Main Construction Zone",
      "zone_type": "active_zone",
      "ppe": {
        "helmet": {"status": "absent", "confidence": 0.821},
        "vest": {"status": "present", "confidence": 0.763}
      },
      "violations": [...]
    }
  ]
}
```

### Stream frames via WebSocket

```python
import asyncio, websockets, cv2

async def stream():
    async with websockets.connect("ws://localhost:8000/stream/camera_01?site_id=site_001") as ws:
        cap = cv2.VideoCapture("rtsp://...")
        while True:
            ret, frame = cap.read()
            _, jpeg = cv2.imencode(".jpg", frame)
            await ws.send(jpeg.tobytes())
            report = await ws.recv()
            print(report)

asyncio.run(stream())
```

### Configure zones

```bash
curl -X PUT http://localhost:8000/config/zones/site_001 \
  -H "Content-Type: application/json" \
  -d '[{
    "name": "Main Construction Area",
    "type": "active_zone",
    "polygon_geojson": {
      "type": "Polygon",
      "coordinates": [[[0,0],[800,0],[800,600],[0,600],[0,0]]]
    },
    "rules_json": {"helmet": true, "vest": true, "harness": false, "gloves": false}
  }]'
```

### Query violations

```bash
# Get recent urgent violations for a site
curl "http://localhost:8000/violations?site_id=site_001&severity=urgent&limit=20"
```

---

## Training

### 1. Download datasets

```bash
export KAGGLE_API_KEY=your_key_here
export ROBOFLOW_API_KEY=your_key_here

python scripts/download_datasets.py --output-dir ./datasets/merged
```

### 2. Validate dataset

```bash
python scripts/prepare_dataset.py --data ./datasets/merged/data.yaml
```

### 3. Run training (both phases)

```bash
# Full training (GPU recommended)
python scripts/train.py \
  --data ./datasets/merged/data.yaml \
  --model-dir ./models \
  --phase both \
  --epochs-yolo 100 \
  --epochs-heads 30 \
  --device 0

# Phase 1 only (YOLOv8 detection)
python scripts/train.py --data ./datasets/merged/data.yaml --phase 1

# Phase 2 only (ResNet-18 PPE heads)
python scripts/train.py --data ./datasets/merged/data.yaml --phase 2
```

### 4. Copy weights to model directory

```bash
cp models/yolov8m_ppe.pt models/yolov8m.pt
# PPE head weights are auto-saved to models/
```

---

## Running tests

```bash
cd construction_safety/
pytest tests/ -v
```

Expected output:
```
tests/test_pipeline.py::TestROIDerivation::test_helmet_roi_computed_correctly PASSED
tests/test_pipeline.py::TestROIDerivation::test_vest_roi_from_shoulders_and_hips PASSED
tests/test_pipeline.py::TestROIDerivation::test_harness_bands_computed PASSED
tests/test_compliance.py::TestViolationTypes::test_ppe_missing_helmet PASSED
tests/test_api.py::TestHealth::test_health_returns_200 PASSED
tests/test_api.py::TestAnalyze::test_analyze_returns_200 PASSED
...
```

---

## Project structure

```
construction_safety/
├── docker-compose.yml          ← Orchestrates api, postgres, redis
├── Dockerfile                  ← API container
├── requirements.txt            ← Pinned Python 3.11 deps
├── .env.example                ← All configurable variables
├── pytest.ini                  ← Test configuration
├── alembic/                    ← DB migrations
│   └── versions/0001_initial.py
├── scripts/
│   ├── download_datasets.py    ← Kaggle + Roboflow download + dedup
│   ├── prepare_dataset.py      ← Validation + statistics
│   └── train.py                ← Phase 1 (YOLO) + Phase 2 (ResNet-18 heads)
├── src/
│   ├── config.py               ← Pydantic Settings: all constants
│   ├── models/
│   │   ├── detector.py         ← YOLOv8 worker detection
│   │   ├── pose.py             ← YOLOv8-pose + ROI derivation
│   │   ├── ppe_heads.py        ← ResNet-18 PPE classification heads
│   │   └── tracker.py         ← ByteTracker wrapper
│   ├── pipeline/
│   │   ├── inference.py        ← Model cascade orchestrator
│   │   ├── compliance.py       ← Zone rules + EMA temporal smoothing
│   │   ├── zones.py            ← Zone manager with DB + TTL cache
│   │   └── reporter.py         ← ViolationReport generator
│   ├── api/
│   │   ├── main.py             ← FastAPI app factory + lifespan
│   │   ├── schemas.py          ← Pydantic request/response models
│   │   └── routes/
│   │       ├── analyze.py      ← POST /analyze
│   │       ├── stream.py       ← WebSocket /stream/{camera_id}
│   │       ├── config.py       ← PUT /config/zones/{site_id}
│   │       └── violations.py   ← GET /violations
│   ├── db/
│   │   ├── models.py           ← SQLAlchemy ORM
│   │   └── crud.py             ← Async CRUD operations
│   └── utils/
│       ├── drawing.py          ← Frame annotation (bboxes, skeleton, ROIs)
│       └── video.py            ← Frame extraction, RTSP reader
└── tests/
    ├── test_pipeline.py        ← ROI derivation + inference tests
    ├── test_compliance.py      ← Violation + temporal smoothing tests
    └── test_api.py             ← API endpoint integration tests
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async PostgreSQL connection URL |
| `REDIS_URL` | `redis://localhost:6379` | Redis pub/sub URL |
| `MODEL_DIR` | `./models` | Directory containing model weights |
| `CONFIDENCE_THRESHOLD` | `0.45` | Worker detection minimum confidence |
| `HELMET_CONF_THRESHOLD` | `0.70` | Helmet classifier threshold |
| `VEST_CONF_THRESHOLD` | `0.65` | Vest classifier threshold |
| `HARNESS_CONF_THRESHOLD` | `0.60` | Harness classifier threshold |
| `VEST_COVERAGE_THRESHOLD` | `0.40` | Minimum vest pixel coverage (0-1) |
| `VIOLATION_SUSTAIN_SECONDS` | `3.0` | Seconds before violation is reported |
| `WARNING_THRESHOLD` | `0.30` | Violation score for WARNING severity |
| `ALERT_THRESHOLD` | `0.60` | Violation score for ALERT severity |
| `URGENT_THRESHOLD` | `0.85` | Violation score for URGENT severity |
| `LOG_LEVEL` | `INFO` | Log verbosity |

---

## Mock mode

If model weights are absent, each model falls back to **mock mode** automatically:
- A warning is logged: `"Model weights not found — running in mock mode"`
- Mock detections/poses/PPE results return plausible random values
- The full pipeline runs end-to-end for integration testing
- `GET /health` reports `"mock"` for each unloaded model

This allows complete API testing before training is complete.
