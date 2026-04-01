"""
run_inference.py — Quick standalone inference script.

Usage:
  # Single image
  python run_inference.py --input d:/Construction/inference_results.png

  # Video file
  python run_inference.py --input d:/Construction/hardhat.mp4 --video

  # With zone rules applied
  python run_inference.py --input image.jpg --zone elevated_zone

Run from: d:/Construction/construction_safety/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Suppress PyTorch CPU Thread Explosion (which silently crashes Windows with code 1)
import torch
torch.set_num_threads(1)


async def analyse_image(
    image_path: str,
    site_id: str = "site_001",
    zone_type: str | None = None,
    output_path: str = "output_annotated.jpg",
) -> None:
    from src.pipeline.inference import InferencePipeline
    from src.pipeline.compliance import ComplianceEngine
    from src.pipeline.reporter import generate_report
    from src.pipeline.zones import ZoneData, ZoneManager, RequiredPPE
    from src.utils.drawing import annotate_frame

    print(f"\n{'='*60}")
    print(f"Analysing: {image_path}")
    print(f"Site ID:   {site_id}")
    if zone_type:
        print(f"Zone type: {zone_type}")
    print(f"{'='*60}\n")

    # Load frame
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Cannot open image: {image_path}")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Frame size: {w}×{h}")

    # Initialise pipeline
    print("Loading models...")
    pipeline = InferencePipeline()
    engine = ComplianceEngine()
    zone_manager = ZoneManager()
    print(f"Models: {pipeline.models_status}\n")

    # Set up a zone if requested
    zone = None
    required_ppe = None
    if zone_type:
        zone = ZoneData(
            id="zone-cli-001",
            site_id=site_id,
            name=f"CLI {zone_type}",
            type=zone_type,
            # Full-frame polygon
            polygon_geojson={
                "type": "Polygon",
                "coordinates": [[[0, 0], [w, 0], [w, h], [0, h], [0, 0]]]
            },
        )
        zone_manager.add_zones_direct(site_id, [zone])
        required_ppe = zone_manager.get_required_ppe(zone)
        print(f"Zone rules: helmet={required_ppe.helmet} vest={required_ppe.vest} "
              f"harness={required_ppe.harness} gloves={required_ppe.gloves}")

    # Run inference
    print("\nRunning inference...")
    result = await pipeline.process_frame(frame, site_id, "frame_0")
    print(f"  Detection took: {result.processing_ms:.1f}ms")
    print(f"  Workers found:  {len(result.workers)}")

    # Compliance evaluation
    all_violations = []
    for worker in result.workers:
        if zone:
            worker.zone_name = zone.name
            worker.zone_type = zone.type
        comp = engine.evaluate(
            worker=worker,
            zone=zone,
            required_ppe=required_ppe,
            site_id=site_id,
            frame_id="frame_0",
        )
        all_violations.extend(comp.violations)

    # Generate report
    report = generate_report(result, all_violations)

    # Print results
    print(f"\n{'-'*60}")
    print(f"OVERALL STATUS:  {report.overall_status}")
    print(f"Workers:         {report.worker_count}")
    print(f"Violations:      {report.violation_count}")
    print(f"Summary: {report.summary_text}")
    print(f"{'-'*60}")

    for i, worker in enumerate(report.workers, 1):
        print(f"\n  Worker #{worker.track_id} (conf: {worker.confidence:.0%}, "
              f"pose: {worker.pose_quality})")
        if worker.helmet:
            print(f"    Helmet:  {worker.helmet.status:8s} (conf: {worker.helmet.confidence:.0%})")
        if worker.vest:
            print(f"    Vest:    {worker.vest.status:8s} (conf: {worker.vest.confidence:.0%}"
                  + (f", coverage: {worker.vest.coverage_pct:.1f}%" if worker.vest.coverage_pct else "") + ")")
        if worker.harness:
            print(f"    Harness: {worker.harness.status:8s} (conf: {worker.harness.confidence:.0%})")
        if worker.gloves:
            print(f"    Gloves:  {worker.gloves.status:8s} (conf: {worker.gloves.confidence:.0%})")
        if worker.boots:
            print(f"    Boots:   {worker.boots.status:8s} (conf: {worker.boots.confidence:.0%})")
        if worker.goggles:
            print(f"    Goggles: {worker.goggles.status:8s} (conf: {worker.goggles.confidence:.0%})")

        for v in worker.violations:
            print(f"    !! VIOLATION [{v.severity.upper()}]: {v.description}")

    # Save annotated image
    annotated = annotate_frame(frame, report)
    cv2.imwrite(output_path, annotated)
    print(f"Annotated frame saved -> {output_path}")

    # Also save full JSON report
    json_path = Path(output_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print(f"JSON report saved     -> {json_path}")


async def analyse_video(
    video_path: str,
    site_id: str = "site_001",
    zone_type: str | None = None,
    output_path: str = "output_safety.mp4",
    fps: float = 2.0,
) -> None:
    from src.pipeline.inference import InferencePipeline
    from src.pipeline.compliance import ComplianceEngine
    from src.pipeline.reporter import generate_report
    from src.pipeline.zones import ZoneData, ZoneManager
    from src.utils.drawing import annotate_frame
    from src.utils.video import extract_frames_at_fps

    print(f"\n{'='*60}")
    print(f"Analysing video: {video_path}")
    print(f"{'='*60}\n")

    pipeline = InferencePipeline()
    engine = ComplianceEngine()
    zone_manager = ZoneManager()
    print(f"Models: {pipeline.models_status}\n")

    # Detect frame size from first frame
    frames = list(extract_frames_at_fps(video_path, target_fps=fps))
    if not frames:
        print("ERROR: No frames extracted")
        sys.exit(1)

    print(f"Extracted {len(frames)} frames at {fps}fps")

    h, w = frames[0][1].shape[:2]
    zone = None
    required_ppe = None
    if zone_type:
        zone = ZoneData(
            id="zone-cli-001", site_id=site_id,
            name=f"CLI {zone_type}", type=zone_type,
            polygon_geojson={"type": "Polygon",
                             "coordinates": [[[0,0],[w,0],[w,h],[0,h],[0,0]]]}
        )
        zone_manager.add_zones_direct(site_id, [zone])
        required_ppe = zone_manager.get_required_ppe(zone)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    total_violations = 0
    for idx, frame in frames:
        frame_id = f"frame_{idx}"
        result = await pipeline.process_frame(frame, site_id, frame_id)

        all_violations = []
        for worker in result.workers:
            if zone:
                worker.zone_name = zone.name
                worker.zone_type = zone.type
            comp = engine.evaluate(worker, zone, required_ppe, site_id, frame_id)
            all_violations.extend(comp.violations)

        report = generate_report(result, all_violations)
        total_violations += report.violation_count

        annotated = annotate_frame(frame, report)
        writer.write(annotated)

        status_icon = {"SAFE": "[OK]", "WARNING": "[WARN]", "UNSAFE": "[FAIL]", "URGENT": "[!!!]"}.get(report.overall_status, "[?]")
        print(f"  Frame {idx:4d} | {status_icon} {report.overall_status:8s} | "
              f"{report.worker_count} workers | {report.violation_count} violations | "
              f"{result.processing_ms:.0f}ms")

    writer.release()
    print(f"Video saved -> {output_path}")
    print(f"Total violations detected: {total_violations}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construction Safety Inference")
    parser.add_argument("--input", required=True, help="Path to image or video file")
    parser.add_argument("--video", action="store_true", help="Force treat input as video")
    parser.add_argument("--site-id", default="site_001", help="Site identifier")
    parser.add_argument(
        "--zone",
        choices=["active_zone", "elevated_zone", "machinery_zone", "transit_zone", "office_zone"],
        default=None,
        help="Zone type to apply (covers entire frame)",
    )
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--fps", type=float, default=2.0, help="Video analysis FPS")
    args = parser.parse_args()

    input_path = args.input
    ext = Path(input_path).suffix.lower()
    is_video = args.video or ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if is_video:
        out = args.output or "output_safety.mp4"
        asyncio.run(analyse_video(
            video_path=input_path,
            site_id=args.site_id,
            zone_type=args.zone,
            output_path=out,
            fps=args.fps,
        ))
    else:
        out = args.output or "output_annotated.jpg"
        asyncio.run(analyse_image(
            image_path=input_path,
            site_id=args.site_id,
            zone_type=args.zone,
            output_path=out,
        ))


if __name__ == "__main__":
    main()
