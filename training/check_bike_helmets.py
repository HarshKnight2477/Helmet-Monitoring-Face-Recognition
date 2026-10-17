"""How often does the detector mistake a bike helmet for a worker hard hat?

    python training/check_bike_helmets.py --weights models/helmet_yolov8.pt \
        --annotations <bike>/annotations --images <bike>/images \
        [--manifest data/helmet-yolo/test_manifest.json]

For every ground-truth "With Helmet" box (a rider wearing a bike helmet), finds
the detection overlapping it most (IoU >= 0.5) and counts what it was called:

* ``head``   - correct: a bike helmet is not PPE, so this is a violation
* ``helmet`` - wrong: the bike helmet was accepted as a hard hat
* missed     - no detection on that head

With ``--manifest``, only images in the held-out test split are scored.
Writes reports/bike_helmet_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python training/...`

from training.prepare_dataset import find_image, parse_voc  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
BIKE_HELMET_LABEL = "with helmet"


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def classify_boxes(truth_boxes: list, detections: list[tuple[list, str]], min_iou: float = 0.5) -> list[str]:
    """For each ground-truth box: the predicted label of its best-overlapping detection, or 'missed'."""
    outcomes = []
    for truth in truth_boxes:
        best_iou, best_label = 0.0, "missed"
        for box, label in detections:
            overlap = iou(truth, box)
            if overlap >= min_iou and overlap > best_iou:
                best_iou, best_label = overlap, label
        outcomes.append(best_label)
    return outcomes


def check(weights: Path, annotations: Path, images: Path, manifest: Path | None = None,
          confidence: float = 0.4) -> dict:
    from ultralytics import YOLO

    allowed = None
    if manifest is not None:
        allowed = {item["image"] for item in json.loads(manifest.read_text(encoding="utf-8"))}

    model = YOLO(str(weights))
    counts = {"helmet": 0, "head": 0, "missed": 0}
    images_scored = 0
    for xml_path in sorted(annotations.glob("*.xml")):
        annotation = parse_voc(xml_path)
        image_path = find_image(images, annotation, xml_path)
        if image_path is None or (allowed is not None and image_path.name not in allowed):
            continue
        truth = [[b.xmin, b.ymin, b.xmax, b.ymax] for b in annotation.boxes if b.label == BIKE_HELMET_LABEL]
        if not truth:
            continue
        result = model.predict(str(image_path), conf=confidence, verbose=False)[0]
        detections = [(box.tolist(), result.names[int(cls)])
                      for box, cls in zip(result.boxes.xyxy.cpu(), result.boxes.cls.cpu())]
        for outcome in classify_boxes(truth, detections):
            counts[outcome] += 1
        images_scored += 1

    total = sum(counts.values())
    detected = counts["helmet"] + counts["head"]
    return {
        "weights": weights.name,
        "subset": "test split" if manifest else "all images",
        "images": images_scored,
        "bike_helmets": total,
        "flagged_as_no_hard_hat": counts["head"],
        "accepted_as_hard_hat": counts["helmet"],
        "missed": counts["missed"],
        "accepted_rate_of_detected": round(counts["helmet"] / detected, 4) if detected else None,
        "flagged_rate_of_detected": round(counts["head"] / detected, 4) if detected else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bike-helmet false-acceptance check")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "bike_helmet_check.json")
    args = parser.parse_args()

    result = check(args.weights, args.annotations, args.images, args.manifest, args.conf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
