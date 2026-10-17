"""Evaluate a trained helmet detector and write the metrics to reports/.

    python training/evaluate.py --weights models/helmet_yolov8.pt --data path/to/data.yaml

Scores the held-out *test* split (never used during training or model
selection) and writes reports/detection_metrics.json and .md with
mAP@0.5, mAP@0.5:0.95, precision, recall and per-class AP.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def evaluate(weights: Path, data: Path, split: str = "test", image_size: int = 640,
             batch: int = 16, device: str | None = None) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs = {"data": str(data), "split": split, "imgsz": image_size, "batch": batch,
              "plots": False, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    results = model.val(**kwargs)

    box = results.box
    per_class = {
        model.names[int(class_id)]: {
            "ap50": round(float(box.ap50[i]), 4),
            "ap50_95": round(float(box.ap[i]), 4),
        }
        for i, class_id in enumerate(box.ap_class_index)
    }
    return {
        "weights": weights.name,
        "split": split,
        "image_size": image_size,
        "evaluated_on": date.today().isoformat(),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
        "inference_ms_per_image": round(float(results.speed["inference"]), 2),
    }


def write_reports(metrics: dict, reports_dir: Path = REPORTS_DIR) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "detection_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rows = "\n".join(f"| {name} | {m['ap50']:.1%} | {m['ap50_95']:.1%} |"
                     for name, m in metrics["per_class"].items())
    (reports_dir / "detection_metrics.md").write_text(f"""# Helmet detector evaluation

Model `{metrics['weights']}` on the **{metrics['split']}** split, image size {metrics['image_size']},
evaluated {metrics['evaluated_on']} with `training/evaluate.py`.

| Metric | Value |
|---|---|
| mAP@0.5 | **{metrics['map50']:.1%}** |
| mAP@0.5:0.95 | **{metrics['map50_95']:.1%}** |
| Precision | {metrics['precision']:.1%} |
| Recall | {metrics['recall']:.1%} |
| Inference | {metrics['inference_ms_per_image']} ms / image |

| Class | AP@0.5 | AP@0.5:0.95 |
|---|---|---|
{rows}
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the helmet detector")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    metrics = evaluate(args.weights, args.data, args.split, args.imgsz, args.batch, args.device)
    write_reports(metrics, args.reports)
    print(json.dumps(metrics, indent=2))
    print(f"Reports written to {args.reports}")


if __name__ == "__main__":
    main()
