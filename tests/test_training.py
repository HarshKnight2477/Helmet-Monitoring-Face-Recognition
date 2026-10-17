import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.check_bike_helmets import classify_boxes
from training.evaluate import write_reports
from training.prepare_dataset import (VocAnnotation, VocBox, parse_label_map, parse_voc, prepare,
                                      to_yolo_lines)

ROOT = Path(__file__).resolve().parent.parent

HARD_HAT_OBJECTS = [("helmet", 10, 10, 50, 50), ("head", 100, 20, 140, 60), ("person", 0, 0, 200, 100)]
BIKE_OBJECTS = [("With Helmet", 10, 10, 50, 50), ("Without Helmet", 100, 20, 140, 60)]


def make_voc_dataset(root: Path, count: int, prefix: str = "hard_hat_workers",
                     objects=HARD_HAT_OBJECTS) -> tuple[Path, Path]:
    annotations, images = root / prefix / "annotations", root / prefix / "images"
    annotations.mkdir(parents=True)
    images.mkdir(parents=True)
    body = "".join(f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
                   f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>" for n, a, b, c, d in objects)
    for i in range(count):
        name = f"{prefix}{i}"
        (annotations / f"{name}.xml").write_text(
            f"<annotation><filename>{name}.png</filename><size><width>200</width>"
            f"<height>100</height><depth>3</depth></size>{body}</annotation>")
        cv2.imwrite(str(images / f"{name}.png"), np.zeros((100, 200, 3), np.uint8))
    return annotations, images


def test_voc_to_yolo_normalises_and_drops_person(tmp_path):
    annotations, _ = make_voc_dataset(tmp_path, 1)
    lines = to_yolo_lines(parse_voc(annotations / "hard_hat_workers0.xml"), ("helmet", "head"))
    assert lines == ["0 0.150000 0.300000 0.200000 0.400000",
                     "1 0.600000 0.400000 0.200000 0.400000"]


def test_bike_helmets_are_remapped_to_no_hard_hat(tmp_path):
    annotations, _ = make_voc_dataset(tmp_path, 1, "BikesHelmets", BIKE_OBJECTS)
    annotation = parse_voc(annotations / "BikesHelmets0.xml")
    lines = to_yolo_lines(annotation, ("helmet", "head"), {"with helmet": "head", "without helmet": "head"})
    assert [line.split()[0] for line in lines] == ["1", "1"]


def test_boxes_are_clipped_to_the_image():
    annotation = VocAnnotation("x.png", 100, 100, [VocBox("head", -10, 50, 50, 120)])
    assert to_yolo_lines(annotation, ("helmet", "head")) == ["1 0.250000 0.750000 0.500000 0.500000"]


def test_parse_label_map():
    assert parse_label_map("With Helmet=head, without helmet = head") == {
        "with helmet": "head", "without helmet": "head"}


def test_prepare_splits_70_20_10_reproducibly(tmp_path):
    source = make_voc_dataset(tmp_path, 20)
    summary = prepare([source], tmp_path / "out", seed=42)
    counts = {split: info["images"] for split, info in summary["splits"].items()}
    assert counts == {"train": 14, "val": 4, "test": 2}
    assert summary["splits"]["train"]["boxes"] == {"helmet": 14, "head": 14}

    out = tmp_path / "out"
    assert len(list((out / "images" / "test").glob("*.png"))) == 2
    assert len(list((out / "labels" / "test").glob("*.txt"))) == 2
    assert "names: [helmet, head]" in (out / "data.yaml").read_text()

    again = prepare([source], tmp_path / "out2", seed=42)
    test_files = lambda d: sorted(p.name for p in (d / "images" / "test").iterdir())
    assert test_files(out) == test_files(tmp_path / "out2")
    assert again["splits"] == summary["splits"]


def test_prepare_combines_datasets_and_splits_each_one(tmp_path):
    hard_hat = make_voc_dataset(tmp_path, 20)
    bike = make_voc_dataset(tmp_path, 10, "BikesHelmets", BIKE_OBJECTS)
    summary = prepare([hard_hat, bike], tmp_path / "out", seed=42)
    assert summary["sources"] == {"hard_hat_workers": {"test": 2, "val": 4, "train": 14},
                                  "BikesHelmets": {"test": 1, "val": 2, "train": 7}}
    # 7 bike images x 2 heads each are added to "head"; no bike box becomes "helmet".
    assert summary["splits"]["train"]["boxes"] == {"helmet": 14, "head": 14 + 14}

    manifest = json.loads((tmp_path / "out" / "test_manifest.json").read_text())
    bike_items = [m for m in manifest if m["source"] == "BikesHelmets"]
    assert len(manifest) == 3 and len(bike_items) == 1
    assert [b["label"] for b in bike_items[0]["boxes"]] == ["with helmet", "without helmet"]


def test_prepare_requires_annotations(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare([(tmp_path, tmp_path)], tmp_path / "out")


def test_classify_boxes_matches_by_iou():
    truth = [[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]]
    detections = [([0, 0, 10, 11], "helmet"), ([51, 51, 61, 61], "head"), ([200, 200, 210, 210], "head")]
    assert classify_boxes(truth, detections) == ["helmet", "head", "missed"]


def test_write_reports(tmp_path):
    metrics = {"weights": "best.pt", "split": "test", "image_size": 640, "evaluated_on": "2026-10-17",
               "map50": 0.9512, "map50_95": 0.6234, "precision": 0.93, "recall": 0.9,
               "per_class": {"helmet": {"ap50": 0.96, "ap50_95": 0.64}},
               "inference_ms_per_image": 5.1}
    write_reports(metrics, tmp_path)
    assert json.loads((tmp_path / "detection_metrics.json").read_text())["map50"] == 0.9512
    report = (tmp_path / "detection_metrics.md").read_text()
    assert "| mAP@0.5 | **95.1%** |" in report and "| helmet | 96.0% | 64.0% |" in report


def test_kaggle_notebook_is_valid_and_uses_repo_scripts():
    notebook = json.loads((ROOT / "training" / "helmet_yolov8_kaggle.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "from training.prepare_dataset import prepare" in source
    assert "from training.evaluate import evaluate, write_reports" in source
    assert 'split="test"' in source
