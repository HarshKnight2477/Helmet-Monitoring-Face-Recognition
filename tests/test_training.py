import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.evaluate import write_reports
from training.prepare_dataset import parse_voc, prepare, to_yolo_lines

ROOT = Path(__file__).resolve().parent.parent

VOC = """<annotation>
  <filename>{name}.png</filename>
  <size><width>200</width><height>100</height><depth>3</depth></size>
  <object><name>helmet</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
  <object><name>head</name><bndbox><xmin>100</xmin><ymin>20</ymin><xmax>140</xmax><ymax>60</ymax></bndbox></object>
  <object><name>person</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>200</xmax><ymax>100</ymax></bndbox></object>
</annotation>"""


def make_voc_dataset(root: Path, count: int) -> tuple[Path, Path]:
    annotations, images = root / "annotations", root / "images"
    annotations.mkdir()
    images.mkdir()
    for i in range(count):
        (annotations / f"hard_hat_workers{i}.xml").write_text(VOC.format(name=f"hard_hat_workers{i}"))
        cv2.imwrite(str(images / f"hard_hat_workers{i}.png"), np.zeros((100, 200, 3), np.uint8))
    return annotations, images


def test_voc_to_yolo_normalises_and_drops_person(tmp_path):
    annotations, _ = make_voc_dataset(tmp_path, 1)
    lines = to_yolo_lines(parse_voc(annotations / "hard_hat_workers0.xml"), ("helmet", "head"))
    assert lines == ["0 0.150000 0.300000 0.200000 0.400000",
                     "1 0.600000 0.400000 0.200000 0.400000"]


def test_boxes_are_clipped_to_the_image():
    from training.prepare_dataset import VocAnnotation, VocBox
    annotation = VocAnnotation("x.png", 100, 100, [VocBox("head", -10, 50, 50, 120)])
    assert to_yolo_lines(annotation, ("helmet", "head")) == ["1 0.250000 0.750000 0.500000 0.500000"]


def test_prepare_splits_70_20_10_reproducibly(tmp_path):
    annotations, images = make_voc_dataset(tmp_path, 20)
    summary = prepare(annotations, images, tmp_path / "out", seed=42)
    counts = {split: info["images"] for split, info in summary["splits"].items()}
    assert counts == {"train": 14, "val": 4, "test": 2}
    assert summary["splits"]["train"]["boxes"] == {"helmet": 14, "head": 14}

    out = tmp_path / "out"
    assert len(list((out / "images" / "test").glob("*.png"))) == 2
    assert len(list((out / "labels" / "test").glob("*.txt"))) == 2
    assert "names: [helmet, head]" in (out / "data.yaml").read_text()

    again = prepare(annotations, images, tmp_path / "out2", seed=42)
    test_files = lambda d: sorted(p.name for p in (d / "images" / "test").iterdir())
    assert test_files(out) == test_files(tmp_path / "out2")
    assert again["splits"] == summary["splits"]


def test_prepare_requires_annotations(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path, tmp_path, tmp_path / "out")


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
