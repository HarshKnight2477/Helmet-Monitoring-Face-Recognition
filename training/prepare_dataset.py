"""Convert a Pascal VOC helmet dataset into YOLO format with a train/val/test split.

Built for the Kaggle "Hard Hat Detection" dataset (andrewmvd/hard-hat-detection:
5,000 images, classes helmet / head / person), but works with any VOC dataset.

    python training/prepare_dataset.py \
        --annotations /kaggle/input/hard-hat-detection/annotations \
        --images /kaggle/input/hard-hat-detection/images \
        --output /kaggle/working/helmet-yolo

Writes images/ and labels/ folders per split, a data.yaml for Ultralytics,
and split_summary.json with image and box counts.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CLASSES = ("helmet", "head")  # "person" boxes are dropped: we only score heads
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class VocBox:
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class VocAnnotation:
    filename: str
    width: int
    height: int
    boxes: list[VocBox]


def parse_voc(path: Path) -> VocAnnotation:
    root = ET.parse(path).getroot()
    size = root.find("size")
    boxes = []
    for obj in root.iter("object"):
        bndbox = obj.find("bndbox")
        boxes.append(VocBox(
            label=obj.findtext("name", "").strip().lower(),
            xmin=float(bndbox.findtext("xmin")),
            ymin=float(bndbox.findtext("ymin")),
            xmax=float(bndbox.findtext("xmax")),
            ymax=float(bndbox.findtext("ymax")),
        ))
    return VocAnnotation(
        filename=root.findtext("filename", path.with_suffix(".png").name).strip(),
        width=int(float(size.findtext("width"))),
        height=int(float(size.findtext("height"))),
        boxes=boxes,
    )


def to_yolo_lines(annotation: VocAnnotation, classes: tuple[str, ...]) -> list[str]:
    """YOLO label lines: ``class x_center y_center width height``, all normalised to 0-1."""
    lines = []
    for box in annotation.boxes:
        if box.label not in classes:
            continue
        xmin, xmax = sorted((max(0.0, box.xmin), min(annotation.width, box.xmax)))
        ymin, ymax = sorted((max(0.0, box.ymin), min(annotation.height, box.ymax)))
        if xmax - xmin < 1 or ymax - ymin < 1:
            continue  # degenerate box after clipping
        lines.append(
            f"{classes.index(box.label)} "
            f"{(xmin + xmax) / 2 / annotation.width:.6f} {(ymin + ymax) / 2 / annotation.height:.6f} "
            f"{(xmax - xmin) / annotation.width:.6f} {(ymax - ymin) / annotation.height:.6f}"
        )
    return lines


def find_image(images_dir: Path, annotation: VocAnnotation, xml_path: Path) -> Path | None:
    candidate = images_dir / annotation.filename
    if candidate.is_file():
        return candidate
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{xml_path.stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def split_items(items: list, val: float, test: float, seed: int) -> dict[str, list]:
    shuffled = sorted(items)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test)
    n_val = round(len(shuffled) * val)
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test:n_test + n_val],
        "train": shuffled[n_test + n_val:],
    }


def prepare(annotations_dir: Path, images_dir: Path, output_dir: Path,
            classes: tuple[str, ...] = DEFAULT_CLASSES, val: float = 0.2, test: float = 0.1,
            seed: int = 42) -> dict:
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml annotations in {annotations_dir}")

    splits = split_items([p.name for p in xml_files], val, test, seed)
    summary: dict = {"classes": list(classes), "seed": seed, "splits": {}, "skipped_missing_image": 0}

    for split, names in splits.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        box_counts: Counter = Counter()
        images = 0
        for name in names:
            xml_path = annotations_dir / name
            annotation = parse_voc(xml_path)
            image_path = find_image(images_dir, annotation, xml_path)
            if image_path is None:
                summary["skipped_missing_image"] += 1
                continue
            lines = to_yolo_lines(annotation, classes)
            box_counts.update(classes[int(line.split()[0])] for line in lines)
            shutil.copy2(image_path, output_dir / "images" / split / image_path.name)
            (output_dir / "labels" / split / f"{image_path.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            images += 1
        summary["splits"][split] = {"images": images, "boxes": dict(box_counts)}

    data_yaml = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(classes)}\nnames: [{', '.join(classes)}]\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    classes = tuple(c.strip().lower() for c in args.classes.split(",") if c.strip())
    summary = prepare(args.annotations, args.images, args.output, classes, args.val, args.test, args.seed)
    print(json.dumps(summary, indent=2))
    print(f"data.yaml written to {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
