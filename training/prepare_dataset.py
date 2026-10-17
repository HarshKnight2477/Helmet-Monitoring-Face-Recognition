"""Convert Pascal VOC helmet datasets into one YOLO dataset with a train/val/test split.

Built for two public Kaggle datasets:

* Hard Hat Detection (andrewmvd/hard-hat-detection): 5,000 construction images,
  classes helmet / head / person. These ``helmet`` boxes are worker hard hats.
* Helmet Detection (andrewmvd/helmet-detection): 764 motorbike images, classes
  "With Helmet" / "Without Helmet". Both are remapped to ``head``, so the model
  learns that a bike helmet is **not** a hard hat.

    python training/prepare_dataset.py \
        --annotations <hard-hat>/annotations <bike>/annotations \
        --images      <hard-hat>/images      <bike>/images \
        --output data/helmet-yolo

Each dataset is split 70/20/10 on its own (fixed seed), so every split has the
same mix. Writes images/ and labels/ per split, data.yaml for Ultralytics,
split_summary.json, and test_manifest.json (original labels of every test box,
used by check_bike_helmets.py).
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
# Original label -> training label. Bike helmets count as "no hard hat".
DEFAULT_LABEL_MAP = {"with helmet": "head", "without helmet": "head"}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
SPLITS = ("train", "val", "test")


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


def clip_box(box: VocBox, width: int, height: int) -> tuple[float, float, float, float] | None:
    xmin, xmax = sorted((max(0.0, box.xmin), min(width, box.xmax)))
    ymin, ymax = sorted((max(0.0, box.ymin), min(height, box.ymax)))
    if xmax - xmin < 1 or ymax - ymin < 1:
        return None  # degenerate box after clipping
    return xmin, ymin, xmax, ymax


def to_yolo_lines(annotation: VocAnnotation, classes: tuple[str, ...],
                  label_map: dict[str, str] | None = None) -> list[str]:
    """YOLO label lines: ``class x_center y_center width height``, all normalised to 0-1."""
    label_map = label_map or {}
    lines = []
    for box in annotation.boxes:
        label = label_map.get(box.label, box.label)
        clipped = clip_box(box, annotation.width, annotation.height)
        if label not in classes or clipped is None:
            continue
        xmin, ymin, xmax, ymax = clipped
        lines.append(
            f"{classes.index(label)} "
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


def prepare(sources: list[tuple[Path, Path]], output_dir: Path,
            classes: tuple[str, ...] = DEFAULT_CLASSES,
            label_map: dict[str, str] | None = None,
            val: float = 0.2, test: float = 0.1, seed: int = 42) -> dict:
    """``sources`` is a list of (annotations_dir, images_dir) pairs."""
    label_map = DEFAULT_LABEL_MAP if label_map is None else label_map
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary: dict = {"classes": list(classes), "label_map": label_map, "seed": seed,
                     "sources": {}, "splits": {s: {"images": 0, "boxes": {}} for s in SPLITS},
                     "skipped_missing_image": 0}
    manifest = []

    for annotations_dir, images_dir in sources:
        xml_files = sorted(annotations_dir.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No .xml annotations in {annotations_dir}")
        # e.g. "hard_hat_workers12.xml" -> "hard_hat_workers"
        source_name = xml_files[0].stem.rstrip("0123456789_-") or annotations_dir.parent.name
        summary["sources"][source_name] = {}

        for split, names in split_items([p.name for p in xml_files], val, test, seed).items():
            images = 0
            for name in names:
                xml_path = annotations_dir / name
                annotation = parse_voc(xml_path)
                image_path = find_image(images_dir, annotation, xml_path)
                if image_path is None:
                    summary["skipped_missing_image"] += 1
                    continue
                target = output_dir / "images" / split / image_path.name
                if target.exists():
                    raise ValueError(f"Duplicate image name across datasets: {image_path.name}")
                lines = to_yolo_lines(annotation, classes, label_map)
                shutil.copy2(image_path, target)
                (output_dir / "labels" / split / f"{image_path.stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                counts = Counter(classes[int(line.split()[0])] for line in lines)
                boxes = summary["splits"][split]["boxes"]
                for label, count in counts.items():
                    boxes[label] = boxes.get(label, 0) + count
                images += 1
                if split == "test":
                    manifest.append({
                        "image": image_path.name,
                        "source": source_name,
                        "boxes": [{"label": b.label, "box": list(clipped)}
                                  for b in annotation.boxes
                                  if (clipped := clip_box(b, annotation.width, annotation.height))],
                    })
            summary["sources"][source_name][split] = images
            summary["splits"][split]["images"] += images

    data_yaml = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(classes)}\nnames: [{', '.join(classes)}]\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "test_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return summary


def parse_label_map(text: str) -> dict[str, str]:
    """``"with helmet=head,without helmet=head"`` -> dict."""
    pairs = (item.split("=", 1) for item in text.split(",") if "=" in item)
    return {k.strip().lower(): v.strip().lower() for k, v in pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--map", default=",".join(f"{k}={v}" for k, v in DEFAULT_LABEL_MAP.items()),
                        help='label remapping, e.g. "with helmet=head,without helmet=head"')
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.annotations) != len(args.images):
        parser.error("--annotations and --images need the same number of folders")

    classes = tuple(c.strip().lower() for c in args.classes.split(",") if c.strip())
    summary = prepare(list(zip(args.annotations, args.images)), args.output, classes,
                      parse_label_map(args.map), args.val, args.test, args.seed)
    print(json.dumps(summary, indent=2))
    print(f"data.yaml written to {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
