# Reports

Results for the published model (`helmet_yolov8.pt` v1.1: YOLOv8s, seed 42, trained on hard hats plus bike-helmet negatives):

- [`detection_metrics.md`](detection_metrics.md) / `.json`: full test-split mAP, precision, recall, per-class AP (`training/evaluate.py`)
- `detection_metrics_hardhat_only.json`: the same metrics on the 500 hard-hat test images only
- `bike_helmet_check.json`: share of test-split bike helmets accepted as hard hats, v1.0 vs v1.1 (`training/check_bike_helmets.py`)
- `training_results.csv`: per-epoch losses and validation metrics
- `split_summary.json`: images and boxes per dataset and split
- [`figures/`](figures/): training curves, precision-recall curve, confusion matrix, example predictions, bike-helmet before/after
