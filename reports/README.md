# Reports

Results for the published model (`helmet_yolov8.pt`, YOLOv8s, 60 epochs, seed 42):

- [`detection_metrics.md`](detection_metrics.md) / `.json`: test-split mAP, precision, recall, per-class AP (from `training/evaluate.py`)
- `training_results.csv`: per-epoch losses and validation metrics
- `split_summary.json`: images and boxes per split
- [`figures/`](figures/): training curves, precision-recall curve, confusion matrix, example predictions
