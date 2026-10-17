# Helmet detector evaluation

Model `helmet_yolov8.pt` on the **test** split, image size 640,
evaluated 2026-10-18 with `training/evaluate.py`.

| Metric | Value |
|---|---|
| mAP@0.5 | **93.8%** |
| mAP@0.5:0.95 | **62.1%** |
| Precision | 88.5% |
| Recall | 90.7% |
| Inference | 5.21 ms / image |

| Class | AP@0.5 | AP@0.5:0.95 |
|---|---|---|
| helmet | 96.6% | 64.0% |
| head | 91.1% | 60.1% |
