# Helmet detector evaluation

Model `helmet_yolov8.pt` on the **test** split, image size 640,
evaluated 2026-10-17 with `training/evaluate.py`.

| Metric | Value |
|---|---|
| mAP@0.5 | **95.5%** |
| mAP@0.5:0.95 | **63.4%** |
| Precision | 92.1% |
| Recall | 91.7% |
| Inference | 5.19 ms / image |

| Class | AP@0.5 | AP@0.5:0.95 |
|---|---|---|
| helmet | 96.6% | 64.3% |
| head | 94.5% | 62.4% |
