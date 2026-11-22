# Trained models

## Road damage model (used by default)

`road_damage_5class_v8n.onnx` (12.8 MB, in git) finds longitudinal, transverse and alligator cracks,
potholes and speed bumps in one pass. With it, the older models below are not needed.

- YOLOv8n, 640 x 640, exported with opset 12; runs through OpenCV's DNN module (tested with 4.10 on the Pi 4)
- 0.48 s per photo on a Raspberry Pi 4 (the older models and crack rules: 4.5 s)
- A first version: 36 epochs (26 minutes) on 4,279 photos
- SHA-256: `f0b0e4b28f5773571ae572a6fc6fc8ded0edc1f1fff7566705f300d1bc7d07ae`

| Validation-chosen cut-off | pothole 0.25, crack 0.15, speed bump 0.15 |
|---|---|
| mAP50 on 1,135 unseen photos | pothole 0.58, longitudinal crack 0.42, transverse crack 0.35, alligator crack 0.54, speed bump 0.85 |

The unseen photos come from the same sources as the training photos. The 40 photos in `test_images/real/`
do not, and are the harder test. Known limits: close-up cracks are often missed or called potholes, and
painted or worn humps can be missed. See the table in `test_images/evaluate.py`'s output.

Training data:
- [RDD2022](https://huggingface.co/datasets/cvtechniques/Road_Damage_Detection_USA) (Japan, India, USA, Czechia, China motorbike): cracks and potholes
- [Speed breaker and rumbler dataset](https://huggingface.co/datasets/Aditya2302/speed-breaker-rumbler-dataset), CC-BY-4.0: speed bumps
- [pothole_dataset_2](https://huggingface.co/datasets/rupesh002/pothole_dataset_2) (MIT), [Pothole-detection-Yolov8](https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8),
  [pothole-segmentation](https://huggingface.co/datasets/keremberke/pothole-segmentation): close-up potholes

Trained with [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8, which is AGPL-3.0.

## Older models (fallback)

Without the model above, `vision_detector.py` uses these two YOLOv8 models when their files are here:

| File | Used for | Size | Without it |
|---|---|---|---|
| `pothole_1class.onnx` | potholes | 45 MB | falls back to the rule-based pothole detector |
| `road_anomaly_7class.onnx` | speed bumps (its other classes are ignored) | 99 MB | speed bumps are skipped |

Cracks are then found by the image rules in `vision_detector.py`. These files are too big for git.

## Download

```bash
curl -L -o models/pothole_1class.onnx https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.onnx
curl -L -o models/road_anomaly_7class.onnx https://huggingface.co/Logesshhh/road-anomaly-pothole-yolov8m/resolve/main/pothole_yolov8.onnx
```

SHA-256:
- `pothole_1class.onnx`: `91dd7de7a110c61314ea19a958fcc85c7b3461cc6d60d87fefc08b41ac6e32c5`
- `road_anomaly_7class.onnx`: `8dbb3232653788ba1b3908ac2d253c24baaf7d772e6b286593938c03e3121555`

## Speed bump model

- Source: [Logesshhh/road-anomaly-pothole-yolov8m](https://huggingface.co/Logesshhh/road-anomaly-pothole-yolov8m), Apache-2.0 licence
- YOLOv8m, 7 classes: Heavy-Vehicle, Light-Vehicle, Pedestrian, Crack, Crack-Severe, Pothole, Speed-Bump
- About 0.65 s per photo on a CPU, so it only runs when speed bumps are requested
- On the 15 real speed bump photos in `test_images/real/bump/` it finds 3 (50% cut-off) with no
  false alarms on the pothole and crack photos. It misses most painted humps and rubber bumps.
  For bumps, the accelerometer (`sensor_detector.py`) is the more reliable signal.

## About the model

- Source: [peterhdd/pothole-detection-yolov8](https://huggingface.co/peterhdd/pothole-detection-yolov8), Apache-2.0 licence
- YOLOv8s, one class (pothole), 640 x 640 input, trained on the
  [Ryukijano/Pothole-detection-Yolov8](https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8) dataset
- Runs through OpenCV's DNN module (`pothole_model.py`), about 0.1 s per photo on a CPU; no PyTorch needed

On the 25 real photos in `test_images/real/` (confidence cut-off 50%):

| | Result |
|---|---|
| Pothole photos with a box on a real pothole | 8 of 12 |
| Wrong boxes on pothole photos | 0 |
| Crack photos with a pothole box | 4 of 13 (all on badly broken or patched asphalt) |
| Missed | deep sinkhole, gravel-filled hole, water-filled hole, very close close-up |
