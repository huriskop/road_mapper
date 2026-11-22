# Trained models

`vision_detector.py` uses two trained YOLOv8 models when their files are here:

| File | Used for | Size | Without it |
|---|---|---|---|
| `pothole_1class.onnx` | potholes | 45 MB | falls back to the rule-based pothole detector |
| `road_anomaly_7class.onnx` | speed bumps (its other classes are ignored) | 99 MB | speed bumps are skipped |

Cracks are found by the image rules in `vision_detector.py` and need no model.
The files are too big for git, so they are not committed.

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
