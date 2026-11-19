# Road defect detection: potholes, bumps, cracks

Two detectors that work together, because each sensor sees a different part of the problem:

| Defect | Accelerometer (`sensor_detector.py`) | Camera (`vision_detector.py`) |
|---|---|---|
| Pothole | yes (drop, then impact) | yes: trained YOLOv8 pothole model (8 of 12 real test photos) |
| Speed bump | yes (lift, then drop) | partly: painted / yellow-black striped bumps, plus a trained model |
| Rough road / gravel | yes | partly |
| Cracks (longitudinal / transverse / alligator) | no (too small to feel) | yes: image rules, limited to the road surface |

```
pip install -r requirements.txt
# trained models (not in git, see models/README.md)
curl.exe -L -o models/pothole_1class.onnx https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.onnx
curl.exe -L -o models/road_anomaly_7class.onnx https://huggingface.co/Logesshhh/road-anomaly-pothole-yolov8m/resolve/main/pothole_yolov8.onnx

python sensor_detector.py --demo                 # synthetic drive, prints detected events
python sensor_detector.py trip.csv --json events.json --plot trip.png
python vision_detector.py --demo                 # writes demo_input.jpg / demo_result.jpg
python vision_detector.py road.jpg -o out.jpg --json defects.json
python vision_detector.py road.jpg -o out.jpg --only pothole,speed_bump
python vision_detector.py dashcam.mp4 -o out.mp4 --roi-top 0.55
python test_images/evaluate.py                   # score all real test photos, save labelled copies
```

## Web console

```
python -m uvicorn app.main:app --app-dir backend --port 8000
python simulator/simulate.py --images-dir test_images/real   # simulated vehicles with real photos
```

- **Map** (`http://127.0.0.1:8000`): every photo a vehicle sends is run through the camera
  detector and stored with labelled boxes. The camera's pothole / speed-bump finding sets the
  incident type over the vehicle's vibration guess; an operator's edit outranks both. The
  incident panel shows what the camera found in each photo and where the type came from.
- **Test a photo** (`http://127.0.0.1:8000/test`): drop photos, see the labelled result, and
  add them to the map (location from the photo's GPS data, or typed in). Test photos are not
  saved unless you add them.
- `RDM_DETECT=pothole,crack,speed_bump` chooses what the server's camera detector marks;
  `RDM_ANNOTATE_IMAGES=0` turns it off.

Real test photos live in `test_images/real/{pothole,crack,bump}/` (freely licensed, see
`CREDITS.tsv`). Put your own regression photos in `test_images/user/<type>/` (not committed).

## 1. Accelerometer algorithm (potholes vs. bumps)

1. **Works at any phone angle**: a 0.3 Hz low-pass gives the gravity direction. Projecting each sample onto it gives the vertical acceleration.
2. **Band-pass 1–15 Hz** (zero-phase Butterworth) keeps wheel impacts and removes gravity, braking and turns.
3. **Adaptive threshold** = 5 × robust σ, where σ comes from the median absolute deviation (MAD) over a rolling 10 s window. The floor is 0.25 g.
4. **Grouping**: hits less than 0.5 s apart become one event, so the front and rear wheels count once.
5. **Classification** uses the sign of the first strong lobe:
   - **pothole**: the wheel drops first, so the signal goes **negative** (free fall), then spikes positive on impact
   - **bump**: the wheel is pushed up first, so the signal goes **positive**, then negative
   - **rough**: the event lasts more than 2 s and crosses zero more than 8 times per second
6. **Severity** comes from the peak-to-peak size: < 0.5 g low, < 1 g medium, otherwise high. GPS and speed are attached at the peak.

CSV format: `time,ax,ay,az[,lat,lon,speed]`. Time can be in s or ms. Acceleration is in m/s² (use `--units g` for g). Record at **≥ 50 Hz, ideally 100 Hz**. Phone apps that export this: *Physics Toolbox Sensor Suite*, *phyphox*, *Sensor Logger*.

Tuning: use a lower `--k` for more sensitivity. Speed changes the amplitude: the same hole feels about 2× stronger at 60 km/h than at 20 km/h. For a fair severity score across a fleet, normalise `peak_to_peak_g` by speed.

## 2. Vision algorithm (classical OpenCV, no training data)

**Cracks**
1. CLAHE equalises shadows and uneven lighting.
2. A **black-hat** transform (15 px kernel) keeps thin dark lines and removes the smooth asphalt.
3. Threshold = median + 5 × MAD of that response, so it adapts to the asphalt's grain.
4. Each connected piece is skeletonised. It counts as a crack if it is **long** (skeleton ≥ 40 px) and **thin** (area ÷ length ≤ 8 px).
5. Type:
   - longitudinal or transverse, from the main direction of the skeleton (the camera is assumed to look along the road)
   - **alligator**, when the crack pattern encloses ≥ 3 cells (a mesh)

**Potholes**
1. Local background = a very large median blur. "Darkness" = background − image.
2. Threshold, then morphological open/close.
3. A blob is kept only if it is large, solid (≥ 0.6), not line-like (aspect ≤ 4), more textured than the road (this rejects shadows and fresh patches), and ≥ 50 % dark pixels (this rejects closed-up crack meshes).

## 3. Test results (synthetic data)

- **Sensor**: 4 potholes, 2 speed bumps and 1 gravel stretch, phone tilted 35°/−20°, 4 random seeds. Every event was found and correctly classified, with no false positives.
- **Vision**: 10 random seeds. All 10 had exactly one pothole, plus the longitudinal, transverse and alligator cracks, with no false positives. In 2 seeds the long crack was reported as 2–3 segments.

This is synthetic data. On real roads, expect to tune `min_dark`, `k_sigma` and `--roi-top`.

## 4. Limits of classical CV, and the production upgrade

Classical CV has trouble with lane markings, tar sealant lines, wet roads, strong shadows and manhole covers. For real deployment, train a detector:

- **Dataset**: [RDD2022](https://github.com/sekilab/RoadDamageDetector), about 47k road images from 6 countries. Labels: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole.
  Also: Kaggle "Pothole Detection" datasets, and CrackForest / DeepCrack for pixel-level crack masks.
- **Model**: YOLOv8 or YOLO11 (detection) via `ultralytics`:
  ```
  pip install ultralytics
  yolo detect train data=rdd2022.yaml model=yolov8s.pt imgsz=640 epochs=100
  yolo detect predict model=runs/detect/train/weights/best.pt source=dashcam.mp4
  ```
  For crack width and length measurement, use a segmentation model (U-Net, or YOLOv8-seg).
- **Best system**: fuse both. A camera detection confirmed by an accelerometer event at the same GPS point (within about 10 m) is very unlikely to be a false alarm. Count repeat detections from several vehicles before marking a defect on the map.
