### MobileFaceNet_TF

Tensorflow implementation for MobileFaceNet.

## dependencies

- tensorflow >= r1.5
- opencv-python 3.x
- python 3.x
- scipy
- sklearn
- numpy
- mxnet
- pickle

## Prepare dataset

1. choose one of the following links to download dataset which is provide by insightface. (Special Recommend MS1M-refine-v2)
* [MS1M-refine-v2@BaiduDrive](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ), [MS1M-refine-v2@GoogleDrive](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
* [Refined-MS1M@BaiduDrive](https://pan.baidu.com/s/1nxmSCch), [Refined-MS1M@GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view)
* [VGGFace2@BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [VGGFace2@GoogleDrive](https://www.dropbox.com/s/m9pm1it7vsw3gj0/faces_vgg2_112x112.zip?dl=0)
* [Insightface Dataset Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
2. move dataset to `${MobileFaceNet_TF_ROOT}/datasets`.
3. run `${MobileFaceNet_TF_ROOT}/utils/data_process.py`.

## pretrained model

* [pretrained_model](https://github.com/sirius-ai/MobileFaceNet_TF/tree/master/arch/pretrained_model/)

## training

1. refined super parameters by yourself special project.
2. run script
`${MobileFaceNet_TF_ROOT}/train_nets.py`
3. have a snapshot result at `${MobileFaceNet_TF_ROOT}/output`.

## performance

|  size  | LFW(%) | Val@1e-3(%) | inference@MSM8976-cpu(ms) |
| ------ | ------ | ----------- | --------------------- |
|  5.7M  |  99.4+ |    98.4+    |          260-         |

## Live RTSP face recognition (`live_face_pipeline.py`)

A standalone live face detection + recognition pipeline built on top of the pretrained
MobileFaceNet model, for real-time RTSP camera streams (not part of the original repo).

Pipeline: GStreamer (rtspsrc/H.265) → appsink → YOLOv8n-face detector (box + 5-point
landmarks) → IoU tracker → quality/pose/landmark-confidence gate → 112×112 similarity
alignment → MobileFaceNet embedding → per-person calibrated cosine-similarity match
against a `enrolled_faces/<name>/*.jpg` gallery cached in LanceDB.

### Setup

- GStreamer 1.0 (with the `msvc_x86_64` runtime + `gst-python`/PyGObject bindings) installed
  and discoverable — see the `GSTREAMER_BIN` fallback paths at the top of the script.
- Python deps beyond the base list above: `ultralytics`, `lancedb`, `pandas`, `pyarrow`, `gi`
  (PyGObject).
- A face-detector weight with a pose/keypoint head at `face_models/yolov8n-face.pt` (the only
  bundled YOLO variant trained with 5-point landmarks; plain bbox-only weights won't give
  alignment or the frontal-pose/landmark-confidence quality gates).

### Camera configuration

RTSP connection settings (camera URL/credentials, jitterbuffer latency) live in
**[`camera.config`](camera.config)** (INI format) — edit that one file to point at a different
camera. Both `live_face_pipeline.py` and `RTSP_correct.py` read their defaults from it via
`configparser`, so credentials aren't duplicated across scripts. The `RTSP_URL` env var, or the
`--rtsp_url` / `--latency` CLI flags, override whatever is set there if you need a one-off change
without editing the file.

### Enrollment

Add reference photos per person under `enrolled_faces/<name>/*.jpg` (see
`enrolled_faces/README.md`). A few photos per person, front-facing and reasonably sharp, work
best. `enrolled_faces/<name>/` folder names become the recognized labels.

### Running

```
python live_face_pipeline.py                      # live window, RTSP_URL env var or --rtsp_url
python live_face_pipeline.py --headless --max_seconds 30   # no GUI, timed test run
```

Detected faces are saved under `detected_faces/<label>/`, where `<label>` is the recognized
name or one of `Unknown` / `LowQuality` / `OffAngle` / `NotAFace` / `Tracking...` depending on
which gate a given detection failed (or passed).

### Key tunables (`--help` for the full list)

| Flag | Purpose |
| --- | --- |
| `--conf` | YOLO face detection confidence |
| `--min_face_pixels`, `--blur_threshold` | reject faces too small/blurry to embed reliably |
| `--pose_threshold` | reject faces turned too far from frontal |
| `--landmark_conf_threshold` | reject detections that don't actually look like a face (filters non-face false positives) |
| `--track_skip_frames`, `--track_recheck_interval` | avoid recognizing a face the instant it appears (blurry entrance frames), and avoid re-running recognition every single frame |
| `--recog_threshold`, `--recog_margin` | fallback match threshold + minimum score gap over the runner-up identity |
| `--threshold_low`, `--threshold_high` | per-person calibrated threshold range (auto-derived per identity from how tightly their own enrolled photos cluster — see `calibrate_person_thresholds()`) |
| `--gallery_db` | LanceDB cache directory; only new/changed enrolled photos are re-embedded on startup |

Thresholds are gallery-specific: they're calibrated from whichever people are currently
enrolled, so re-check them (`calibrate_person_thresholds()`, or just watch the
`<name>: threshold=...` startup log lines) after adding or removing enrolled identities.

## References

1. [facenet](https://github.com/davidsandberg/facenet)
2. [InsightFace mxnet](https://github.com/deepinsight/insightface)
3. [InsightFace_TF](https://github.com/auroua/InsightFace_TF)
4. [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)
5. [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)
6. [InsightFace : Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
