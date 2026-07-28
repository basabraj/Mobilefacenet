# -*- coding: utf-8 -*-
'''
Live RTSP face detection + recognition using MobileFaceNet embeddings.

Pipeline: GStreamer (rtspsrc/H.265) -> appsink -> YOLOv8n-face detector
(box + 5-point landmarks: eyes/nose/mouth-corners) -> frontal-pose + quality
gate -> crop -> MobileFaceNet frozen graph -> embedding -> cosine-similarity
match against enrolled_faces/<name>/*.jpg gallery.

Note: yolov8n-face.pt is the only bundled face-detector weight trained with a
pose/keypoint head (5-point landmarks). The v11/v12 face weights in face_models/
are plain bbox-only detectors, so this pipeline is pinned to yolov8n-face.pt.

RTSP credentials are read from the RTSP_URL env var. Falls back to the
camera already hardcoded in RTSP_correct.py for convenience, but prefer
setting the env var so credentials aren't duplicated in source.
'''

import os
import sys
import time
import glob
import hashlib
import argparse
import datetime

# --- Windows: make the GStreamer DLLs / typelibs discoverable before `import gi` ---
if os.name == 'nt':
    _default_gst_bins = [
        os.environ.get('GSTREAMER_BIN', ''),
        os.path.expandvars(r'%LOCALAPPDATA%\Programs\gstreamer\1.0\msvc_x86_64\bin'),
        r'C:\gstreamer\1.0\msvc_x86_64\bin',
        r'C:\Program Files\GStreamer\1.0\msvc_x86_64\bin',
    ]
    _gst_bin = next((p for p in _default_gst_bins if p and os.path.isdir(p)), None)
    if _gst_bin is None:
        raise RuntimeError(
            'GStreamer bin directory not found. Set GSTREAMER_BIN env var to '
            'e.g. C:\\Users\\<you>\\AppData\\Local\\Programs\\gstreamer\\1.0\\msvc_x86_64\\bin'
        )
    os.add_dll_directory(_gst_bin)
    os.environ['PATH'] = _gst_bin + os.pathsep + os.environ.get('PATH', '')
    _gst_root = os.path.dirname(_gst_bin)
    os.environ.setdefault('GI_TYPELIB_PATH', os.path.join(_gst_root, 'lib', 'girepository-1.0'))

import numpy as np
import cv2
import tensorflow as tf

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp  # noqa: F401  (GstApp import registers appsink.try_pull_sample etc.)

from ultralytics import YOLO
import lancedb
import pyarrow as pa

tf1 = tf.compat.v1
tf1.disable_eager_execution()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RTSP_URL = 'rtsp://admin:Digital%40123@192.168.96.83:554/live1.sdp'
DEFAULT_FACENET_PB = os.path.join(SCRIPT_DIR, 'arch', 'pretrained_model', 'MobileFaceNet_9925_9680.pb')
DEFAULT_YOLO_WEIGHTS = os.path.join(SCRIPT_DIR, 'face_models', 'yolov8n-face.pt')
DEFAULT_SAVE_DIR = os.path.join(SCRIPT_DIR, 'detected_faces')
DEFAULT_ENROLL_DIR = os.path.join(SCRIPT_DIR, 'enrolled_faces')
DEFAULT_GALLERY_DB = os.path.join(SCRIPT_DIR, 'gallery_db')
EMBEDDING_SIZE_HINT = 112
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
GALLERY_TABLE = 'faces'
EMBEDDING_DIM = 128


def get_parser():
    parser = argparse.ArgumentParser(description='Live RTSP face detect + MobileFaceNet embedding')
    parser.add_argument('--rtsp_url', default=os.environ.get('RTSP_URL', DEFAULT_RTSP_URL),
                        help='RTSP camera URL (prefer setting RTSP_URL env var instead)')
    parser.add_argument('--facenet_pb', default=DEFAULT_FACENET_PB, help='frozen MobileFaceNet .pb path')
    parser.add_argument('--yolo_weights', default=DEFAULT_YOLO_WEIGHTS, help='YOLO face detector weights path')
    parser.add_argument('--conf', type=float, default=0.5, help='YOLO face detection confidence threshold')
    parser.add_argument('--latency', type=int, default=200, help='rtspsrc jitterbuffer latency (ms)')
    parser.add_argument('--frame_skip', type=int, default=0,
                        help='process every (frame_skip + 1)-th frame; higher = faster but choppier')
    parser.add_argument('--max_seconds', type=float, default=0,
                        help='stop after this many seconds (0 = run until q / stream end)')
    parser.add_argument('--headless', action='store_true',
                        help='skip cv2.imshow display (useful for automated / no-GUI test runs)')
    parser.add_argument('--save_dir', default=DEFAULT_SAVE_DIR,
                        help='folder to store cropped face images whenever a face is detected')
    parser.add_argument('--no_enhance', action='store_true',
                        help='skip saving a *_enhanced.jpg (upscaled/brightened/sharpened, for manual review only -- '
                             'never used for embedding) alongside each raw saved crop')
    parser.add_argument('--enhance_upscale', type=int, default=3,
                        help='upscale factor for the *_enhanced.jpg review copy')
    parser.add_argument('--enroll_dir', default=DEFAULT_ENROLL_DIR,
                        help='folder of enrolled_faces/<name>/*.jpg used as the recognition gallery')
    parser.add_argument('--gallery_db', default=DEFAULT_GALLERY_DB,
                        help='LanceDB directory that persistently caches enrolled embeddings. On startup only '
                             'new/changed files under --enroll_dir are re-embedded; the rest is loaded from here '
                             'instead of re-running the network on every enrolled photo every time.')
    parser.add_argument('--recog_threshold', type=float, default=0.6,
                        help='fallback cosine similarity threshold, used only for an identity with too few enrolled '
                             'photos to auto-calibrate its own threshold (see calibrate_person_thresholds())')
    parser.add_argument('--threshold_low', type=float, default=0.6,
                        help='floor for the per-person calibrated threshold, assigned to the most spread-out '
                             '(least consistent) enrolled identity. hailo-ai\'s reference default (0.1) is a pure '
                             'relative ranking with no floor -- whichever identity is least consistent always gets '
                             'pinned this low even after curating outlier photos, which is too permissive for our '
                             'small (3-person) gallery. Set to 0.6 (an earlier 0.35 floor was still measured BELOW '
                             'the observed inter-person max of 0.569 on this gallery -- i.e. still mathematically '
                             'capable of a cross-person false match; 0.6 matches the global default threshold and '
                             'sits safely above that measured ceiling).')
    parser.add_argument('--threshold_high', type=float, default=0.65,
                        help='per-person calibrated threshold assigned to the most tightly-clustered (most '
                             'consistent) enrolled identity. hailo-ai\'s reference default (0.9) assumes a tight '
                             'relative cluster also means high absolute similarity -- measured false on our data: '
                             'at 0.9, Shraban_Sir (our tightest-clustered identity) had 7/14 of his OWN enrolled '
                             'photos score BELOW his own threshold (mean similarity to his centroid was only 0.865, '
                             'min 0.673) -- i.e. a ~50% chance of being told "Unknown" while looking at himself. '
                             '0.65 sits below that measured floor (0.673) while staying above the measured '
                             'inter-person max (0.569).')
    parser.add_argument('--recog_margin', type=float, default=0.05,
                        help='minimum score gap the best match must have over the runner-up identity; '
                             'closer calls fall back to Unknown instead of guessing between two enrolled people')
    parser.add_argument('--min_face_pixels', type=int, default=3000,
                        help='minimum face bbox area (px^2); smaller/farther faces skip recognition entirely. '
                             'Calibrated empirically for this camera (real faces measured ~5000-6000px, a '
                             'background/far face measured ~1000-1200px) -- the hailo-ai reference repo default '
                             'of 12000 was tuned for a different camera setup and rejected every real face here.')
    parser.add_argument('--blur_threshold', type=float, default=50.0,
                        help='minimum Laplacian variance (sharpness); blurrier faces skip recognition entirely. '
                             'Calibrated empirically for this camera (legit in-focus faces measured 60-520).')
    parser.add_argument('--track_iou_threshold', type=float, default=0.3,
                        help='min IoU between frames to consider a detection the same tracked face')
    parser.add_argument('--track_skip_frames', type=int, default=5,
                        help='frames to wait after a face first appears before attempting recognition '
                             '(avoids the blurry/half-turned frames right as someone enters the scene)')
    parser.add_argument('--track_recheck_interval', type=int, default=15,
                        help='once a track has an identity, re-run recognition again after this many frames '
                             '(instead of every frame) -- cuts redundant recognition attempts and saved images')
    parser.add_argument('--track_max_missed', type=int, default=10,
                        help='drop a track if it goes unmatched for this many consecutive frames (person left frame)')
    parser.add_argument('--track_embedding_window', type=int, default=5,
                        help='average the last N good-quality embeddings of a track before the gallery lookup '
                             '(rolling window) instead of matching on a single frame -- damps single-frame noise '
                             'that can otherwise push a correct identity just under its threshold')
    parser.add_argument('--pose_threshold', type=float, default=0.15,
                        help='minimum frontal-pose symmetry score (0=profile, 1=dead-on frontal) from the 5-point '
                             'landmarks (eyes/nose/mouth-corners); more angled faces skip recognition entirely '
                             '(lowered from 0.4: measured on real OffAngle-rejected crops, e.g. "looking down at '
                             'monitor" posture, 3/4 would have correctly matched at 0.15)')
    parser.add_argument('--landmark_conf_threshold', type=float, default=0.5,
                        help='minimum mean confidence of the 5 landmark points. Catches false-positive "face" '
                             'detections on non-face objects (measured ~0.25-0.4 on a bag/chair in testing) that '
                             'the box confidence alone let through -- real faces measured ~0.8-0.9.')
    return parser.parse_args()


def load_facenet(pb_path):
    with tf1.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())
    graph = tf1.Graph()
    with graph.as_default():
        tf1.import_graph_def(graph_def, name='')
    sess = tf1.Session(graph=graph)
    input_tensor = graph.get_tensor_by_name('input:0')
    embeddings_tensor = graph.get_tensor_by_name('embeddings:0')
    return sess, input_tensor, embeddings_tensor


def build_pipeline(rtsp_url, latency):
    pipeline_str = (
        'rtspsrc location="{url}" latency={latency} name=source ! '
        'rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! '
        'video/x-raw,format=BGR ! '
        'appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true'
    ).format(url=rtsp_url, latency=latency)
    return Gst.parse_launch(pipeline_str)


def pull_frame(appsink, timeout_ns=Gst.SECOND):
    sample = appsink.try_pull_sample(timeout_ns)
    if sample is None:
        return None
    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value('width')
    height = structure.get_value('height')
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3)).copy()
    finally:
        buf.unmap(mapinfo)
    return frame


# Canonical 112x112 landmark template MobileFaceNet/ArcFace-style models are trained against
# (left_eye, right_eye, nose, left_mouth, right_mouth). Matches the order our YOLO pose model
# outputs. Without warping detected landmarks onto this template, raw bbox crops carry whatever
# arbitrary in-plane rotation/scale/off-centering the detector happened to produce -- the network
# was never trained to be invariant to that, so the same person's embedding drifts between frames
# and can drift into a different enrolled person's neighborhood.
REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image_bgr, landmarks, output_size=EMBEDDING_SIZE_HINT):
    '''Similarity-transform (rotation + uniform scale + translation) the face onto the canonical
    112x112 landmark template using the 5 detected points. Returns None if the transform can't
    be estimated (degenerate/collinear landmarks).
    '''
    src = np.array(landmarks, dtype=np.float32)
    dst = REFERENCE_LANDMARKS_112 * (output_size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(image_bgr, M, (output_size, output_size), borderValue=0.0)


def enhance_for_review(face_bgr, upscale=3):
    '''Upscale + brighten + sharpen a saved face crop purely so a human can see it more clearly
    when reviewing detected_faces/. Never used for embedding/recognition -- classic Lanczos
    upscale + CLAHE (local contrast, helps dark CCTV crops) + mild unsharp mask. Chosen over
    CodeFormer/GFPGAN-style restoration after testing: on our small, oddly-framed CCTV crops
    CodeFormer's internal re-detection/alignment produced distorted, hallucinated results (and
    took ~2 minutes per face on CPU) -- actively worse for verification, not better. This classic
    combo can't hallucinate detail that isn't there, and is effectively instant.
    '''
    h, w = face_bgr.shape[:2]
    upscaled = cv2.resize(face_bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4)

    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    return cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)


def embed_face(sess, input_tensor, embeddings_tensor, face_bgr):
    '''Embed an already-cropped (and ideally already-aligned) face; resizes if needed.
    Prefer get_face_embedding() when landmarks are available so the face is properly aligned.
    '''
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (EMBEDDING_SIZE_HINT, EMBEDDING_SIZE_HINT))
    face_norm = (face_resized.astype(np.float32) - 127.5) * 0.0078125
    emb = sess.run(embeddings_tensor, feed_dict={input_tensor: face_norm[None, ...]})[0]
    return emb


def get_face_embedding(sess, input_tensor, embeddings_tensor, image_bgr, box, landmarks):
    '''Preferred embedding path: similarity-align the face via its 5 landmarks (computed from
    the full image, not the crop, for accurate geometry) before embedding. Falls back to a
    plain bbox crop+resize if landmarks aren't available or the transform is degenerate.
    '''
    if landmarks is not None:
        aligned = align_face(image_bgr, landmarks)
        if aligned is not None:
            return embed_face(sess, input_tensor, embeddings_tensor, aligned)
    if box is None:
        return embed_face(sess, input_tensor, embeddings_tensor, image_bgr)
    x1, y1, x2, y2 = box
    return embed_face(sess, input_tensor, embeddings_tensor, image_bgr[y1:y2, x1:x2])


def face_quality_ok(face_bgr, min_pixels, blur_threshold):
    '''Reject faces that are too small (too far from camera) or too blurry to embed reliably.

    Mirrors the "cropper quality gate" idea from hailo-ai/hailo-apps' face_recognition pipeline:
    low-quality inputs produce noisy embeddings, which is a bigger source of wrong matches than
    the similarity threshold itself.
    '''
    h, w = face_bgr.shape[:2]
    if h * w < min_pixels:
        return False, 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return blur_score >= blur_threshold, blur_score


LANDMARK_NAMES = ('left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth')


def frontal_pose_score(landmarks):
    '''Score how front-facing a face is from its 5 landmarks (left_eye, right_eye, nose,
    left_mouth, right_mouth), using nose-symmetry: how centered the nose is between the
    eyes and between the mouth corners. 1.0 = perfectly frontal, drops toward 0.0 as the
    face turns to profile. Cheap standalone alternative to the procrustes-based frontal
    check hailo-ai's reference pipeline does in C++.
    '''
    eye_l, eye_r, nose, mouth_l, mouth_r = landmarks

    def symmetry(a, mid, b):
        left, right = mid[0] - a[0], b[0] - mid[0]
        if left <= 0 or right <= 0:
            return 0.0
        return min(left, right) / max(left, right)

    return (symmetry(eye_l, nose, eye_r) + symmetry(mouth_l, nose, mouth_r)) / 2.0


def get_face_detections(yolo_model, image_bgr, conf):
    '''Run the face detector and return a list of (box, landmarks_or_None, landmark_conf_or_None,
    det_conf).

    landmarks is a 5x2 array [left_eye, right_eye, nose, left_mouth, right_mouth] in image
    pixel coords, and landmark_conf the per-point confidence, if the loaded model has a
    pose/keypoint head, else None. Real faces measured ~0.8-0.9 mean landmark confidence;
    false-positive detections on non-face objects (a bag, a chair) measured ~0.25-0.4 --
    a much cleaner signal for "is this actually a face" than the box confidence alone,
    which YOLO can still emit high on out-of-distribution crops. det_conf is that raw YOLO
    box/class confidence (0-1), kept separately for the "face NN%" overlay label.
    '''
    results = yolo_model.predict(image_bgr, verbose=False, conf=conf)[0]
    detections = []
    has_kpts = results.keypoints is not None
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image_bgr.shape[1]), min(y2, image_bgr.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue
        landmarks = results.keypoints.xy[i].tolist() if has_kpts else None
        landmark_conf = results.keypoints.conf[i].tolist() if has_kpts and results.keypoints.conf is not None else None
        det_conf = float(box.conf[0]) if box.conf is not None else 0.0
        detections.append(((x1, y1, x2, y2), landmarks, landmark_conf, det_conf))
    return detections


def detect_largest_face(yolo_model, image_bgr, conf):
    detections = get_face_detections(yolo_model, image_bgr, conf)
    if not detections:
        return None, None, None, None
    return max(detections, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))


def _embed_enrollment_photo(fpath, name, yolo_model, sess, input_tensor, embeddings_tensor, conf,
                            min_face_pixels, blur_threshold, pose_threshold, landmark_conf_threshold=0.0):
    '''Load one enrollment photo, run it through the same detect/quality/pose gate as live frames,
    and return its embedding, or None (with an explanatory print) if it should be skipped.
    '''
    image = cv2.imread(fpath)
    if image is None:
        print('  WARNING: could not read %s, skipping' % fpath)
        return None

    box, landmarks, landmark_conf, _det_conf = detect_largest_face(yolo_model, image, conf)
    face = image[box[1]:box[3], box[0]:box[2]] if box is not None else image
    if face.size == 0:
        return None

    if landmark_conf is not None:
        mean_lm_conf = sum(landmark_conf) / len(landmark_conf)
        if mean_lm_conf < landmark_conf_threshold:
            print('  WARNING: %s <- %s does not look like a real face (landmark_conf=%.2f), skipping enrollment photo'
                  % (name, os.path.basename(fpath), mean_lm_conf))
            return None

    ok, blur_score = face_quality_ok(face, min_face_pixels, blur_threshold)
    if not ok:
        print('  WARNING: %s <- %s is too small/blurry (blur=%.0f), skipping enrollment photo'
              % (name, os.path.basename(fpath), blur_score))
        return None

    if landmarks is not None:
        pose_score = frontal_pose_score(landmarks)
        if pose_score < pose_threshold:
            print('  WARNING: %s <- %s face is too angled (pose=%.2f), skipping enrollment photo'
                  % (name, os.path.basename(fpath), pose_score))
            return None

    return get_face_embedding(sess, input_tensor, embeddings_tensor, image, box, landmarks)


def build_gallery(enroll_dir, yolo_model, sess, input_tensor, embeddings_tensor, conf,
                  min_face_pixels=0, blur_threshold=0.0, pose_threshold=0.0, landmark_conf_threshold=0.0):
    '''Scan enroll_dir/<name>/*.jpg, embed every enrolled photo. Returns list of (name, embedding).

    Re-embeds every photo on every call -- use sync_gallery_db() instead for a persistent,
    incrementally-updated gallery that skips re-embedding unchanged photos.
    '''
    gallery = []
    if not os.path.isdir(enroll_dir):
        return gallery

    for name in sorted(os.listdir(enroll_dir)):
        person_dir = os.path.join(enroll_dir, name)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            emb = _embed_enrollment_photo(os.path.join(person_dir, fname), name, yolo_model, sess,
                                          input_tensor, embeddings_tensor, conf,
                                          min_face_pixels, blur_threshold, pose_threshold, landmark_conf_threshold)
            if emb is not None:
                gallery.append((name, emb))
                print('  enrolled %s <- %s' % (name, fname))

    return gallery


def sync_gallery_db(db_path, enroll_dir, yolo_model, sess, input_tensor, embeddings_tensor, conf,
                    min_face_pixels=0, blur_threshold=0.0, pose_threshold=0.0, landmark_conf_threshold=0.0):
    '''Persistent, incrementally-updated version of build_gallery() backed by LanceDB.

    Only embeds photos under enroll_dir that aren't already cached in the DB, and drops DB
    rows whose source photo was deleted from enroll_dir. Returns list of (name, embedding),
    same shape as build_gallery(), for build_centroids() to consume.
    '''
    db = lancedb.connect(db_path)
    table = db.open_table(GALLERY_TABLE) if GALLERY_TABLE in db.table_names() else None

    current_files = {}
    if os.path.isdir(enroll_dir):
        for name in sorted(os.listdir(enroll_dir)):
            person_dir = os.path.join(enroll_dir, name)
            if not os.path.isdir(person_dir):
                continue
            for fpath in sorted(glob.glob(os.path.join(person_dir, '*'))):
                if fpath.lower().endswith(IMAGE_EXTS):
                    current_files[hashlib.md5(fpath.encode('utf-8')).hexdigest()] = (name, fpath)

    existing_ids = set()
    if table is not None:
        existing_ids = set(table.to_pandas()['id'].tolist())

    to_add_ids = set(current_files) - existing_ids
    to_remove_ids = existing_ids - set(current_files)

    new_rows = []
    for file_id in sorted(to_add_ids):
        name, fpath = current_files[file_id]
        emb = _embed_enrollment_photo(fpath, name, yolo_model, sess, input_tensor, embeddings_tensor, conf,
                                      min_face_pixels, blur_threshold, pose_threshold, landmark_conf_threshold)
        if emb is not None:
            new_rows.append({'id': file_id, 'name': name, 'image_path': fpath,
                             'embedding': emb.astype(np.float32).tolist()})
            print('  enrolled %s <- %s' % (name, os.path.basename(fpath)))

    if new_rows:
        schema = pa.schema([
            pa.field('id', pa.string()),
            pa.field('name', pa.string()),
            pa.field('image_path', pa.string()),
            pa.field('embedding', pa.list_(pa.float32(), EMBEDDING_DIM)),
        ])
        if table is None:
            table = db.create_table(GALLERY_TABLE, data=new_rows, schema=schema)
        else:
            table.add(new_rows)

    if table is not None and to_remove_ids:
        for file_id in to_remove_ids:
            table.delete("id = '%s'" % file_id)

    print('  gallery_db sync: +%d new, -%d removed (of %d cached photos rejected by quality/pose gate: %d)'
         % (len(new_rows), len(to_remove_ids), len(to_add_ids), len(to_add_ids) - len(new_rows)))

    if table is None:
        return []
    df = table.to_pandas()
    return [(row['name'], np.array(row['embedding'], dtype=np.float32)) for row in df.to_dict('records')]


def build_centroids(gallery):
    '''Collapse per-image embeddings into one mean (re-normalized) embedding per person.

    Matching against a single averaged-out embedding per person is much less prone to a
    single noisy/outlier enrolled photo causing a false match than raw nearest-neighbor
    against every individual enrolled image.
    '''
    per_person = {}
    for name, emb in gallery:
        per_person.setdefault(name, []).append(emb)

    centroids = {}
    for name, embs in per_person.items():
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        norm = np.linalg.norm(mean_emb)
        centroids[name] = mean_emb / norm if norm > 0 else mean_emb
    return centroids


def calibrate_person_thresholds(gallery, low=0.1, high=0.9):
    '''Per-identity confidence threshold, auto-calibrated from how spread out each person's own
    enrolled embeddings are -- mirrors hailo-ai's reference db_handler.py
    (calibrate_classification_confidence_threshold): project each person's embeddings onto their
    top-2 principal components, treat the resulting spread as an ellipse area, then map smaller
    area (tight, consistent enrollment photos) to a strict threshold and larger area (more varied
    photos) to a lenient one. A single global threshold otherwise either misses a person whose
    photos are naturally more spread out, or is too permissive for one with tightly clustered
    photos.
    '''
    per_person = {}
    for name, emb in gallery:
        per_person.setdefault(name, []).append(emb)

    names = sorted(per_person.keys())
    areas = []
    for name in names:
        embs = np.stack(per_person[name], axis=0)
        if len(embs) < 2:
            areas.append(0.0)
            continue
        centered = embs - embs.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        std_devs = singular_values[:2] / np.sqrt(max(len(embs) - 1, 1))
        semi_major, semi_minor = (list(std_devs) + [0.0, 0.0])[:2]
        areas.append(float(np.pi * semi_major * semi_minor))

    areas = np.array(areas)
    if len(areas) > 0 and areas.max() != areas.min():
        norm_areas = (areas - areas.min()) / (areas.max() - areas.min())
    else:
        norm_areas = np.zeros_like(areas)

    return {name: low + (high - low) * (1 - norm_area) for name, norm_area in zip(names, norm_areas)}


def recognize(centroids, embedding, person_thresholds, default_threshold, margin=0.05):
    '''Match against per-person centroids using each identity's own calibrated threshold (falling
    back to default_threshold if a person has no calibrated value), with a margin test against
    the runner-up identity so two enrolled people with similar (competing) scores fall back to
    "Unknown" instead of being confused with each other.
    '''
    if not centroids:
        return 'Unknown', 0.0

    scores = sorted(
        ((name, float(np.dot(embedding, c))) for name, c in centroids.items()),
        key=lambda kv: kv[1], reverse=True,
    )
    best_name, best_score = scores[0]
    threshold = person_thresholds.get(best_name, default_threshold)
    if best_score < threshold:
        return 'Unknown', best_score
    if len(scores) > 1 and (best_score - scores[1][1]) < margin:
        return 'Unknown', best_score
    return best_name, best_score


def aggregate_track_embedding(track_embeddings, new_embedding, window):
    '''Append new_embedding to a track's rolling embedding history (capped at `window`, oldest
    dropped first) and return the mean-normalized aggregate. Matching the same live face across
    several good-quality frames and averaging their embeddings before the gallery lookup damps
    single-frame noise -- the same reason enrollment matches against a centroid rather than one
    photo. Without this, an otherwise-correct identity can land just under threshold on one noisy
    frame (observed: 0.600 vs a 0.612 threshold) and get reported as Unknown.
    '''
    track_embeddings.append(new_embedding)
    if len(track_embeddings) > window:
        track_embeddings.pop(0)
    agg = np.mean(track_embeddings, axis=0)
    norm = np.linalg.norm(agg)
    return agg / norm if norm > 0 else agg


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    '''Lightweight IoU-based multi-face tracker (no external dependency).

    Assigns a stable track_id to each face across frames so the pipeline can:
    (a) skip recognition for the first few frames after a face appears (avoids blurry
        entrance frames -- reportedly the single biggest accuracy lever in hailo-ai's
        reference face_recognition pipeline), and
    (b) avoid re-running recognition (and re-saving a crop) every single frame for a
        face that's already been identified, only periodically re-checking it.
    '''

    def __init__(self, iou_threshold, max_missed):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 0

    def update(self, boxes, frame_count):
        unmatched_track_ids = set(self.tracks.keys())
        assigned_ids = []

        for box in boxes:
            best_id, best_iou = None, self.iou_threshold
            for track_id in unmatched_track_ids:
                score = iou(box, self.tracks[track_id]['bbox'])
                if score > best_iou:
                    best_id, best_iou = track_id, score

            if best_id is not None:
                track = self.tracks[best_id]
                track['bbox'] = box
                track['last_seen_frame'] = frame_count
                unmatched_track_ids.discard(best_id)
                assigned_ids.append(best_id)
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'bbox': box, 'first_frame': frame_count, 'last_seen_frame': frame_count,
                    'name': None, 'score': 0.0, 'last_recog_frame': None, 'embeddings': [],
                }
                assigned_ids.append(track_id)

        stale = [tid for tid, t in self.tracks.items() if frame_count - t['last_seen_frame'] > self.max_missed]
        for tid in stale:
            del self.tracks[tid]

        return assigned_ids


def main():
    args = get_parser()

    print('Loading MobileFaceNet frozen graph: %s' % args.facenet_pb)
    sess, input_tensor, embeddings_tensor = load_facenet(args.facenet_pb)

    print('Loading YOLO face detector: %s' % args.yolo_weights)
    yolo_model = YOLO(args.yolo_weights)

    os.makedirs(args.save_dir, exist_ok=True)
    print('Detected faces will be saved to: %s' % args.save_dir)
    save_count = 0

    tracker = SimpleTracker(iou_threshold=args.track_iou_threshold, max_missed=args.track_max_missed)

    print('Syncing recognition gallery from: %s (cached in %s)' % (args.enroll_dir, args.gallery_db))
    gallery = sync_gallery_db(args.gallery_db, args.enroll_dir, yolo_model, sess, input_tensor, embeddings_tensor,
                              args.conf, min_face_pixels=args.min_face_pixels, blur_threshold=args.blur_threshold,
                              pose_threshold=args.pose_threshold,
                              landmark_conf_threshold=args.landmark_conf_threshold)
    centroids = build_centroids(gallery)
    person_thresholds = calibrate_person_thresholds(gallery, low=args.threshold_low, high=args.threshold_high)
    print('Gallery ready: %d identities, %d enrolled images' % (len(centroids), len(gallery)))
    for name in sorted(person_thresholds):
        print('  %s: threshold=%.2f' % (name, person_thresholds[name]))
    if not gallery:
        print('  (empty gallery -> every face will show as Unknown; see enrolled_faces/README.md)')

    Gst.init([])
    pipeline = build_pipeline(args.rtsp_url, args.latency)
    appsink = pipeline.get_by_name('sink')

    bus = pipeline.get_bus()
    pipeline.set_state(Gst.State.PLAYING)
    print('Connecting to %s ...' % args.rtsp_url, flush=True)

    frame_count = 0
    fps_t0 = time.time()
    fps_counter = 0
    fps = 0.0
    start_time = time.time()

    try:
        while True:
            if args.max_seconds and (time.time() - start_time) >= args.max_seconds:
                print('max_seconds reached, stopping', flush=True)
                break

            msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is not None:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    print('[GStreamer ERROR] %s (%s)' % (err, debug), flush=True)
                else:
                    print('[GStreamer] End of stream', flush=True)
                break

            frame = pull_frame(appsink)
            if frame is None:
                continue

            frame_count += 1
            if frame_count == 1:
                print('First frame received: %s' % (frame.shape,), flush=True)
            if args.frame_skip and (frame_count % (args.frame_skip + 1)) != 0:
                continue

            detections = get_face_detections(yolo_model, frame, args.conf)
            boxes = [d[0] for d in detections]
            landmarks_list = [d[1] for d in detections]
            landmark_conf_list = [d[2] for d in detections]
            det_conf_list = [d[3] for d in detections]

            track_ids = tracker.update(boxes, frame_count)

            for (x1, y1, x2, y2), landmarks, landmark_conf, det_conf, track_id in zip(
                boxes, landmarks_list, landmark_conf_list, det_conf_list, track_ids
            ):
                track = tracker.tracks[track_id]
                frames_tracked = frame_count - track['first_frame']
                due_for_recheck = (track['last_recog_frame'] is None or
                                   frame_count - track['last_recog_frame'] >= args.track_recheck_interval)

                if frames_tracked < args.track_skip_frames:
                    name, score = 'Tracking...', 0.0
                elif due_for_recheck:
                    face = frame[y1:y2, x1:x2]
                    mean_lm_conf = (sum(landmark_conf) / len(landmark_conf)) if landmark_conf is not None else 1.0
                    quality_ok, blur_score = face_quality_ok(face, args.min_face_pixels, args.blur_threshold)
                    pose_score = frontal_pose_score(landmarks) if landmarks is not None else 1.0
                    if mean_lm_conf < args.landmark_conf_threshold:
                        name, score = 'NotAFace', mean_lm_conf
                    elif not quality_ok:
                        name, score = 'LowQuality', blur_score
                    elif pose_score < args.pose_threshold:
                        name, score = 'OffAngle', pose_score
                    else:
                        emb = get_face_embedding(sess, input_tensor, embeddings_tensor, frame,
                                                 (x1, y1, x2, y2), landmarks)
                        agg_emb = aggregate_track_embedding(track['embeddings'], emb, args.track_embedding_window)
                        name, score = recognize(centroids, agg_emb, person_thresholds, args.recog_threshold,
                                                args.recog_margin)

                    track['name'], track['score'], track['last_recog_frame'] = name, score, frame_count

                    person_dir = os.path.join(args.save_dir, name)
                    os.makedirs(person_dir, exist_ok=True)
                    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                    save_path = os.path.join(person_dir, 'face_%s_track%d.jpg' % (ts, track_id))
                    cv2.imwrite(save_path, face)
                    if not args.no_enhance:
                        enhanced_path = os.path.join(person_dir, 'face_%s_track%d_enhanced.jpg' % (ts, track_id))
                        cv2.imwrite(enhanced_path, enhance_for_review(face, upscale=args.enhance_upscale))
                    save_count += 1
                else:
                    name, score = track['name'], track['score']

                if name == 'Tracking...':
                    box_color = (255, 165, 0)
                elif name == 'NotAFace':
                    box_color = (0, 0, 0)
                elif name == 'LowQuality':
                    box_color = (128, 128, 128)
                elif name == 'OffAngle':
                    box_color = (255, 0, 255)
                elif name == 'Unknown':
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Above the box (outside): raw YOLO detection confidence, e.g. "face 73%"
                # -- matches the hailo-ai reference app's overlay layout.
                cv2.putText(frame, 'face %d%%' % round(det_conf * 100), (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Inside the box, near the top: recognized identity + its confidence
                if name in ('Tracking...', 'NotAFace', 'LowQuality', 'OffAngle'):
                    id_label = '%s (%.2f)' % (name, score)
                else:
                    id_label = '%s %d%%' % (name, round(max(0.0, min(1.0, score)) * 100))
                cv2.putText(frame, id_label, (x1 + 4, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Inside the box, near the bottom: track id
                cv2.putText(frame, str(track_id), (x1 + 4, y2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                if landmarks is not None:
                    for lx, ly in landmarks:
                        cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 255), -1)

            fps_counter += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_counter / (time.time() - fps_t0)
                fps_counter = 0
                fps_t0 = time.time()
                print('frame %d, FPS %.1f, boxes %d, active_tracks %d, saved %d'
                      % (frame_count, fps, len(boxes), len(tracker.tracks), save_count), flush=True)
            cv2.putText(frame, 'FPS: %.1f' % fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if not args.headless:
                cv2.imshow('MobileFaceNet Live (q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        sess.close()
        print('shutdown complete', flush=True)


if __name__ == '__main__':
    main()
