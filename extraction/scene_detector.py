"""
scene_detector.py — Shot detection and scene grouping for anime episodes.

Standalone tunable module that separates two distinct problems:
  Step 1: Shot boundary detection (TransNetV2 + fade detection)
  Step 2: Per-shot feature extraction (color, motion, audio — lightweight OpenCV only)
           Optional GPU models: YOLOv8 segmentation, CLIP background embeddings
           Optional audio classifier: librosa-based music/ambient/silence detection
  Step 3: Scene boundary classification (weighted multi-signal scoring)

Can be imported by anime_metadata.py or run independently for tuning.

Usage:
    from scene_detector import detect_shots, extract_shot_features, detect_scenes

    shots = detect_shots(video_path)
    shots = extract_shot_features(video_path, shots, audio_path=audio_path)
    scenes = detect_scenes(shots)

    # With GPU models (optional):
    gpu_models = load_gpu_models(use_yolo=True, use_clip=True)
    shots = extract_shot_features(video_path, shots, gpu_models=gpu_models,
                                  keyframes_dir=keyframes_dir)

    # With audio classification (optional):
    from scene_detector import extract_audio_track
    audio_path = extract_audio_track(video_path, output_dir)
    shots = extract_shot_features(video_path, shots, audio_path=audio_path)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import time
import numpy as np
import cv2
import torch
import logging
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all tunable parameters in one place
# ---------------------------------------------------------------------------

CONFIG = {
    # TransNetV2
    "transnetv2_threshold": 0.5,        # prediction confidence for a shot boundary

    # Fade detection
    "fade_brightness_black": 15,         # avg pixel value below this = black frame
    "fade_brightness_white": 240,        # avg pixel value above this = white frame
    "fade_min_consecutive": 3,           # minimum consecutive dark/bright frames for a fade

    # Scene boundary classification
    "scene_break_threshold": 0.55,       # combined score above this = scene break
    "color_distance_threshold": 0.4,    # color histogram distance for strong break signal
    "audio_discontinuity_weight": 0.2,
    "dialogue_gap_threshold": 2.5,       # seconds of silence that contributes to break score
    "cast_discontinuity_weight": 0.15,

    # Signal weights for combined score (when no fade override)
    "weight_background": 0.40,
    "weight_audio": 0.20,
    "weight_dialogue_gap": 0.15,
    "weight_cast": 0.15,
    "weight_color": 0.10,

    # Shot type classification keywords (for VLM fallback — primary classification
    # happens in anime_metadata.py via VLM; these are heuristic fallbacks)
    "wide_shot_aspect_ratio": 0.4,       # if subject < 40% of frame height → wide
    "closeup_face_ratio": 0.3,           # if face > 30% of frame area → close-up

    # Optical flow (Farneback)
    "flow_pyr_scale": 0.5,
    "flow_levels": 3,
    "flow_winsize": 15,
    "flow_iterations": 3,
    "flow_poly_n": 5,
    "flow_poly_sigma": 1.2,
    "flow_offset_seconds": 0.5,          # compute flow between mid-frame and this offset

    # Color palette
    "color_histogram_bins": 16,          # bins per channel for color histogram

    # Flashback detection
    "saturation_drop_threshold": 0.6,    # if saturation < 60% of episode median → flashback candidate

    # GPU models (Phase 5 — all optional)
    "use_yolo": False,                   # YOLOv8-seg for character segmentation
    "use_clip": False,                   # CLIP for background embeddings
    "yolo_model_path": "yolov8n-seg.pt", # YOLOv8-seg model path
    "yolo_confidence": 0.3,              # YOLOv8 detection confidence threshold
    "yolo_person_class": 0,              # COCO class ID for "person"
    "clip_model_name": "ViT-B/32",       # CLIP model variant
    "color_hex_count": 5,                # number of dominant hex colors to extract

    # CLIP scene grouping (replaces color histogram as primary when available)
    "clip_distance_threshold": 0.4,      # cosine distance for strong scene break
    "weight_clip_background": 0.45,      # weight for CLIP background signal (replaces weight_background when CLIP active)

    # Multi-keyframe extraction (adaptive per shot duration)
    "keyframe_short_threshold": 3.0,     # shots < 3s: 1 keyframe (middle)
    "keyframe_medium_threshold": 6.0,    # shots 3-6s: 2 keyframes (1/4 + 3/4)
    "keyframe_long_interval": 2.5,       # shots > 6s: 1 keyframe per ~2.5s

    # Audio classifier (Phase 6 — optional, enabled via --use-audio)
    "audio_sample_rate": 22050,          # librosa default; sufficient for music/silence detection
    "audio_silence_threshold_db": -40.0, # RMS below this dB = silence
    "audio_music_harmonic_ratio": 0.6,   # harmonic energy / total energy above this = music candidate
    "audio_music_onset_rate_min": 1.0,   # onsets/sec above this supports music classification
    "audio_music_spectral_flatness_max": 0.15,  # spectral flatness below this = tonal (music-like)
    "weight_music_continuity": 0.03,     # music continuing across boundary = weak anti-break (anime BGM spans scenes)
    "weight_music_cut": 0.20,            # music stopping at boundary = pro-break signal
    "weight_silence_onset": 0.10,        # silence onset at boundary = moderate pro-break
}


# ---------------------------------------------------------------------------
# GPU model loading (optional — all models are lazy-loaded)
# ---------------------------------------------------------------------------

def load_gpu_models(use_yolo=False, use_clip=False, config=None):
    """Load optional GPU models for enhanced feature extraction.

    Returns a dict of loaded models (empty dict if none requested).
    Models that fail to load are logged and skipped — the pipeline
    continues without them.

    Args:
        use_yolo: Load YOLOv8-seg for character segmentation + background plates
        use_clip: Load CLIP for background embeddings (requires use_yolo=True
                  for character masking, but works standalone on full frames too)
        config: Optional config overrides

    Returns:
        {"yolo_model": ..., "clip_model": ..., "clip_preprocess": ...,
         "clip_device": ...}
        Missing keys mean the model was not loaded.
    """
    cfg = {**CONFIG, **(config or {})}
    models = {}

    if use_yolo:
        try:
            from ultralytics import YOLO
            model_path = cfg["yolo_model_path"]
            print(f"[GPU Models] Loading YOLOv8-seg from {model_path}...")
            yolo = YOLO(model_path)
            models["yolo_model"] = yolo
            print("[GPU Models] YOLOv8-seg loaded")
        except Exception as e:
            logger.warning(f"Could not load YOLOv8-seg: {e}")
            print(f"[GPU Models] YOLOv8-seg unavailable: {e}")

    if use_clip:
        try:
            import clip as clip_module
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = cfg["clip_model_name"]
            print(f"[GPU Models] Loading CLIP {model_name} on {device}...")
            clip_model, clip_preprocess = clip_module.load(model_name, device=device)
            models["clip_model"] = clip_model
            models["clip_preprocess"] = clip_preprocess
            models["clip_device"] = device
            print(f"[GPU Models] CLIP loaded on {device}")
        except Exception as e:
            logger.warning(f"Could not load CLIP: {e}")
            print(f"[GPU Models] CLIP unavailable: {e}")

    return models


def _run_yolo_segmentation(keyframe, yolo_model, cfg):
    """Run YOLOv8-seg on a keyframe to detect characters.

    Returns:
        (character_boxes, background_plate)
        character_boxes: list of {"bbox": [x1,y1,x2,y2], "confidence": float}
        background_plate: numpy array with characters masked out (inpainted), or None
    """
    results = yolo_model(keyframe, verbose=False, conf=cfg.get("yolo_confidence", 0.3))
    person_class = cfg.get("yolo_person_class", 0)

    character_boxes = []
    mask_combined = np.zeros(keyframe.shape[:2], dtype=np.uint8)

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes
        for i, cls_id in enumerate(boxes.cls):
            if int(cls_id) != person_class:
                continue

            box = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu())
            character_boxes.append({
                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "confidence": round(conf, 3),
            })

            # Use segmentation mask if available, else use bounding box
            if result.masks is not None and i < len(result.masks.data):
                seg_mask = result.masks.data[i].cpu().numpy()
                # Resize mask to frame size
                seg_resized = cv2.resize(seg_mask, (keyframe.shape[1], keyframe.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                mask_combined = np.maximum(mask_combined, (seg_resized > 0.5).astype(np.uint8))
            else:
                # Fallback: use bounding box as mask
                mask_combined[box[1]:box[3], box[0]:box[2]] = 1

    # Create background plate by inpainting masked regions
    background_plate = None
    if np.any(mask_combined):
        # Dilate mask slightly to cover edges
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask_combined * 255, kernel, iterations=2)
        background_plate = cv2.inpaint(keyframe, mask_dilated, inpaintRadius=5,
                                        flags=cv2.INPAINT_TELEA)
    else:
        # No characters detected — full frame is the background
        background_plate = keyframe.copy()

    return character_boxes, background_plate


def _compute_clip_embedding(image, clip_model, clip_preprocess, device):
    """Compute CLIP embedding for an image.

    Args:
        image: numpy array (BGR from OpenCV)
        clip_model: loaded CLIP model
        clip_preprocess: CLIP preprocessing transform
        device: torch device string

    Returns:
        List of 512 floats (CLIP embedding), or empty list on failure.
    """
    try:
        from PIL import Image as PILImage
        # Convert BGR to RGB PIL image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)

        image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalize
            embedding = embedding.squeeze().cpu().numpy()

        return [round(float(v), 6) for v in embedding]
    except Exception as e:
        logger.warning(f"CLIP embedding failed: {e}")
        return []


def _extract_dominant_hex_colors(frame_bgr, n_colors=5):
    """Extract dominant colors from a frame using k-means clustering.

    Args:
        frame_bgr: numpy array (BGR from OpenCV)
        n_colors: number of dominant colors to extract

    Returns:
        List of hex color strings, e.g. ["#FF5733", "#2E86AB", "#A23B72"]
    """
    # Downsample for speed
    small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    # k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 3,
                                     cv2.KMEANS_PP_CENTERS)

    # Sort by frequency (most common first)
    label_counts = Counter(labels.flatten())
    sorted_indices = [idx for idx, _ in label_counts.most_common()]

    hex_colors = []
    for idx in sorted_indices:
        b, g, r = [int(c) for c in centers[idx]]
        hex_colors.append(f"#{r:02X}{g:02X}{b:02X}")

    return hex_colors


# ---------------------------------------------------------------------------
# Step 1 — Shot boundary detection
# ---------------------------------------------------------------------------

def detect_shots(video_path, config=None):
    """
    Run TransNetV2 on the full episode to get every shot boundary.

    Returns a list of shot dicts:
        [{"shot_id": int, "start_frame": int, "end_frame": int,
          "start_time": float, "end_time": float, "duration": float,
          "fade_type": "none"|"fade_to_black"|"fade_to_white"|"dissolve"}, ...]
    """
    cfg = {**CONFIG, **(config or {})}
    print("[Shot Detection] Loading TransNetV2...")
    from transnetv2_pytorch import TransNetV2

    # Force CPU — TransNetV2 is lightweight and avoids CUDA deterministic algo issues
    model = TransNetV2(device="cpu")
    model.eval()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[Shot Detection] Video: {total_frames} frames at {fps:.2f} fps "
          f"({total_frames / fps:.1f}s)")
    print("[Shot Detection] Running TransNetV2 prediction...")

    # TransNetV2 expects video path directly
    video_frames, single_frame_preds, all_frame_preds = model.predict_video(video_path)

    # Convert predictions to numpy (predict_video returns torch tensors)
    if hasattr(single_frame_preds, 'cpu'):
        single_frame_preds = single_frame_preds.cpu().numpy()

    # Get scene boundaries from predictions
    scene_list = model.predictions_to_scenes(
        single_frame_preds,
        threshold=cfg["transnetv2_threshold"]
    )

    print(f"[Shot Detection] Found {len(scene_list)} shots")

    # Detect fades at each boundary
    fade_map = _detect_fades(video_path, scene_list, fps, total_frames, cfg)

    # Build shot list
    shots = []
    for i, (start_frame, end_frame) in enumerate(scene_list):
        start_time = start_frame / fps
        end_time = end_frame / fps
        fade_type = fade_map.get(i, "none")

        shots.append({
            "shot_id": i + 1,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "duration": round(end_time - start_time, 3),
            "fade_type": fade_type,
        })

    print(f"[Shot Detection] Fades detected: "
          f"{sum(1 for s in shots if s['fade_type'] != 'none')} "
          f"({sum(1 for s in shots if s['fade_type'] == 'fade_to_black')} black, "
          f"{sum(1 for s in shots if s['fade_type'] == 'fade_to_white')} white, "
          f"{sum(1 for s in shots if s['fade_type'] == 'dissolve')} dissolve)")

    # Clean up
    del model, video_frames, single_frame_preds, all_frame_preds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return shots


def _detect_fades(video_path, scene_list, fps, total_frames, cfg):
    """
    Detect fade-to-black / fade-to-white at shot boundaries by checking
    average frame brightness over consecutive frames spanning each boundary.

    Returns dict: {shot_index: "fade_to_black"|"fade_to_white"|"dissolve"}
    Only includes shots where a fade was detected.
    """
    fade_map = {}
    min_consecutive = cfg["fade_min_consecutive"]
    black_thresh = cfg["fade_brightness_black"]
    white_thresh = cfg["fade_brightness_white"]

    cap = cv2.VideoCapture(video_path)

    for i in range(len(scene_list) - 1):
        boundary_frame = scene_list[i][1]  # end frame of current shot

        # Check frames around the boundary
        check_start = max(0, boundary_frame - min_consecutive)
        check_end = min(total_frames - 1, boundary_frame + min_consecutive)

        brightnesses = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, check_start)
        for f in range(check_start, check_end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightnesses.append(float(np.mean(gray)))

        if len(brightnesses) < min_consecutive:
            continue

        # Check for consecutive dark frames
        dark_run = _max_consecutive(brightnesses, lambda b: b < black_thresh)
        if dark_run >= min_consecutive:
            fade_map[i + 1] = "fade_to_black"  # fade affects the NEXT shot's start
            continue

        # Check for consecutive bright frames
        bright_run = _max_consecutive(brightnesses, lambda b: b > white_thresh)
        if bright_run >= min_consecutive:
            fade_map[i + 1] = "fade_to_white"
            continue

        # Check for dissolve: brightness variance is high across boundary
        # (frames blend between two different images)
        if len(brightnesses) >= 4:
            first_half = brightnesses[:len(brightnesses) // 2]
            second_half = brightnesses[len(brightnesses) // 2:]
            variance = abs(np.mean(first_half) - np.mean(second_half))
            if variance > 30 and dark_run < 2 and bright_run < 2:
                fade_map[i + 1] = "dissolve"

    cap.release()
    return fade_map


def _max_consecutive(values, predicate):
    """Return the length of the longest consecutive run where predicate is true."""
    max_run = 0
    current_run = 0
    for v in values:
        if predicate(v):
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


# ---------------------------------------------------------------------------
# Step 2 — Per-shot feature extraction
# ---------------------------------------------------------------------------

def _compute_keyframe_positions(shot, fps, cfg):
    """Compute adaptive keyframe positions for a shot based on its duration.

    Returns a list of (frame_index, is_primary) tuples.
    Exactly one entry will have is_primary=True (the one closest to middle).

    - duration < short_threshold (3s): 1 keyframe at middle
    - short <= duration < medium_threshold (6s): 2 keyframes at 1/4 and 3/4
    - duration >= medium_threshold: sample every ~long_interval (2.5s)
    """
    start = shot["start_frame"]
    end = shot["end_frame"]
    duration = shot.get("duration", (end - start) / max(fps, 1))

    short_thresh = cfg.get("keyframe_short_threshold", 3.0)
    medium_thresh = cfg.get("keyframe_medium_threshold", 6.0)
    long_interval = cfg.get("keyframe_long_interval", 2.5)

    mid = (start + end) // 2

    if duration < short_thresh:
        return [(mid, True)]

    if duration < medium_thresh:
        quarter = start + (end - start) // 4
        three_quarter = start + 3 * (end - start) // 4
        # Primary is the one closest to the middle
        if abs(quarter - mid) <= abs(three_quarter - mid):
            return [(quarter, True), (three_quarter, False)]
        return [(quarter, False), (three_quarter, True)]

    # Long shot: sample every ~long_interval seconds
    num_frames = max(2, round(duration / long_interval))
    positions = []
    for i in range(num_frames):
        frac = (i + 0.5) / num_frames
        frame_idx = start + int(frac * (end - start))
        frame_idx = min(frame_idx, end)
        positions.append(frame_idx)

    # Mark the one closest to middle as primary
    closest_idx = min(range(len(positions)), key=lambda i: abs(positions[i] - mid))
    return [(f, i == closest_idx) for i, f in enumerate(positions)]


def extract_shot_features(video_path, shots, audio_path=None, subtitle_segments=None,
                          config=None, keyframes_dir=None, gpu_models=None):
    """
    For each shot, extract lightweight features using OpenCV, plus optional GPU models:
      - color_palette (HSV histogram — used for scene grouping)
      - color_hex (dominant hex colors — for generation pipeline color grading)
      - motion_intensity (optical flow between mid-frame and offset frame)
      - saturation (for flashback detection)
      - brightness (for fade/eyecatch detection)
      - audio_features (dialogue presence from subtitle segments)
      - keyframe_paths (saved keyframe images, adaptive count based on shot duration)

    Optional GPU model features (when gpu_models provided):
      - character_crops (YOLOv8-seg bounding boxes per keyframe, with bbox_area_fraction)
      - background_plate_path (characters masked out from primary keyframe, saved as JPEG)
      - background_embedding (512-dim CLIP vector on primary keyframe's background plate)

    Shot type and character detection are deferred to VLM analysis stage.
    Modifies shots in-place and returns them.

    Args:
        keyframes_dir: Optional directory to save per-shot keyframe JPEGs.
            Adaptive extraction: shots < 3s get 1 keyframe (middle), 3-6s
            get 2 (quarter + three-quarter), >6s get 1 per ~2.5s.
            Sets ``keyframe_paths`` (list) on the shot dict.
        gpu_models: Optional dict from load_gpu_models() with loaded
            YOLOv8 and/or CLIP models.
    """
    cfg = {**CONFIG, **(config or {})}
    gpu_models = gpu_models or {}

    has_yolo = "yolo_model" in gpu_models
    has_clip = "clip_model" in gpu_models

    model_tags = []
    if has_yolo:
        model_tags.append("YOLOv8")
    if has_clip:
        model_tags.append("CLIP")

    # Load audio file for classification if audio_path is provided
    audio_loaded = None
    audio_sr = None
    if audio_path and os.path.exists(audio_path):
        try:
            import librosa
        except ImportError as e:
            raise ImportError(
                f"--use-audio requires librosa but it failed to import: {e}\n"
                "Install with: pip install librosa soundfile"
            ) from e
        try:
            audio_sr = cfg.get("audio_sample_rate", 22050)
            print(f"[Feature Extraction] Loading audio track for classification...")
            audio_loaded, audio_sr = librosa.load(audio_path, sr=audio_sr, mono=True)
            model_tags.append("Audio")
            print(f"[Feature Extraction] Audio loaded: {len(audio_loaded)/audio_sr:.1f}s at {audio_sr}Hz")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {audio_path}: {e}") from e

    model_str = f" + {', '.join(model_tags)}" if model_tags else ""

    print(f"[Feature Extraction] Extracting features (OpenCV{model_str})...")

    if keyframes_dir:
        os.makedirs(keyframes_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    episode_saturations = []

    total = len(shots)
    for idx, shot in enumerate(shots):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"[Feature Extraction] Processing shot {idx + 1}/{total}...")

        # --- Adaptive multi-keyframe extraction ---
        kf_positions = _compute_keyframe_positions(shot, fps, cfg)
        keyframes_data = []  # list of (frame, is_primary, path)
        for kf_frame_idx, is_primary in kf_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, kf_frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            keyframes_data.append((frame, is_primary, kf_frame_idx))

        if not keyframes_data:
            mid_frame_idx = (shot["start_frame"] + shot["end_frame"]) // 2
            logger.warning(f"Could not read any frames for shot {shot['shot_id']} (tried {len(kf_positions)} positions near frame {mid_frame_idx})")
            _set_default_features(shot)
            continue

        # Identify primary keyframe (for color, motion, saturation features)
        primary_frame = None
        primary_frame_idx = None
        for frame, is_primary, frame_idx in keyframes_data:
            if is_primary:
                primary_frame = frame
                primary_frame_idx = frame_idx
                break
        if primary_frame is None:
            primary_frame = keyframes_data[0][0]
            primary_frame_idx = keyframes_data[0][2]

        # --- Save keyframe images to disk ---
        saved_paths = []
        if keyframes_dir:
            if len(keyframes_data) == 1:
                # Single keyframe: keep backward-compatible naming
                kf_path = os.path.join(
                    keyframes_dir, f"shot_{shot['shot_id']:04d}_keyframe.jpg"
                )
                cv2.imwrite(kf_path, keyframes_data[0][0])
                saved_paths.append(kf_path)
            else:
                for ki, (frame, _, _) in enumerate(keyframes_data):
                    kf_path = os.path.join(
                        keyframes_dir, f"shot_{shot['shot_id']:04d}_kf_{ki:02d}.jpg"
                    )
                    cv2.imwrite(kf_path, frame)
                    saved_paths.append(kf_path)
        shot["keyframe_paths"] = saved_paths

        # --- Color palette (used for scene boundary detection) ---
        shot["color_palette"] = _compute_color_histogram(primary_frame, cfg)

        # --- Dominant hex colors (for generation pipeline color grading) ---
        shot["color_hex"] = _extract_dominant_hex_colors(primary_frame, cfg.get("color_hex_count", 5))

        # --- Motion intensity (optical flow) ---
        mid_frame_idx = primary_frame_idx
        offset_frame_idx = mid_frame_idx + int(cfg["flow_offset_seconds"] * fps)
        offset_frame_idx = min(offset_frame_idx, shot["end_frame"])
        if offset_frame_idx != mid_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, offset_frame_idx)
            ret2, frame2 = cap.read()
            if ret2:
                shot["motion_intensity"] = _compute_motion_intensity(primary_frame, frame2, cfg)
            else:
                shot["motion_intensity"] = 0.0
        else:
            shot["motion_intensity"] = 0.0

        # --- Saturation and brightness (from primary keyframe) ---
        hsv = cv2.cvtColor(primary_frame, cv2.COLOR_BGR2HSV)
        sat = float(np.mean(hsv[:, :, 1]))
        brightness = float(np.mean(hsv[:, :, 2]))
        shot["_saturation"] = sat
        shot["_brightness"] = brightness
        episode_saturations.append(sat)

        # --- Audio features from subtitle segments ---
        if subtitle_segments is not None:
            shot["audio_features"] = _extract_audio_features_for_shot(
                shot, subtitle_segments
            )
        else:
            shot["audio_features"] = {
                "has_dialogue": False,
                "dialogue_density": 0.0,
                "silence_ratio": 1.0,
            }

        # --- Audio classification from actual audio track (optional) ---
        if audio_loaded is not None:
            try:
                shot["audio_classification"] = _classify_audio_for_shot(
                    audio_loaded, audio_sr, shot, cfg
                )
            except Exception as e:
                logger.warning(f"Audio classification failed on shot {shot['shot_id']}: {e}")
                shot["audio_classification"] = _default_audio_classification()
        else:
            shot["audio_classification"] = _default_audio_classification()

        # --- GPU model features (optional) ---

        # YOLOv8 character segmentation + background plates
        # Run on ALL keyframes for richer character crops; background plate from primary only
        shot["character_crops"] = []
        shot["background_plate_path"] = None
        if has_yolo:
            try:
                all_char_crops = []
                for ki, (frame, is_primary, _) in enumerate(keyframes_data):
                    char_boxes, bg_plate = _run_yolo_segmentation(
                        frame, gpu_models["yolo_model"], cfg
                    )

                    # Compute bbox_area_fraction for prominence filtering
                    frame_area = frame.shape[0] * frame.shape[1]
                    for cbox in char_boxes:
                        bx1, by1, bx2, by2 = cbox["bbox"]
                        bbox_area = (bx2 - bx1) * (by2 - by1)
                        cbox["bbox_area_fraction"] = round(bbox_area / max(frame_area, 1), 4)
                        cbox["keyframe_index"] = ki

                    # Save background plate from primary keyframe only
                    if is_primary and bg_plate is not None and keyframes_dir:
                        bg_path = os.path.join(
                            keyframes_dir, f"shot_{shot['shot_id']:04d}_bg_plate.jpg"
                        )
                        cv2.imwrite(bg_path, bg_plate)
                        shot["background_plate_path"] = bg_path
                        shot["_bg_plate_frame"] = bg_plate  # temp: for CLIP embedding below

                    # Save character crop images
                    if char_boxes and keyframes_dir:
                        crops_dir = os.path.join(keyframes_dir, "character_crops")
                        os.makedirs(crops_dir, exist_ok=True)
                        for ci, cbox in enumerate(char_boxes):
                            bx1, by1, bx2, by2 = cbox["bbox"]
                            char_crop = frame[by1:by2, bx1:bx2]
                            if char_crop.size > 0:
                                if len(keyframes_data) == 1:
                                    crop_fname = f"shot_{shot['shot_id']:04d}_char_{ci:02d}.jpg"
                                else:
                                    crop_fname = f"shot_{shot['shot_id']:04d}_kf_{ki:02d}_char_{ci:02d}.jpg"
                                crop_path = os.path.join(crops_dir, crop_fname)
                                cv2.imwrite(crop_path, char_crop)
                                cbox["crop_image_path"] = crop_path

                    all_char_crops.extend(char_boxes)
                shot["character_crops"] = all_char_crops
            except Exception as e:
                logger.warning(f"YOLOv8 failed on shot {shot['shot_id']}: {e}")

        # CLIP background embedding (primary keyframe only)
        shot["background_embedding"] = []
        if has_clip:
            try:
                # Prefer background plate (characters masked); fall back to primary frame
                embed_frame = shot.pop("_bg_plate_frame", None)
                if embed_frame is None:
                    embed_frame = primary_frame

                shot["background_embedding"] = _compute_clip_embedding(
                    embed_frame,
                    gpu_models["clip_model"],
                    gpu_models["clip_preprocess"],
                    gpu_models["clip_device"],
                )
            except Exception as e:
                logger.warning(f"CLIP failed on shot {shot['shot_id']}: {e}")
        else:
            # Clean up temp frame if CLIP not used
            shot.pop("_bg_plate_frame", None)

        # Placeholders — VLM will fill these later
        shot["shot_type"] = "medium"

    cap.release()

    # Post-process: flag flashback candidates based on episode median saturation
    if episode_saturations:
        median_sat = float(np.median(episode_saturations))
        sat_threshold = median_sat * cfg["saturation_drop_threshold"]
        for shot in shots:
            shot["_flashback_candidate"] = shot.get("_saturation", 255) < sat_threshold
    else:
        for shot in shots:
            shot["_flashback_candidate"] = False

    print(f"[Feature Extraction] Completed features for {total} shots")
    return shots


def _set_default_features(shot):
    """Set default feature values for a shot whose keyframe could not be read."""
    shot["shot_type"] = "medium"
    shot["motion_intensity"] = 0.0
    shot["color_palette"] = []
    shot["color_hex"] = []
    shot["background_embedding"] = []
    shot["character_crops"] = []
    shot["background_plate_path"] = None
    shot["keyframe_paths"] = []
    shot["audio_features"] = {
        "has_dialogue": False,
        "dialogue_density": 0.0,
        "silence_ratio": 1.0,
    }
    shot["audio_classification"] = _default_audio_classification()
    shot["_saturation"] = 128.0
    shot["_brightness"] = 128.0
    shot["_flashback_candidate"] = False


def _compute_motion_intensity(frame1, frame2, cfg):
    """
    Compute optical flow magnitude between two frames using Farneback method.
    Returns a normalized float 0.0–1.0.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=cfg["flow_pyr_scale"],
        levels=cfg["flow_levels"],
        winsize=cfg["flow_winsize"],
        iterations=cfg["flow_iterations"],
        poly_n=cfg["flow_poly_n"],
        poly_sigma=cfg["flow_poly_sigma"],
        flags=0
    )

    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_mag = float(np.mean(magnitude))

    # Normalize: typical anime has mean flow magnitude 0–20 pixels
    # Clamp to 0–1 with 20 as the max reference
    return round(min(mean_mag / 20.0, 1.0), 3)


def _compute_color_histogram(frame_bgr, cfg):
    """
    Compute a normalized color histogram of the frame.
    Returns a flat list of floats (histogram values).
    """
    bins = cfg["color_histogram_bins"]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    # Normalize each channel
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)

    combined = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    return [round(float(v), 4) for v in combined]


def _extract_audio_features_for_shot(shot, subtitle_segments):
    """
    Determine audio characteristics for a shot based on overlapping subtitle segments.
    """
    shot_start = shot["start_time"]
    shot_end = shot["end_time"]
    shot_duration = shot_end - shot_start

    if shot_duration <= 0:
        return {"has_dialogue": False, "dialogue_density": 0.0, "silence_ratio": 1.0}

    dialogue_time = 0.0
    for seg in subtitle_segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)

        # Compute overlap
        overlap_start = max(shot_start, seg_start)
        overlap_end = min(shot_end, seg_end)
        if overlap_end > overlap_start:
            dialogue_time += (overlap_end - overlap_start)

    has_dialogue = dialogue_time > 0.1
    dialogue_density = round(dialogue_time / shot_duration, 3)
    silence_ratio = round(1.0 - dialogue_density, 3)

    return {
        "has_dialogue": has_dialogue,
        "dialogue_density": dialogue_density,
        "silence_ratio": silence_ratio,
    }


# ---------------------------------------------------------------------------
# Audio classifier (Phase 6 — librosa-based spectral analysis)
# ---------------------------------------------------------------------------

def extract_audio_track(video_path, output_dir, config=None):
    """Extract audio track from video as mono WAV using ffmpeg.

    Cached: if the WAV already exists and resume is enabled, returns the path
    without re-extracting.

    Args:
        video_path: Path to episode video file
        output_dir: Episode output directory
        config: Optional config overrides

    Returns:
        Path to extracted WAV file, or None on failure.
    """
    import subprocess

    audio_path = os.path.join(output_dir, "audio_track.wav")

    cfg = {**CONFIG, **(config or {})}
    sr = cfg.get("audio_sample_rate", 22050)

    if os.path.exists(audio_path):
        return audio_path

    print(f"[Audio] Extracting audio track from video...")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                    # no video
            "-ac", "1",               # mono
            "-ar", str(sr),           # target sample rate
            "-acodec", "pcm_s16le",   # 16-bit PCM WAV
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning(f"ffmpeg audio extraction failed: {result.stderr[:500]}")
            return None
        print(f"[Audio] Audio track saved: {audio_path}")
        return audio_path
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return None


def _load_audio_segment(audio_data, sr, start_time, end_time):
    """Extract a time segment from loaded audio data.

    Args:
        audio_data: numpy array of audio samples
        sr: sample rate
        start_time: segment start in seconds
        end_time: segment end in seconds

    Returns:
        numpy array of the audio segment
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    start_sample = max(0, min(start_sample, len(audio_data)))
    end_sample = max(start_sample, min(end_sample, len(audio_data)))
    return audio_data[start_sample:end_sample]


def _classify_audio_for_shot(audio_data, sr, shot, cfg):
    """Classify audio characteristics for a shot using spectral analysis.

    Uses librosa to compute:
    - RMS energy (silence detection)
    - Harmonic-percussive decomposition (music detection)
    - Onset detection rate (rhythmic regularity)
    - Spectral flatness (tonal vs noise)

    Args:
        audio_data: Full episode audio as numpy array
        sr: Sample rate
        shot: Shot dict with start_time, end_time
        cfg: Config dict

    Returns:
        Dict with audio classification results:
        {
            "has_music": bool,
            "music_confidence": float (0.0-1.0),
            "is_silence": bool,
            "rms_db": float,
            "spectral_flatness": float,
            "onset_rate": float,
            "audio_type": "music" | "dialogue" | "ambient" | "silence"
        }
    """
    import librosa

    segment = _load_audio_segment(audio_data, sr, shot["start_time"], shot["end_time"])

    # Default result for very short or empty segments
    if len(segment) < sr * 0.1:  # less than 100ms
        return _default_audio_classification()

    # --- RMS energy (silence detection) ---
    rms = librosa.feature.rms(y=segment, frame_length=2048, hop_length=512)[0]
    mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.0
    rms_db = float(librosa.amplitude_to_db(np.array([mean_rms]), ref=1.0)[0]) if mean_rms > 0 else -80.0
    is_silence = rms_db < cfg.get("audio_silence_threshold_db", -40.0)

    if is_silence:
        return {
            "has_music": False,
            "music_confidence": 0.0,
            "is_silence": True,
            "rms_db": round(rms_db, 1),
            "spectral_flatness": 0.0,
            "onset_rate": 0.0,
            "audio_type": "silence",
        }

    # --- Harmonic-percussive source separation ---
    harmonic, percussive = librosa.effects.hpss(segment)
    harmonic_energy = float(np.sum(harmonic ** 2))
    total_energy = float(np.sum(segment ** 2))
    harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.0

    # --- Spectral flatness (tonal vs noise) ---
    flatness = librosa.feature.spectral_flatness(y=segment, hop_length=512)[0]
    mean_flatness = float(np.mean(flatness)) if len(flatness) > 0 else 0.5

    # --- Onset rate (rhythmic regularity) ---
    onsets = librosa.onset.onset_detect(y=segment, sr=sr, hop_length=512)
    shot_duration = shot["end_time"] - shot["start_time"]
    onset_rate = len(onsets) / shot_duration if shot_duration > 0 else 0.0

    # --- Music classification heuristic ---
    # Music = high harmonic ratio + low spectral flatness + some rhythmic activity
    music_signals = 0.0
    hr_threshold = cfg.get("audio_music_harmonic_ratio", 0.6)
    flat_threshold = cfg.get("audio_music_spectral_flatness_max", 0.15)
    onset_threshold = cfg.get("audio_music_onset_rate_min", 1.0)

    if harmonic_ratio > hr_threshold:
        music_signals += 0.4
    elif harmonic_ratio > hr_threshold * 0.75:
        music_signals += 0.2

    if mean_flatness < flat_threshold:
        music_signals += 0.35
    elif mean_flatness < flat_threshold * 2:
        music_signals += 0.15

    if onset_rate > onset_threshold:
        music_signals += 0.25
    elif onset_rate > onset_threshold * 0.5:
        music_signals += 0.1

    music_confidence = min(music_signals, 1.0)
    has_music = music_confidence >= 0.5

    # --- Classify audio type ---
    if has_music:
        audio_type = "music"
    elif mean_flatness > 0.3:
        audio_type = "ambient"
    else:
        audio_type = "dialogue"

    return {
        "has_music": has_music,
        "music_confidence": round(music_confidence, 3),
        "is_silence": False,
        "rms_db": round(rms_db, 1),
        "spectral_flatness": round(mean_flatness, 4),
        "onset_rate": round(onset_rate, 2),
        "audio_type": audio_type,
    }


def _default_audio_classification():
    """Default audio classification for shots that can't be analyzed."""
    return {
        "has_music": False,
        "music_confidence": 0.0,
        "is_silence": False,
        "rms_db": -80.0,
        "spectral_flatness": 0.0,
        "onset_rate": 0.0,
        "audio_type": "unknown",
    }


# ---------------------------------------------------------------------------
# Step 3 — Scene boundary classification
# ---------------------------------------------------------------------------

def detect_scenes(shots, config=None):
    """
    Classify scene boundaries from the shot list.

    For each consecutive shot pair, compute a scene-break score using:
      1. Fade override (automatic break)
      2. Background embedding cosine distance
      3. Audio discontinuity
      4. Dialogue gap
      5. Cast discontinuity
      6. Color palette distance

    Returns list of scene dicts:
        [{"scene_number": 1, "start_shot_id": 1, "end_shot_id": 8,
          "start_time": 0.0, "end_time": 45.2, "duration": 45.2,
          "shot_ids": [1,2,3,4,5,6,7,8],
          "has_flashback_candidate": False}, ...]
    """
    cfg = {**CONFIG, **(config or {})}

    if not shots:
        return []

    # Compute scene break scores between consecutive shots
    boundaries = []  # indices where scene breaks occur (these are shot indices)

    for i in range(len(shots) - 1):
        shot_a = shots[i]
        shot_b = shots[i + 1]

        score, reason = _compute_scene_break_score(shot_a, shot_b, cfg)

        if score >= cfg["scene_break_threshold"]:
            boundaries.append(i + 1)  # break BEFORE shot at index i+1

    # Group shots into scenes
    scenes = []
    scene_start_idx = 0

    for b_idx, boundary in enumerate(boundaries):
        scene_shots = shots[scene_start_idx:boundary]
        if scene_shots:
            scenes.append(_build_scene(len(scenes) + 1, scene_shots))
        scene_start_idx = boundary

    # Last scene
    remaining = shots[scene_start_idx:]
    if remaining:
        scenes.append(_build_scene(len(scenes) + 1, remaining))

    print(f"[Scene Detection] Grouped {len(shots)} shots into {len(scenes)} scenes")

    # Log scene statistics
    durations = [s["duration"] for s in scenes]
    if durations:
        print(f"[Scene Detection] Scene durations: "
              f"min={min(durations):.1f}s, max={max(durations):.1f}s, "
              f"median={sorted(durations)[len(durations)//2]:.1f}s, "
              f"mean={sum(durations)/len(durations):.1f}s")

    return scenes


def _compute_scene_break_score(shot_a, shot_b, cfg):
    """
    Compute a weighted scene-break score between two consecutive shots.
    Returns (score: float, reason: str).
    """

    # Priority 1: Fade override — automatic scene break
    if shot_b.get("fade_type") in ("fade_to_black", "fade_to_white"):
        return 1.0, f"fade: {shot_b['fade_type']}"

    score = 0.0
    reasons = []

    # Signal 1: Background similarity — CLIP embedding if available, else color histogram
    embed_a = shot_a.get("background_embedding", [])
    embed_b = shot_b.get("background_embedding", [])
    used_clip = False

    if embed_a and embed_b and len(embed_a) == len(embed_b) and len(embed_a) > 1:
        # CLIP cosine distance (embeddings are already L2-normalized)
        vec_a = np.array(embed_a, dtype=np.float32)
        vec_b = np.array(embed_b, dtype=np.float32)
        cosine_sim = float(np.dot(vec_a, vec_b))
        clip_dist = 1.0 - max(cosine_sim, 0)

        bg_weight = cfg.get("weight_clip_background", cfg["weight_background"])
        if clip_dist > cfg.get("clip_distance_threshold", 0.4):
            bg_score = min(clip_dist / 0.6, 1.0)
            score += bg_weight * bg_score
            reasons.append(f"clip_dist={clip_dist:.3f}")
        else:
            continuity_bonus = (cfg.get("clip_distance_threshold", 0.4) - clip_dist) * 0.2
            score -= continuity_bonus
        used_clip = True

    # Fall back to color histogram if no CLIP embeddings
    if not used_clip:
        palette_a = shot_a.get("color_palette", [])
        palette_b = shot_b.get("color_palette", [])
        if palette_a and palette_b:
            hist_a = np.array(palette_a, dtype=np.float32)
            hist_b = np.array(palette_b, dtype=np.float32)
            if hist_a.shape == hist_b.shape:
                color_corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
                color_dist = 1.0 - max(color_corr, 0)
                if color_dist > cfg["color_distance_threshold"]:
                    bg_score = min(color_dist / 0.6, 1.0)
                    score += cfg["weight_background"] * bg_score
                    reasons.append(f"color_dist={color_dist:.3f}")
                else:
                    continuity_bonus = (cfg["color_distance_threshold"] - color_dist) * 0.2
                    score -= continuity_bonus

    # Signal 2: Audio discontinuity (subtitle-based dialogue change)
    audio_a = shot_a.get("audio_features", {})
    audio_b = shot_b.get("audio_features", {})
    # Dialogue to silence transition or vice versa
    if audio_a.get("has_dialogue") != audio_b.get("has_dialogue"):
        audio_signal = 0.5
    else:
        audio_signal = 0.0
    score += cfg["weight_audio"] * audio_signal
    if audio_signal > 0:
        reasons.append("audio_discontinuity")

    # Signal 2b: Audio classifier signals (music continuity/cut, silence onset)
    ac_a = shot_a.get("audio_classification", {})
    ac_b = shot_b.get("audio_classification", {})
    if ac_a.get("audio_type") != "unknown" and ac_b.get("audio_type") != "unknown":
        music_a = ac_a.get("has_music", False)
        music_b = ac_b.get("has_music", False)
        silence_b = ac_b.get("is_silence", False)

        if music_a and music_b:
            # Music continuing across boundary = anti-break signal
            score -= cfg.get("weight_music_continuity", 0.15)
            reasons.append("music_continuity(-)")
        elif music_a and not music_b:
            # Music stops at boundary = pro-break signal
            score += cfg.get("weight_music_cut", 0.20)
            reasons.append("music_cut(+)")
        elif not music_a and music_b:
            # Music starts at boundary = moderate pro-break signal
            score += cfg.get("weight_music_cut", 0.20) * 0.5
            reasons.append("music_start(+)")

        if not ac_a.get("is_silence", False) and silence_b:
            # Silence onset = moderate pro-break signal
            score += cfg.get("weight_silence_onset", 0.10)
            reasons.append("silence_onset(+)")

    # Signal 3: Dialogue gap
    gap = shot_b["start_time"] - shot_a["end_time"]
    if gap > cfg["dialogue_gap_threshold"]:
        # Silence gap between shots
        gap_signal = min(gap / 5.0, 1.0)
        score += cfg["weight_dialogue_gap"] * gap_signal
        reasons.append(f"dialogue_gap={gap:.1f}s")
    else:
        # Check for silence within the shots themselves
        silence_a = audio_a.get("silence_ratio", 0)
        silence_b = audio_b.get("silence_ratio", 0)
        if silence_a > 0.8 and silence_b > 0.8:
            score += cfg["weight_dialogue_gap"] * 0.3
            reasons.append("mutual_silence")

    # Signal 4: Cast discontinuity
    cast_a = set(shot_a.get("characters_present", []))
    cast_b = set(shot_b.get("characters_present", []))
    if cast_a and cast_b and not cast_a.intersection(cast_b):
        score += cfg["weight_cast"] * 1.0
        reasons.append("cast_discontinuity")

    # Dissolve at boundary is a soft scene break signal
    if shot_b.get("fade_type") == "dissolve":
        score += 0.15
        reasons.append("dissolve")

    score = max(0.0, min(1.0, score))
    return score, "; ".join(reasons) if reasons else "no_signal"


def _build_scene(scene_number, scene_shots):
    """Build a scene dict from a group of consecutive shots."""
    shot_ids = [s["shot_id"] for s in scene_shots]
    start_time = scene_shots[0]["start_time"]
    end_time = scene_shots[-1]["end_time"]

    has_flashback = any(s.get("_flashback_candidate", False) for s in scene_shots)

    return {
        "scene_number": scene_number,
        "start_shot_id": shot_ids[0],
        "end_shot_id": shot_ids[-1],
        "start_time": start_time,
        "end_time": end_time,
        "duration": round(end_time - start_time, 3),
        "shot_count": len(scene_shots),
        "shot_ids": shot_ids,
        "has_flashback_candidate": has_flashback,
    }


# ---------------------------------------------------------------------------
# Eyecatch detection
# ---------------------------------------------------------------------------

def detect_eyecatch(shots, config=None):
    """
    Detect mid-episode eyecatch (title card break) in One Piece episodes.
    Eyecatch is typically: very short shot (< 5s), high brightness, preceded and
    followed by fade-to-black. Located roughly in the middle of the episode.

    Returns list of shot_ids that are eyecatch frames, or empty list.
    """
    cfg = {**CONFIG, **(config or {})}

    if len(shots) < 10:
        return []

    total_duration = shots[-1]["end_time"] - shots[0]["start_time"]
    mid_point = total_duration / 2.0
    mid_window = total_duration * 0.15  # search within ±15% of midpoint

    eyecatch_ids = []

    for i, shot in enumerate(shots):
        shot_mid = (shot["start_time"] + shot["end_time"]) / 2.0

        # Must be near episode midpoint
        if abs(shot_mid - mid_point) > mid_window:
            continue

        # Must be short
        if shot["duration"] > 5.0:
            continue

        # Check for surrounding fades
        has_fade_before = (i > 0 and shots[i - 1].get("fade_type") in
                          ("fade_to_black", "fade_to_white"))
        has_fade_after = (shot.get("fade_type") in ("fade_to_black", "fade_to_white")
                         or (i + 1 < len(shots) and shots[i + 1].get("fade_type") in
                             ("fade_to_black", "fade_to_white")))

        # Check brightness (from color palette — high V channel values)
        palette = shot.get("color_palette", [])
        if palette:
            bins = cfg["color_histogram_bins"]
            # V channel is the last third of the palette
            v_hist = palette[2 * bins:3 * bins] if len(palette) >= 3 * bins else []
            if v_hist:
                # High brightness = weight concentrated in upper bins
                upper_weight = sum(v_hist[bins // 2:])
                if upper_weight > 0.7 and has_fade_before and has_fade_after:
                    eyecatch_ids.append(shot["shot_id"])

    if eyecatch_ids:
        print(f"[Eyecatch Detection] Found eyecatch at shot(s): {eyecatch_ids}")
    else:
        # Fallback: find very short shots near midpoint with fades on both sides
        for i, shot in enumerate(shots):
            shot_mid = (shot["start_time"] + shot["end_time"]) / 2.0
            if abs(shot_mid - mid_point) > mid_window:
                continue
            if shot["duration"] > 3.0:
                continue
            has_fade_before = (i > 0 and shots[i - 1].get("fade_type") != "none")
            has_fade_after = (i + 1 < len(shots) and
                             shots[i + 1].get("fade_type") != "none")
            if has_fade_before and has_fade_after:
                eyecatch_ids.append(shot["shot_id"])
                break

        if eyecatch_ids:
            print(f"[Eyecatch Detection] Found eyecatch (fallback) at shot(s): "
                  f"{eyecatch_ids}")

    return eyecatch_ids


# ---------------------------------------------------------------------------
# Standalone entry point for tuning
# ---------------------------------------------------------------------------

def run_standalone(video_path, audio_path=None, output_path=None, config_overrides=None):
    """
    Run the full shot→scene pipeline standalone for tuning.
    Saves shot_index.json and scene_list.json to output_path.
    """
    cfg = {**CONFIG, **(config_overrides or {})}
    output_path = output_path or "output/scene_detector_debug"
    os.makedirs(output_path, exist_ok=True)

    print("=" * 60)
    print("Scene Detector — Standalone Mode")
    print("=" * 60)

    # Step 1: Shot detection
    t0 = time.time()
    shots = detect_shots(video_path, config=cfg)
    t1 = time.time()
    print(f"[Timing] Shot detection: {t1 - t0:.1f}s")

    # Step 2: Feature extraction
    shots = extract_shot_features(video_path, shots, audio_path=audio_path, config=cfg)
    t2 = time.time()
    print(f"[Timing] Feature extraction: {t2 - t1:.1f}s")

    # Step 3: Eyecatch detection
    eyecatch_ids = detect_eyecatch(shots, config=cfg)

    # Step 4: Scene detection
    scenes = detect_scenes(shots, config=cfg)
    t3 = time.time()
    print(f"[Timing] Scene detection: {t3 - t2:.1f}s")

    # Save outputs (strip internal fields for debug readability)
    shots_output = []
    for s in shots:
        s_copy = {k: v for k, v in s.items() if not k.startswith("_")}
        shots_output.append(s_copy)

    with open(os.path.join(output_path, "shot_index_debug.json"), "w") as f:
        json.dump(shots_output, f, indent=2)

    with open(os.path.join(output_path, "scene_list_debug.json"), "w") as f:
        json.dump(scenes, f, indent=2)

    with open(os.path.join(output_path, "eyecatch_ids.json"), "w") as f:
        json.dump(eyecatch_ids, f, indent=2)

    print(f"\n[Done] {len(shots)} shots → {len(scenes)} scenes")
    print(f"[Done] Outputs saved to {output_path}/")
    print(f"[Done] Total time: {t3 - t0:.1f}s")

    return shots, scenes, eyecatch_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scene Detector — Shot detection and scene grouping for anime episodes"
    )
    parser.add_argument("video", help="Path to episode video file (MKV/MP4)")
    parser.add_argument("--audio", help="Path to extracted audio (WAV). "
                        "If not provided, audio features are skipped.")
    parser.add_argument("--output", default="output/scene_detector_debug",
                        help="Output directory for debug files")
    parser.add_argument("--threshold", type=float,
                        help="Override scene break threshold")
    parser.add_argument("--bg-threshold", type=float,
                        help="Override background cosine distance threshold")

    args = parser.parse_args()

    overrides = {}
    if args.threshold is not None:
        overrides["scene_break_threshold"] = args.threshold
    if args.bg_threshold is not None:
        overrides["color_distance_threshold"] = args.bg_threshold

    run_standalone(args.video, audio_path=args.audio, output_path=args.output,
                   config_overrides=overrides)
