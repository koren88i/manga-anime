import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Input — same MKV as speaker_diarization.py
    "mkv_path": Path(r"c:\Users\User\helloworld_claude\mange-anime\anime\[Anime Time] One Piece - 0001 - I'm Luffy! The Man Who's Gonna Be King of the Pirates!.mkv"),

    # Output — same output dir as speaker_diarization.py
    "output_dir": Path(r"c:\Users\User\helloworld_claude\mange-anime\output"),

    # Anime and character info
    "anime_title": "One Piece",
    "character_names": ["Luffy", "Shanks", "Coby", "Alvida", "Nami"],

    # Frame extraction settings
    "frames_per_speaker": 10,

    # CLIP matching
    "clip_model": "ViT-B-32",
    "clip_pretrained": "openai",
    "match_threshold": 0.20,

    # Face detection
    "face_confidence": 0.5,
    "min_face_size": 40,  # pixels — ignore tiny background faces
}


# ============================================================
# STAGE 1: Encode Reference Images with CLIP
# ============================================================
def encode_references(config, downloaded):
    """Encode all reference images with CLIP and average per character."""
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        config["clip_model"], pretrained=config["clip_pretrained"]
    )
    model = model.to(device).eval()
    print(f"  CLIP model loaded on {device}")

    character_embeddings = {}

    for char_name, img_paths in downloaded.items():
        if not img_paths:
            print(f"  {char_name}: no reference images, skipping")
            continue

        embeddings = []
        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(img_tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu())
            except Exception as e:
                print(f"  Warning: Failed to encode {img_path.name}: {e}")

        if embeddings:
            avg_emb = torch.stack(embeddings).mean(dim=0)
            avg_emb = avg_emb / avg_emb.norm(dim=-1, keepdim=True)
            character_embeddings[char_name] = avg_emb
            print(f"  {char_name}: encoded {len(embeddings)} reference images")

    return model, preprocess, character_embeddings, device


# ============================================================
# STAGE 3: Extract Frames at Speaker Timestamps
# ============================================================
def extract_speaker_frames(config, manifest):
    """Extract video frames at each speaker's timestamps."""
    mkv_path = config["mkv_path"]
    frames_dir = config["output_dir"] / "frames"
    max_frames = config["frames_per_speaker"]

    all_frames = {}

    for speaker_id, speaker_data in manifest["speakers"].items():
        speaker_frames_dir = frames_dir / speaker_id
        speaker_frames_dir.mkdir(parents=True, exist_ok=True)

        clips = speaker_data["clips"]
        # Pick timestamps spread across the speaker's clips
        if len(clips) <= max_frames:
            selected = clips
        else:
            indices = np.linspace(0, len(clips) - 1, max_frames, dtype=int)
            selected = [clips[i] for i in indices]

        frame_paths = []
        for i, clip in enumerate(selected):
            # Use midpoint of each clip
            ts = (clip["start"] + clip["end"]) / 2
            frame_path = speaker_frames_dir / f"frame_{i + 1:04d}_{ts:.2f}.png"

            if frame_path.exists():
                frame_paths.append(frame_path)
                continue

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{ts:.3f}",
                "-i", str(mkv_path),
                "-vframes", "1",
                "-sn",
                "-q:v", "2",
                str(frame_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and frame_path.exists():
                frame_paths.append(frame_path)

        all_frames[speaker_id] = frame_paths
        print(f"  {speaker_id}: {len(frame_paths)} frames extracted")

    return all_frames


# ============================================================
# STAGE 4: Detect Anime Faces in Frames
# ============================================================
def detect_faces_in_frames(config, all_frames):
    """Detect anime faces in extracted frames using YOLOv8."""
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download

    print("  Downloading YOLOv8 anime face model...")
    model_path = hf_hub_download(
        repo_id="Fuyucchi/yolov8_animeface",
        filename="yolov8x6_animeface.pt",
    )
    print("  Loading model (CPU mode to avoid cuDNN conflicts)...")
    model = YOLO(model_path)

    min_size = config["min_face_size"]
    conf = config["face_confidence"]

    all_crops = {}

    for speaker_id, frame_paths in all_frames.items():
        crops = []
        for frame_path in frame_paths:
            try:
                results = model.predict(source=str(frame_path), conf=conf, verbose=False, device="cpu")
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue

                h, w = image.shape[:2]
                frame_center_x = w / 2

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        bw = x2 - x1
                        bh = y2 - y1

                        # Skip tiny faces (background characters)
                        if bw < min_size or bh < min_size:
                            continue

                        crop = image[y1:y2, x1:x2]
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_pil = Image.fromarray(crop_rgb)

                        # Save crop to disk for UI display
                        crops_dir = config["output_dir"] / "crops" / speaker_id
                        crops_dir.mkdir(parents=True, exist_ok=True)
                        crop_filename = f"crop_{frame_path.stem}_{len(crops):03d}.png"
                        crop_path = crops_dir / crop_filename
                        crop_pil.save(str(crop_path))

                        # Score: prefer larger, more central faces
                        face_center_x = (x1 + x2) / 2
                        centrality = 1.0 - abs(face_center_x - frame_center_x) / frame_center_x
                        area = bw * bh
                        confidence = box.conf[0].item()

                        crops.append({
                            "image": crop_pil,
                            "area": area,
                            "centrality": centrality,
                            "confidence": confidence,
                            "frame": frame_path.name,
                            "crop_path": str(crop_path),
                        })
            except Exception as e:
                print(f"  Warning: Face detection failed on {frame_path.name}: {e}")

        all_crops[speaker_id] = crops
        print(f"  {speaker_id}: {len(crops)} face crops detected")

    return all_crops


# ============================================================
# STAGE 5: Match Faces to Characters via CLIP
# ============================================================
def match_faces_to_characters(config, clip_model, preprocess, character_embeddings,
                               device, all_crops):
    """Match detected face crops against character reference embeddings."""
    threshold = config["match_threshold"]
    results = {}

    for speaker_id, crops in all_crops.items():
        if not crops:
            results[speaker_id] = {
                "predicted_character": None,
                "confidence": 0.0,
                "votes": {},
            }
            continue

        # Sort crops by area * centrality (best candidates first)
        crops.sort(key=lambda c: c["area"] * c["centrality"], reverse=True)

        # Match each crop against all characters
        votes = defaultdict(list)

        for crop_info in crops:
            try:
                img_tensor = preprocess(crop_info["image"]).unsqueeze(0).to(device)
                with torch.no_grad():
                    crop_emb = clip_model.encode_image(img_tensor)
                    crop_emb = crop_emb / crop_emb.norm(dim=-1, keepdim=True)

                for char_name, ref_emb in character_embeddings.items():
                    similarity = (crop_emb.cpu() @ ref_emb.T).item()
                    if similarity > threshold:
                        votes[char_name].append(similarity)
            except Exception:
                continue

        # Determine winner by: most votes, then highest average similarity
        if votes:
            scored = {}
            for char_name, sims in votes.items():
                scored[char_name] = {
                    "vote_count": len(sims),
                    "avg_similarity": sum(sims) / len(sims),
                    "max_similarity": max(sims),
                    "score": len(sims) * (sum(sims) / len(sims)),
                }

            winner = max(scored.items(), key=lambda x: x[1]["score"])
            results[speaker_id] = {
                "predicted_character": winner[0],
                "confidence": winner[1]["avg_similarity"],
                "votes": {k: v for k, v in sorted(scored.items(),
                                                    key=lambda x: x[1]["score"],
                                                    reverse=True)},
            }
        else:
            results[speaker_id] = {
                "predicted_character": None,
                "confidence": 0.0,
                "votes": {},
            }

        pred = results[speaker_id]["predicted_character"] or "Unknown"
        conf = results[speaker_id]["confidence"]
        print(f"  {speaker_id} -> {pred} (confidence: {conf:.3f})")

    return results


# ============================================================
# STAGE 6: Update Manifest & Rename Folders
# ============================================================
def update_and_rename(config, manifest, identification_results):
    """Update manifest with character names and rename speaker folders."""
    output_dir = config["output_dir"]
    speakers_dir = output_dir / "speakers"

    # Save detailed identification results
    id_path = output_dir / "character_identification.json"
    # Convert for JSON serialization
    serializable_results = {}
    for speaker_id, result in identification_results.items():
        serializable_results[speaker_id] = {
            "predicted_character": result["predicted_character"],
            "confidence": round(result["confidence"], 4),
            "votes": result["votes"],
        }
    with open(id_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"  Identification results saved to: {id_path}")

    # Track name assignments to handle duplicates
    used_names = {}
    rename_map = {}

    for speaker_id, result in identification_results.items():
        char_name = result["predicted_character"]
        if char_name is None:
            # Keep original name for unmatched speakers
            rename_map[speaker_id] = speaker_id
        elif char_name in used_names:
            # Duplicate — the one with higher confidence keeps the name
            existing_speaker = used_names[char_name]
            existing_conf = identification_results[existing_speaker]["confidence"]
            if result["confidence"] > existing_conf:
                # New one wins — demote the old one
                rename_map[existing_speaker] = existing_speaker
                rename_map[speaker_id] = char_name
                used_names[char_name] = speaker_id
            else:
                rename_map[speaker_id] = speaker_id
        else:
            rename_map[speaker_id] = char_name
            used_names[char_name] = speaker_id

    # Update manifest with character names
    for speaker_id, speaker_data in manifest["speakers"].items():
        char_name = rename_map.get(speaker_id, speaker_id)
        speaker_data["character_name"] = char_name

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest updated at: {manifest_path}")

    # Rename folders and clips
    for speaker_id, new_name in rename_map.items():
        if new_name == speaker_id:
            continue

        old_dir = speakers_dir / speaker_id
        new_dir = speakers_dir / new_name

        if not old_dir.exists():
            continue
        if new_dir.exists():
            print(f"  Warning: {new_dir} already exists, skipping rename for {speaker_id}")
            continue

        # Rename clip files inside the folder
        for clip_path in old_dir.glob("*.wav"):
            new_clip_name = clip_path.name.replace(speaker_id, new_name)
            clip_path.rename(old_dir / new_clip_name)

        # Rename the folder
        old_dir.rename(new_dir)
        print(f"  Renamed: {speaker_id} -> {new_name}")

    return rename_map


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Visual Character Identification Pipeline")
    print("=" * 60)

    output_dir = CONFIG["output_dir"]
    manifest_path = output_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found at {manifest_path}\n"
            "Run speaker_diarization.py first to generate speaker diarization output."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Loaded manifest: {len(manifest['speakers'])} speakers")

    start_time = time.time()

    # Load references from disk (managed via the UI)
    refs_dir = output_dir / "references"
    downloaded = {}
    for char_dir in sorted(refs_dir.iterdir()):
        if char_dir.is_dir():
            imgs = [p for p in char_dir.glob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if imgs:
                downloaded[char_dir.name] = imgs
    print(f"References: {len(downloaded)} characters ({', '.join(downloaded.keys())})")

    if not downloaded:
        raise RuntimeError(
            f"No reference images found in {refs_dir}\n"
            "Add reference images via the UI (Reference Images tab)."
        )

    print("\n[1/5] Encoding reference images with CLIP...")
    clip_model, preprocess, character_embeddings, device = encode_references(CONFIG, downloaded)

    if not character_embeddings:
        raise RuntimeError("No character embeddings could be created. Check reference images.")

    # Free CLIP from GPU before YOLO loads (avoids cuDNN conflicts)
    del clip_model
    torch.cuda.empty_cache()

    print("\n[2/5] Extracting frames at speaker timestamps...")
    all_frames = extract_speaker_frames(CONFIG, manifest)

    print("\n[3/5] Detecting anime faces in frames...")
    all_crops = detect_faces_in_frames(CONFIG, all_frames)

    print("\n[4/5] Matching faces to characters via CLIP...")
    # Reload CLIP for matching (lightweight, embeddings already computed)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CONFIG["clip_model"], pretrained=CONFIG["clip_pretrained"]
    )
    clip_model = clip_model.to(device).eval()

    identification_results = match_faces_to_characters(
        CONFIG, clip_model, preprocess, character_embeddings, device, all_crops
    )

    del clip_model
    torch.cuda.empty_cache()

    print("\n[5/5] Updating manifest and renaming folders...")
    rename_map = update_and_rename(CONFIG, manifest, identification_results)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Identification complete in {elapsed:.1f} seconds")
    print(f"\nResults:")
    for speaker_id, new_name in sorted(rename_map.items()):
        conf = identification_results[speaker_id]["confidence"]
        status = f"-> {new_name}" if new_name != speaker_id else "(unmatched)"
        print(f"  {speaker_id} {status} (conf: {conf:.3f})")
    print(f"\nCheck output/character_identification.json for detailed results")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
