import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import re
import json
import time
import base64
import argparse
from pathlib import Path

import logging
import cv2
import numpy as np
from openai import OpenAI, RateLimitError, APIError

manga_logger = logging.getLogger(__name__)

# ============================================================
# OPTIONAL: Wiki grounding for chapter context
# ============================================================
try:
    from wiki.manga_wiki_api import (
        get_chapter_characters,
        get_chapter_summary,
        get_chapter_short_summary,
        get_chapter_infobox,
        get_chapter_notes,
        get_chapter_cover_page,
        get_arc_for_chapter,
    )
    from wiki.one_piece_wiki_api import normalize_arc_id
    HAS_MANGA_WIKI_API = True
except ImportError:
    HAS_MANGA_WIKI_API = False

    def normalize_arc_id(arc_name):
        """Fallback when wiki API is unavailable."""
        if not arc_name:
            return ""
        import re as _re
        name = _re.sub(r"\s+(Arc|Saga)$", "", arc_name.strip(), flags=_re.IGNORECASE)
        return _re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Input -- set by CLI args
    "chapter_dir": None,
    "chapter_id": None,

    # Output -- set by CLI args (or derived from chapter_id)
    "output_dir": None,

    # OpenAI API
    "model": "gpt-4o",
    "max_tokens": 4096,
    "temperature": 0.1,

    # Processing
    "resume": False,
    "context_window": 2,        # Previous pages to include as context
    "retry_attempts": 3,
    "retry_delay_base": 2.0,

    # Rate limiting
    "delay_between_requests": 2.0,

    # Wiki context (populated at runtime)
    "wiki_context": None,
}

# ============================================================
# CHARACTER KNOWLEDGE BASE
# ============================================================
# Core aliases for recurring characters (cross-chapter).
# Wiki-derived aliases are merged at runtime via build_character_aliases().
CORE_ALIASES = {
    # Straw Hats
    "monkey d. luffy": "luffy",
    "monkey d luffy": "luffy",
    "straw hat luffy": "luffy",
    "roronoa zoro": "zoro",
    "pirate hunter zoro": "zoro",
    "cat burglar nami": "nami",
    "usopp": "usopp",
    "sogeking": "usopp",
    "vinsmoke sanji": "sanji",
    "black leg sanji": "sanji",
    "tony tony chopper": "chopper",
    "nico robin": "robin",
    "cyborg franky": "franky",
    "cutty flam": "franky",
    "soul king brook": "brook",
    "jinbe": "jinbe",
    "jimbei": "jinbe",
    # Roger
    "gol d. roger": "gold_roger",
    "gol d roger": "gold_roger",
    "gold roger": "gold_roger",
    "pirate king roger": "gold_roger",
    # Shanks crew
    "red-haired shanks": "shanks",
    "red hair shanks": "shanks",
    "captain shanks": "shanks",
    "benn beckman": "ben_beckman",
    "ben beckman": "ben_beckman",
    # Supporting
    "mayor woop slap": "woop_slap",
}

# Module-level alias dict -- replaced at startup by build_character_aliases()
CHARACTER_ALIASES = dict(CORE_ALIASES)


def build_character_aliases(wiki_context=None):
    """Build CHARACTER_ALIASES from core aliases + wiki character data."""
    aliases = dict(CORE_ALIASES)

    if wiki_context and wiki_context.get("characters_raw"):
        for char in wiki_context["characters_raw"]:
            name = char["name"]
            wiki_page = char["wiki_page"]

            # Canonical ID: snake_case of wiki_page
            canonical = re.sub(r"[^a-z0-9]+", "_", wiki_page.lower()).strip("_")

            # Add name -> canonical
            name_lower = name.lower().strip()
            if name_lower not in aliases:
                aliases[name_lower] = canonical

            # Add wiki_page -> canonical
            wp_lower = wiki_page.lower().strip()
            if wp_lower not in aliases:
                aliases[wp_lower] = canonical

    return aliases


# ============================================================
# SYSTEM PROMPT BUILDER
# ============================================================

# Chapter-agnostic structural rules (reading order, text types, JSON schema)
PROMPT_STRUCTURAL_RULES = """
READING ORDER RULES:
- Panels are read RIGHT-TO-LEFT, TOP-TO-BOTTOM (traditional manga convention)
- Within each panel, speech bubbles are also read roughly right-to-left, top-to-bottom
- The text CONTENT is English (read left-to-right within each bubble)

TEXT ELEMENT TYPES:
- "dialog": Spoken words in speech bubbles. Always attribute to a speaker.
- "narration": Text in rectangular narration boxes (usually at edges of panels). No speaker.
- "internal_monologue": Character's thoughts, often in cloud-shaped or special bubbles.
- "sfx": Sound effects - both English text and descriptions of Japanese onomatopoeia visible in the art.
- "caption": Story captions, location/time indicators.

SPEECH BUBBLE TYPES (for "speech_bubble_type" field ONLY -- do NOT use these as "type" values):
- "speech": Normal round speech bubble
- "thought": Cloud-shaped or dotted bubble for internal thoughts
- "shout": Jagged/spiky bubble or very large bold text indicating yelling
- "whisper": Small text, sometimes with dotted bubble
- "narration_box": Rectangular box for narration/captions
IMPORTANT: "shout" and "whisper" are speech_bubble_type values describing delivery style. The "type" field must always be one of: dialog, narration, internal_monologue, sfx, caption. A character yelling in a shout bubble has type="dialog" and speech_bubble_type="shout".

SOUND EFFECT CATEGORIES:
- "impact": Punches, crashes, collisions, breaking things
- "environment": Wind, water, rain, ambient sounds
- "action": Movement sounds, swooshes, footsteps
- "emotion": Dramatic stings, shock effects, comedic sounds
- "voice": Gasps, sighs, laughter (not dialog)

PANEL BOUNDING BOXES:
- For each panel, provide a "bbox" array of 4 floats: [x1, y1, x2, y2]
- These are FRACTIONAL coordinates (0.0 to 1.0) relative to the full page image dimensions
- x1,y1 is the top-left corner; x2,y2 is the bottom-right corner
- Example: a panel in the top-right quadrant might be [0.5, 0.0, 1.0, 0.5]
- For full-page panels, use [0.0, 0.0, 1.0, 1.0]
- Be as precise as possible — the bbox will be used to crop the panel from the page image

You MUST respond with ONLY valid JSON. No markdown code fences, no commentary outside the JSON.

Respond with this exact JSON structure:
{
  "page_number": <int>,
  "source_file": "<filename>",
  "page_type": "cover" | "credits" | "title_spread" | "story",
  "page_narrative": "<1-3 sentence summary of what happens on this page as a whole, advancing the story>",
  "panels": [
    {
      "panel_id": "<page>_panel_<n>",
      "reading_order": <int starting at 1>,
      "position_description": "<e.g. top-right, middle, bottom-left, full-page>",
      "bbox": [<x1>, <y1>, <x2>, <y2>],
      "scene_description": "<brief visual description of what's happening>",
      "panel_narrative": "<what happens in this panel -- the action, reaction, or story beat being conveyed>",
      "location": "<setting/place where this panel occurs>",
      "time_context": "<temporal context -- e.g. '10 years ago', 'present day', 'flashback'. Use 'present day' if no time shift>",
      "characters_present": [
        {
          "character_id": "<snake_case id>",
          "name": "<display name>",
          "action": "<what they're doing>"
        }
      ],
      "text_elements": [
        {
          "text_id": "<page>_panel_<n>_text_<m>",
          "type": "dialog" | "narration" | "internal_monologue" | "sfx" | "caption",
          "content": "<the actual text>",
          "speaker": "<character name or null>",
          "speaker_id": "<character_id or null>",
          "speaker_confidence": <0.0-1.0 or null>,
          "speech_bubble_type": "speech" | "thought" | "shout" | "whisper" | "narration_box",
          "emotion": "<emotional tone e.g. angry, happy, scared, dramatic, calm>",
          "tone_direction": "<TTS voice direction e.g. 'loud and excited', 'calm and warm', 'deep solemn narrator'>",
          "reading_order": <int within this panel>
        }
      ],
      "sound_effects": [
        {
          "sfx_id": "<page>_panel_<n>_sfx_<m>",
          "content": "<English text if visible, or transliteration>",
          "description": "<what the sound is, e.g. 'glass shattering', 'punch landing'>",
          "intensity": <1-10>,
          "category": "impact" | "environment" | "action" | "emotion" | "voice",
          "japanese_text_visible": <true/false>
        }
      ]
    }
  ]
}

For non-story pages (cover, credits, title_spread), panels can be empty or contain a single entry describing the page."""


def build_system_prompt(wiki_context=None):
    """Build the system prompt dynamically from wiki context (or generic fallback)."""
    parts = []

    # Header
    parts.append("You are a manga metadata extraction expert.")
    if wiki_context and wiki_context.get("chapter_title"):
        ch_num = wiki_context.get("chapter_number", "?")
        title = wiki_context["chapter_title"]
        parts.append(
            f'You are analyzing a COLORIZED ENGLISH TRANSLATION (fan scanlation) '
            f'of One Piece Chapter {ch_num}: "{title}".'
        )
    else:
        parts.append(
            "You are analyzing a COLORIZED ENGLISH TRANSLATION (fan scanlation) "
            "of a One Piece manga chapter."
        )
    parts.append("")
    parts.append(
        "Your task is to extract structured metadata from each manga page for a "
        "text-to-speech and audio generation pipeline that will bring the manga "
        "to life with voice acting and sound effects."
    )

    # Character guide
    parts.append("")
    if wiki_context and wiki_context.get("character_groups"):
        parts.append("CHARACTER IDENTIFICATION GUIDE (use ONLY these character names):")
        for group, names in wiki_context["character_groups"].items():
            name_list = ", ".join(names)
            parts.append(f'- {group}: {name_list}')
        parts.append('- For unnamed characters use descriptive IDs like "bandit_1", '
                      '"villager_1", "pirate_crew_member", "crowd"')
        parts.append('- CRITICAL: Do NOT use generic labels like "unknown", '
                      '"Character 1", "man", "boy". If you cannot identify a '
                      'character from the list above, omit them from characters_present '
                      'rather than guessing.')
    else:
        parts.append("CHARACTER IDENTIFICATION:")
        parts.append(
            "- Identify characters by their visual appearance and any in-panel "
            "name mentions. Use snake_case IDs (e.g. \"luffy\", \"zoro\")."
        )
        parts.append('- For unnamed characters use descriptive IDs like "bandit_1", '
                      '"villager_1", "pirate_crew_member", "crowd"')

    # Structural rules (chapter-agnostic)
    parts.append(PROMPT_STRUCTURAL_RULES)

    return "\n".join(parts)


# ============================================================
# WIKI CONTEXT FETCHER
# ============================================================

def fetch_wiki_chapter_context(chapter_id):
    """
    Fetch structured chapter context from the One Piece wiki.

    Returns a dict with character_list, synopsis, arc info, etc.,
    or None if the wiki API is unavailable or the fetch fails.
    """
    if not HAS_MANGA_WIKI_API:
        return None

    ch_num = int(chapter_id) if chapter_id.isdigit() else None
    if ch_num is None:
        return None

    try:
        characters = get_chapter_characters(ch_num)
        summary = get_chapter_summary(ch_num)
        short_summary = get_chapter_short_summary(ch_num)
        infobox = get_chapter_infobox(ch_num)
        notes = get_chapter_notes(ch_num)
        cover = get_chapter_cover_page(ch_num)
    except Exception as e:
        print(f"  Wiki API fetch failed for chapter {ch_num}: {e}")
        return None

    # Build flat character list with annotations
    char_entries = []
    for c in characters:
        entry = c["name"]
        if c.get("annotation"):
            entry += f" ({c['annotation']})"
        char_entries.append(entry)

    # Build grouped character dict
    character_groups = {}
    for c in characters:
        group = c.get("group") or "Others"
        character_groups.setdefault(group, []).append(c["name"])

    # Get arc info
    arc_info = None
    try:
        arc_info = get_arc_for_chapter(ch_num)
    except Exception:
        pass

    # Extract chapter title from infobox
    chapter_title = (
        infobox.get("viz_title")
        or infobox.get("english_title")
        or infobox.get("romanized_title")
        or ""
    )

    context = {
        "chapter_number": ch_num,
        "chapter_title": chapter_title,
        "synopsis": summary,
        "short_summary": short_summary,
        "characters": char_entries,
        "characters_raw": characters,
        "character_groups": character_groups,
        "arc": arc_info["arc"] if arc_info else "",
        "saga": arc_info["saga"] if arc_info else "",
        "cover_page": cover,
        "chapter_notes": notes,
        "infobox": infobox,
    }

    return context


# ============================================================
# UTILITIES
# ============================================================
def encode_image(image_path):
    """Read and base64-encode an image file. Returns (base64_str, media_type)."""
    ext = image_path.suffix.lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return data, media_type


def normalize_character_id(name):
    """Normalize a character name to a stable snake_case ID."""
    normalized = name.lower().strip()
    if normalized in CHARACTER_ALIASES:
        return CHARACTER_ALIASES[normalized]
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def compute_iou(box1, box2):
    """IOU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def match_panels_by_iou(vlm_panels, kumiko_bboxes, threshold=0.2):
    """Greedy IOU matching between VLM panels and Kumiko bboxes.

    Returns list of (vlm_idx, kumiko_idx) pairs.
    """
    pairs = []
    for vi, vp in enumerate(vlm_panels):
        vbox = vp.get("bbox")
        if not vbox or len(vbox) != 4:
            continue
        for ki, kbox in enumerate(kumiko_bboxes):
            iou = compute_iou(vbox, kbox)
            pairs.append((iou, vi, ki))

    pairs.sort(reverse=True)
    matched = []
    used_vlm, used_kumiko = set(), set()
    for iou, vi, ki in pairs:
        if iou < threshold:
            break
        if vi not in used_vlm and ki not in used_kumiko:
            matched.append((vi, ki))
            used_vlm.add(vi)
            used_kumiko.add(ki)
    return matched


def _panel_sort_key(panel):
    """Sort key for manga reading order: top-to-bottom, then right-to-left.

    Uses bbox when available, falls back to reading_order field.
    Panels on the same row (vertical overlap > 50% of shorter panel height)
    are sorted right-to-left; otherwise top-to-bottom.
    """
    bbox = panel.get("bbox")
    if bbox and len(bbox) == 4:
        # Return (y_center, -x_center) as a simple approximation.
        # The full row-grouping logic is in sort_panels_reading_order().
        y_center = (bbox[1] + bbox[3]) / 2
        x_center = (bbox[0] + bbox[2]) / 2
        return (y_center, -x_center)
    # Fallback: use reading_order field (1-indexed)
    return (panel.get("reading_order", 999), 0)


def sort_panels_reading_order(panels):
    """Sort panels into manga reading order (top-to-bottom, right-to-left).

    Groups panels into rows based on vertical overlap, then sorts
    right-to-left within each row. Panels without bboxes are sorted
    by their reading_order field.
    """
    if len(panels) <= 1:
        return panels

    # Separate panels with and without bboxes
    with_bbox = []
    without_bbox = []
    for i, p in enumerate(panels):
        bbox = p.get("bbox")
        if bbox and len(bbox) == 4:
            with_bbox.append((i, p, bbox))
        else:
            without_bbox.append(p)

    if not with_bbox:
        # No bboxes at all -- sort by reading_order field only
        return sorted(panels, key=lambda p: p.get("reading_order", 999))

    # Sort by vertical position first
    with_bbox.sort(key=lambda t: t[2][1])  # sort by y1

    # Group into rows: panels overlap vertically if their y-ranges share
    # more than 50% of the shorter panel's height
    rows = []
    current_row = [with_bbox[0]]
    for item in with_bbox[1:]:
        _, _, bbox = item
        # Compare with the first panel in the current row
        _, _, row_bbox = current_row[0]
        # Vertical overlap
        overlap_top = max(bbox[1], row_bbox[1])
        overlap_bot = min(bbox[3], row_bbox[3])
        overlap = max(0, overlap_bot - overlap_top)
        shorter_h = min(bbox[3] - bbox[1], row_bbox[3] - row_bbox[1])
        if shorter_h > 0 and overlap / shorter_h >= 0.5:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    # Within each row, sort right-to-left (descending x1) for manga RTL
    sorted_panels = []
    for row in rows:
        row.sort(key=lambda t: -t[2][0])  # sort by -x1 (rightmost first)
        sorted_panels.extend(p for _, p, _ in row)

    # Append panels without bboxes at the end, sorted by reading_order
    without_bbox.sort(key=lambda p: p.get("reading_order", 999))
    sorted_panels.extend(without_bbox)

    # Update reading_order to reflect the new sort
    for i, panel in enumerate(sorted_panels):
        panel["reading_order"] = i + 1

    return sorted_panels


def validate_page_json(data, page_number, kumiko_bboxes=None):
    """Validate and clean the extracted page metadata."""
    if "page_number" not in data:
        data["page_number"] = page_number
    if "panels" not in data:
        data["panels"] = []
    if "page_type" not in data:
        data["page_type"] = "story"
    if "page_narrative" not in data:
        data["page_narrative"] = ""

    for panel in data["panels"]:
        if "panel_narrative" not in panel:
            panel["panel_narrative"] = panel.get("scene_description", "")
        if "location" not in panel:
            panel["location"] = ""
        if "time_context" not in panel:
            panel["time_context"] = "present day"
        if "text_elements" not in panel:
            panel["text_elements"] = []
        if "sound_effects" not in panel:
            panel["sound_effects"] = []
        if "characters_present" not in panel:
            panel["characters_present"] = []

        # Normalize character IDs
        for char in panel["characters_present"]:
            if "character_id" in char:
                char["character_id"] = normalize_character_id(
                    char.get("name", char["character_id"])
                )

        VALID_TEXT_TYPES = {"dialog", "narration", "internal_monologue", "sfx", "caption"}
        BUBBLE_TYPE_AS_TEXT_TYPE = {"shout", "whisper", "speech", "thought", "narration_box"}

        for te in panel["text_elements"]:
            # Normalize misclassified type values: bubble types used as text types
            te_type = te.get("type", "dialog")
            if te_type not in VALID_TEXT_TYPES:
                if te_type in BUBBLE_TYPE_AS_TEXT_TYPE:
                    te["type"] = "dialog"
                else:
                    te["type"] = "dialog"

            if te.get("speaker_id"):
                te["speaker_id"] = normalize_character_id(
                    te.get("speaker", te["speaker_id"])
                )

    # Override bboxes with Kumiko data when available (always use IOU matching)
    if kumiko_bboxes:
        vlm_panels = data.get("panels", [])
        matched = match_panels_by_iou(vlm_panels, kumiko_bboxes)
        matched_vlm = {m[0] for m in matched}
        for vlm_idx, kumiko_idx in matched:
            vlm_panels[vlm_idx]["bbox"] = list(kumiko_bboxes[kumiko_idx])
            vlm_panels[vlm_idx]["bbox_source"] = "kumiko"
        # Tag unmatched VLM panels
        for i, panel in enumerate(vlm_panels):
            if i not in matched_vlm:
                panel["bbox_source"] = "vlm"

    # Sort panels into manga reading order (RTL, top-to-bottom)
    data["panels"] = sort_panels_reading_order(data.get("panels", []))

    return data


def validate_bounding_box(bbox):
    """Validate and clamp a panel bounding box.

    Args:
        bbox: list of 4 floats [x1, y1, x2, y2] in 0.0-1.0 range.

    Returns:
        Clamped (x1, y1, x2, y2) tuple, or None if invalid.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return None

    # Clamp to [0, 1]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    # Must have positive area (at least 1% of page in each dimension)
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None

    return (x1, y1, x2, y2)


def crop_and_save_panels(page_metadata, page_image_path, panels_dir):
    """Crop panel images from the page using VLM bounding boxes.

    Args:
        page_metadata: validated page JSON dict (panels may have 'bbox' field)
        page_image_path: Path to the source page image
        panels_dir: Path to the panels/ output directory

    Returns:
        Number of panels successfully cropped. Panel dicts are updated
        in-place with 'panel_image_path' (relative to output_dir).
    """
    img = cv2.imread(str(page_image_path))
    if img is None:
        print(f"    Warning: could not read page image for cropping: {page_image_path}")
        return 0

    h, w = img.shape[:2]
    panels_dir.mkdir(parents=True, exist_ok=True)
    cropped_count = 0

    page_num = page_metadata.get("page_number", 0)

    for panel in page_metadata.get("panels", []):
        raw_bbox = panel.get("bbox")
        bbox = validate_bounding_box(raw_bbox)
        if bbox is None:
            # Graceful degradation: skip crop, leave panel_image_path unset
            panel["panel_image_path"] = None
            continue

        x1, y1, x2, y2 = bbox
        px1 = int(x1 * w)
        py1 = int(y1 * h)
        px2 = int(x2 * w)
        py2 = int(y2 * h)

        crop = img[py1:py2, px1:px2]
        if crop.size == 0:
            panel["panel_image_path"] = None
            continue

        panel_id = panel.get("panel_id", f"{page_num:03d}_panel_{panel.get('reading_order', 0)}")
        filename = f"{panel_id}.png"
        crop_path = panels_dir / filename

        cv2.imwrite(str(crop_path), crop)
        # Store path relative to the chapter output dir (panels/filename)
        panel["panel_image_path"] = f"panels/{filename}"
        cropped_count += 1

    return cropped_count


def load_manga_gpu_models(use_depth=False, use_segmentation=False, device=None):
    """Load optional GPU models for panel crop processing.

    Returns a dict of loaded models (empty dict if none requested).
    Models that fail to load are logged and skipped -- the pipeline
    continues without them.

    Args:
        use_depth: Load a depth estimation model (MiDaS/DPT)
        use_segmentation: Load a segmentation model (SAM or similar)
        device: Force a specific device ("cuda" or "cpu"). Auto-detects if None.

    Returns:
        {"depth_model": ..., "depth_transform": ..., "seg_model": ...,
         "seg_processor": ..., "device": ...}
        Missing keys mean the model was not loaded.
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    models = {}
    models["device"] = device

    if use_depth:
        try:
            print(f"[GPU Models] Loading MiDaS depth model on {device}...")
            # Use torch hub MiDaS -- widely available, no extra pip install needed
            depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            depth_model.to(device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            depth_transform = midas_transforms.small_transform
            models["depth_model"] = depth_model
            models["depth_transform"] = depth_transform
            print("[GPU Models] MiDaS depth model loaded")
        except Exception as e:
            manga_logger.warning(f"Could not load depth model: {e}")
            print(f"[GPU Models] Depth model unavailable: {e}")

    if use_segmentation:
        try:
            print(f"[GPU Models] Loading segmentation model on {device}...")
            # Use torchvision's DeepLabV3 -- available via torchvision, no extra install
            import torchvision
            seg_model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
                weights="DEFAULT"
            )
            seg_model.to(device).eval()
            models["seg_model"] = seg_model
            print("[GPU Models] DeepLabV3 segmentation model loaded")
        except Exception as e:
            manga_logger.warning(f"Could not load segmentation model: {e}")
            print(f"[GPU Models] Segmentation model unavailable: {e}")

    return models


def compute_depth_map(crop_bgr, depth_model, depth_transform, device):
    """Estimate depth map for a panel crop.

    Args:
        crop_bgr: numpy array (BGR from OpenCV)
        depth_model: Loaded MiDaS model
        depth_transform: MiDaS input transform
        device: torch device string

    Returns:
        Depth map as uint8 numpy array (0-255, higher = closer), or None on failure.
    """
    import torch
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        input_batch = depth_transform(rgb).to(device)

        with torch.no_grad():
            prediction = depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=crop_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        # Normalize to 0-255 uint8
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth, dtype=np.uint8)

        return depth_normalized
    except Exception as e:
        manga_logger.warning(f"Depth estimation failed: {e}")
        return None


def compute_segmentation_mask(crop_bgr, seg_model, device):
    """Segment a panel crop using DeepLabV3.

    Args:
        crop_bgr: numpy array (BGR from OpenCV)
        seg_model: Loaded DeepLabV3 model
        device: torch device string

    Returns:
        Segmentation mask as uint8 numpy array (class indices), or None on failure.
    """
    import torch
    import torchvision.transforms as T
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(520),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = seg_model(input_tensor)["out"]
            pred = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

        # Resize back to original crop size
        pred_resized = cv2.resize(pred, (crop_bgr.shape[1], crop_bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        return pred_resized
    except Exception as e:
        manga_logger.warning(f"Segmentation failed: {e}")
        return None


def process_panel_gpu_models(panel, panels_dir, gpu_models, resume=False):
    """Run depth estimation and segmentation on a single panel crop.

    Reads the panel crop from disk, processes with available GPU models,
    and saves results alongside the crop. Updates panel dict in-place
    with depth_map_path and segmentation_map_path.

    Args:
        panel: Panel dict (must have panel_image_path set)
        panels_dir: Path to panels/ output directory
        gpu_models: Dict from load_manga_gpu_models()
        resume: Skip if output files already exist

    Returns:
        (depth_ok, seg_ok) tuple of booleans
    """
    panel_image_path = panel.get("panel_image_path")
    if not panel_image_path:
        panel.setdefault("depth_map_path", None)
        panel.setdefault("segmentation_map_path", None)
        return False, False

    # Derive the full path from panels_dir parent (panel_image_path is "panels/XXX.png")
    full_crop_path = panels_dir.parent / panel_image_path
    if not full_crop_path.exists():
        panel.setdefault("depth_map_path", None)
        panel.setdefault("segmentation_map_path", None)
        return False, False

    panel_id = panel.get("panel_id", "unknown")
    device = gpu_models.get("device", "cpu")
    depth_ok = False
    seg_ok = False

    # Depth estimation
    if "depth_model" in gpu_models:
        depth_filename = f"{panel_id}_depth.png"
        depth_path = panels_dir / depth_filename
        rel_depth_path = f"panels/{depth_filename}"

        if resume and depth_path.exists():
            panel["depth_map_path"] = rel_depth_path
            depth_ok = True
        else:
            crop = cv2.imread(str(full_crop_path))
            if crop is not None:
                depth_map = compute_depth_map(
                    crop, gpu_models["depth_model"],
                    gpu_models["depth_transform"], device
                )
                if depth_map is not None:
                    cv2.imwrite(str(depth_path), depth_map)
                    panel["depth_map_path"] = rel_depth_path
                    depth_ok = True
                else:
                    panel["depth_map_path"] = None
            else:
                panel["depth_map_path"] = None
    else:
        panel.setdefault("depth_map_path", None)

    # Segmentation
    if "seg_model" in gpu_models:
        seg_filename = f"{panel_id}_seg.png"
        seg_path = panels_dir / seg_filename
        rel_seg_path = f"panels/{seg_filename}"

        if resume and seg_path.exists():
            panel["segmentation_map_path"] = rel_seg_path
            seg_ok = True
        else:
            crop = cv2.imread(str(full_crop_path))
            if crop is not None:
                seg_mask = compute_segmentation_mask(
                    crop, gpu_models["seg_model"], device
                )
                if seg_mask is not None:
                    cv2.imwrite(str(seg_path), seg_mask)
                    panel["segmentation_map_path"] = rel_seg_path
                    seg_ok = True
                else:
                    panel["segmentation_map_path"] = None
            else:
                panel["segmentation_map_path"] = None
    else:
        panel.setdefault("segmentation_map_path", None)

    return depth_ok, seg_ok


def summarize_page_for_context(metadata):
    """Create a brief text summary of a page for use as context in subsequent pages."""
    page_type = metadata.get("page_type", "story")
    if page_type in ("cover", "credits"):
        return f"Page {metadata['page_number']}: {page_type} page."

    # Collect locations and time contexts for this page
    locations = set()
    time_contexts = set()
    for panel in metadata.get("panels", []):
        loc = panel.get("location", "")
        tc = panel.get("time_context", "")
        if loc:
            locations.add(loc)
        if tc and tc != "present day":
            time_contexts.add(tc)

    context_prefix = ""
    if locations or time_contexts:
        tags = list(locations) + list(time_contexts)
        context_prefix = f"[{', '.join(tags)}] "

    # Prefer page_narrative if available
    page_narrative = metadata.get("page_narrative", "")
    if page_narrative:
        return f"Page {metadata['page_number']}: {context_prefix}{page_narrative}"

    parts = []
    for panel in metadata.get("panels", []):
        narrative = panel.get("panel_narrative", "") or panel.get("scene_description", "")
        if narrative:
            parts.append(narrative)

        dialogs = [
            f'{te.get("speaker", "?")}: "{te["content"]}"'
            for te in panel.get("text_elements", [])
            if te.get("type") == "dialog"
        ]
        if dialogs:
            parts.extend(dialogs[:3])

    summary = " | ".join(parts[:6])
    return f"Page {metadata['page_number']}: {context_prefix}{summary}"


# ============================================================
# STAGE 1: Discover Pages
# ============================================================
def discover_pages(config):
    """Scan chapter directory for manga page images, sorted numerically."""
    chapter_dir = config["chapter_dir"]
    if not chapter_dir.exists():
        raise FileNotFoundError(f"Chapter directory not found: {chapter_dir}")

    pages = []
    for f in chapter_dir.iterdir():
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            try:
                page_num = int(f.stem)
                pages.append((page_num, f))
            except ValueError:
                print(f"  Skipping non-numeric filename: {f.name}")

    pages.sort(key=lambda x: x[0])
    return pages


def load_kumiko_chapter_data(chapter_id):
    """Load Kumiko panel detection data for the chapter, if available."""
    kumiko_path = Path(f"output/kumiko_panels/{chapter_id}/kumiko_panels.json")
    if not kumiko_path.exists():
        return None
    with open(kumiko_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_kumiko_bboxes(kumiko_data, page_filename):
    """Get normalized [x1,y1,x2,y2] bboxes for a page from Kumiko data.

    Returns list of [x1,y1,x2,y2] or None if unavailable.
    """
    if not kumiko_data:
        return None
    page_data = kumiko_data.get("pages", {}).get(page_filename)
    if not page_data or not page_data.get("panels"):
        return None
    return [p["bbox"] for p in page_data["panels"]]


# ============================================================
# STAGE 2: Load Progress (for resumability)
# ============================================================
def load_progress(config):
    """Load previously processed pages and character registry."""
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find completed pages
    completed = set()
    page_contexts = {}

    for f in output_dir.iterdir():
        if f.name.startswith("page_") and f.suffix == ".json":
            try:
                page_num = int(f.stem.split("_")[1])
                completed.add(page_num)

                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    page_contexts[page_num] = summarize_page_for_context(data)
            except (ValueError, json.JSONDecodeError):
                pass

    # Load character registry
    registry_path = output_dir / "character_registry.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {
            "chapter_id": config["chapter_id"],
            "characters": {},
        }

    return completed, registry, page_contexts


# ============================================================
# STAGE 3: Build Prompt
# ============================================================
def get_relevant_synopsis(synopsis, page_number, total_pages):
    """Extract the paragraph(s) of the synopsis most relevant to the current page.

    Uses proportional mapping: page position in the chapter maps to paragraph
    position in the synopsis.  Window radius adapts to the pages-per-paragraph
    ratio so that sparse synopses (few paragraphs, many pages) get wider
    windows while dense synopses get narrower ones.

    For short synopses (<=1500 chars or <=2 paragraphs), returns the full text.

    Returns:
        (text, is_windowed) -- the relevant excerpt and whether it was narrowed.
    """
    if not synopsis or len(synopsis) <= 1500:
        return synopsis, False

    paragraphs = [p.strip() for p in synopsis.split("\n\n") if p.strip()]
    num_paras = len(paragraphs)

    if num_paras <= 2:
        return synopsis, False

    # Adaptive window radius based on pages-per-paragraph ratio:
    #   ~3 pages/para (dense, e.g. ch 1)  -> radius 1  -> 3 paragraphs
    #   ~6 pages/para (sparse, e.g. ch 200) -> radius 2  -> 5 paragraphs
    #   very sparse  -> wider still, naturally covers entire synopsis
    pages_per_para = total_pages / num_paras
    radius = max(1, round(pages_per_para / 3))

    # Proportional mapping: page position -> paragraph index
    progress = max(0.0, page_number - 1) / max(total_pages - 1, 1)
    progress = min(progress, 1.0)
    target_index = int(progress * (num_paras - 1))

    # Window: target paragraph +/- radius
    start = max(0, target_index - radius)
    end = min(num_paras, target_index + radius + 1)
    window = paragraphs[start:end]

    # If window covers all paragraphs, just return the full synopsis
    if start == 0 and end == num_paras:
        return synopsis, False

    text = "\n\n".join(window)
    if start > 0:
        text = "...\n\n" + text
    if end < num_paras:
        text = text + "\n\n..."

    # Safety cap
    if len(text) > 2000:
        text = text[:2000] + "..."

    return text, True


def build_prompt(page_path, page_number, character_registry, previous_context, config, kumiko_bboxes=None):
    """Construct the vision API messages for a single page."""
    image_data, media_type = encode_image(page_path)
    wiki_context = config.get("wiki_context")

    # Build context text
    context_parts = []

    # Chapter synopsis from wiki (windowed to relevant paragraphs)
    if wiki_context and wiki_context.get("synopsis"):
        total_pages = config.get("total_pages")
        if total_pages:
            synopsis, is_windowed = get_relevant_synopsis(
                wiki_context["synopsis"], page_number, total_pages
            )
        else:
            # Fallback: old truncation if total_pages not in config
            synopsis = wiki_context["synopsis"]
            is_windowed = False
            if len(synopsis) > 1500:
                synopsis = synopsis[:1500] + "..."

        if is_windowed:
            label = f"RELEVANT CHAPTER SYNOPSIS (page {page_number} of {total_pages})"
        else:
            label = "CHAPTER SYNOPSIS"
        context_parts.append(f"{label}:\n  {synopsis}")

    # Chapter notes from wiki
    if wiki_context and wiki_context.get("chapter_notes"):
        notes = wiki_context["chapter_notes"][:8]  # Cap at 8 notes
        context_parts.append("KEY CHAPTER EVENTS:")
        for note in notes:
            context_parts.append(f"  - {note}")

    # Previous page context
    if previous_context:
        context_parts.append("\nCONTEXT FROM PREVIOUS PAGES:")
        window = config["context_window"]
        for pn in sorted(previous_context.keys()):
            if page_number - window <= pn < page_number:
                context_parts.append(f"  - {previous_context[pn]}")

    # Known characters
    known_chars = character_registry.get("characters", {})
    if known_chars:
        context_parts.append("\nKNOWN CHARACTERS DISCOVERED SO FAR:")
        for cid, cdata in known_chars.items():
            name = cdata.get("name", cid)
            desc = cdata.get("visual_description", "")
            context_parts.append(f'  - {cid}: {name} - {desc}')

    # Pre-detected panel boundaries from Kumiko
    if kumiko_bboxes:
        context_parts.append(f"\nPRE-DETECTED PANEL BOUNDARIES (computer vision):")
        context_parts.append(f"This page has exactly {len(kumiko_bboxes)} panels:")
        for i, (x1, y1, x2, y2) in enumerate(kumiko_bboxes):
            context_parts.append(f"  Panel {i+1}: bbox [{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}]")
        context_parts.append(
            "IMPORTANT: Return EXACTLY this many panels. Copy each panel's bbox "
            "values exactly into the corresponding panel's 'bbox' field. "
            "Do NOT estimate or change the bounding boxes."
        )

    context_text = "\n".join(context_parts) if context_parts else ""

    # Dynamic chapter reference
    chapter_ref = f"Chapter {config['chapter_id']}"
    if wiki_context and wiki_context.get("chapter_title"):
        chapter_ref = f"One Piece Chapter {wiki_context['chapter_number']}"

    user_text = f"""Analyze page {page_number:03d} ({page_path.name}) of {chapter_ref}.

{context_text}

Extract ALL metadata following the JSON schema from the system prompt.
Be thorough: capture every text element, every character visible, every sound effect.
Use the exact character names from the character guide when possible."""

    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ],
        },
    ]

    return messages


# ============================================================
# STAGE 4: Extract Page Metadata via API
# ============================================================
def extract_page_metadata(client, page_path, page_number, config, character_registry, previous_context, kumiko_bboxes=None):
    """Call the OpenAI vision API and parse the JSON response."""
    messages = build_prompt(page_path, page_number, character_registry, previous_context, config, kumiko_bboxes=kumiko_bboxes)

    for attempt in range(config["retry_attempts"]):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content.strip()

            # Strip markdown fences if present despite json mode
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1]
                raw_text = raw_text.rsplit("```", 1)[0]

            data = json.loads(raw_text)
            validated = validate_page_json(data, page_number, kumiko_bboxes=kumiko_bboxes)

            # Track token usage
            usage = response.usage
            validated["_api_metadata"] = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "model": config["model"],
            }

            return validated

        except json.JSONDecodeError as e:
            print(f"    JSON parse error on attempt {attempt + 1}: {e}")
            if attempt < config["retry_attempts"] - 1:
                time.sleep(config["retry_delay_base"] ** (attempt + 1))

        except RateLimitError as e:
            wait = config["retry_delay_base"] ** (attempt + 2)
            print(f"    Rate limited (attempt {attempt + 1}): {e}")
            print(f"    Waiting {wait:.0f}s...")
            time.sleep(wait)

        except APIError as e:
            print(f"    API error on attempt {attempt + 1}: {e}")
            if attempt < config["retry_attempts"] - 1:
                time.sleep(config["retry_delay_base"] ** (attempt + 1))

    raise RuntimeError(f"Failed to extract metadata for page {page_number} after {config['retry_attempts']} attempts")


# ============================================================
# STAGE 5: Update Character Registry
# ============================================================
def update_character_registry(registry, page_metadata):
    """Merge newly discovered characters into the registry."""
    page_num = page_metadata.get("page_number", 0)
    characters = registry.setdefault("characters", {})

    for panel in page_metadata.get("panels", []):
        for char in panel.get("characters_present", []):
            cid = char.get("character_id", "")
            if not cid or cid == "crowd":
                continue

            if cid in characters:
                # Update appearances
                if page_num not in characters[cid].get("appearances", []):
                    characters[cid].setdefault("appearances", []).append(page_num)
            else:
                # New character
                characters[cid] = {
                    "character_id": cid,
                    "name": char.get("name", cid),
                    "visual_description": char.get("action", ""),
                    "first_appearance_page": page_num,
                    "appearances": [page_num],
                    "voice_profile": {},
                }

        # Also check text_elements for speakers
        for te in panel.get("text_elements", []):
            sid = te.get("speaker_id")
            if not sid or sid == "crowd":
                continue
            if sid in characters:
                if page_num not in characters[sid].get("appearances", []):
                    characters[sid].setdefault("appearances", []).append(page_num)

                # Enrich voice profile from tone_direction hints
                tone = te.get("tone_direction", "")
                emotion = te.get("emotion", "")
                vp = characters[sid].setdefault("voice_profile", {})
                if tone and "suggested_voice_style" not in vp:
                    vp["suggested_voice_style"] = tone
                if emotion:
                    vp.setdefault("observed_emotions", [])
                    if emotion not in vp["observed_emotions"]:
                        vp["observed_emotions"].append(emotion)

    return registry


# ============================================================
# STAGE 6: Save Outputs
# ============================================================
def save_outputs(page_metadata, page_number, registry, config):
    """Write per-page JSON and updated character registry."""
    output_dir = config["output_dir"]

    # Save page metadata (without internal _api_metadata)
    api_meta = page_metadata.pop("_api_metadata", None)

    page_path = output_dir / f"page_{page_number:03d}.json"
    with open(page_path, "w", encoding="utf-8") as f:
        json.dump(page_metadata, f, indent=2, ensure_ascii=False)

    # Restore for caller
    if api_meta:
        page_metadata["_api_metadata"] = api_meta

    # Save registry
    registry_path = output_dir / "character_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ============================================================
# STAGE 7: Build Chapter Manifest
# ============================================================
def build_manifest(config):
    """Aggregate all page JSONs into a chapter manifest."""
    output_dir = config["output_dir"]
    wiki_context = config.get("wiki_context")

    pages_index = []
    dialogue_index = []
    all_locations = set()
    total_dialog = 0
    total_narration = 0
    total_sfx = 0
    total_panels = 0
    dialogue_seq = 0

    for f in sorted(output_dir.glob("page_*.json")):
        page_num = int(f.stem.split("_")[1])

        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        num_panels = len(data.get("panels", []))
        num_dialog = 0
        num_narration = 0
        num_sfx = 0

        for panel in data.get("panels", []):
            panel_id = panel.get("panel_id", "")
            panel_location = panel.get("location", "")
            panel_time = panel.get("time_context", "")
            panel_image_path = panel.get("panel_image_path")
            depth_map_path = panel.get("depth_map_path")
            segmentation_map_path = panel.get("segmentation_map_path")

            # Collect locations for the locations vocabulary
            if panel_location:
                all_locations.add(panel_location)

            for te in panel.get("text_elements", []):
                if te.get("type") == "dialog":
                    num_dialog += 1
                    dialogue_seq += 1
                    entry = {
                        "seq": dialogue_seq,
                        "content": te.get("content", ""),
                        "speaker_id": te.get("speaker_id"),
                        "speaker": te.get("speaker"),
                        "emotion": te.get("emotion", ""),
                        "page": page_num,
                        "panel_id": panel_id,
                        "panel_image_path": panel_image_path,
                        "location": panel_location,
                        "time_context": panel_time,
                    }
                    if depth_map_path:
                        entry["depth_map_path"] = depth_map_path
                    if segmentation_map_path:
                        entry["segmentation_map_path"] = segmentation_map_path
                    dialogue_index.append(entry)
                elif te.get("type") in ("narration", "caption"):
                    num_narration += 1
            num_sfx += len(panel.get("sound_effects", []))

        total_panels += num_panels
        total_dialog += num_dialog
        total_narration += num_narration
        total_sfx += num_sfx

        pages_index.append({
            "page_number": page_num,
            "source_file": data.get("source_file", f""),
            "page_type": data.get("page_type", "story"),
            "metadata_file": f.name,
            "panels": num_panels,
            "dialog_lines": num_dialog,
            "narration_lines": num_narration,
            "sfx_count": num_sfx,
        })

    # Load character registry for count
    registry_path = output_dir / "character_registry.json"
    num_characters = 0
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
            num_characters = len(reg.get("characters", {}))

    # Chapter title: prefer wiki, fall back to "Unknown"
    chapter_title = "Unknown"
    if wiki_context and wiki_context.get("chapter_title"):
        chapter_title = wiki_context["chapter_title"]

    # Compute arc_id from wiki context
    arc_id = ""
    if wiki_context and wiki_context.get("arc"):
        arc_id = normalize_arc_id(wiki_context["arc"])

    manifest = {
        "chapter_id": config["chapter_id"],
        "chapter_title": chapter_title,
        "series": "One Piece",
        "arc_id": arc_id,
        "source_type": "colorized_english_scanlation",
        "total_pages": len(pages_index),
        "pages": pages_index,
        "dialogue_index": dialogue_index,
        "character_registry_file": "character_registry.json",
        "statistics": {
            "total_panels": total_panels,
            "total_dialog_lines": total_dialog,
            "total_narration_lines": total_narration,
            "total_sfx": total_sfx,
            "unique_characters": num_characters,
        },
        "locations": sorted(all_locations),
        "processing_metadata": {
            "model_used": config["model"],
            "processing_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_tokens_used": 0,
            "processing_time_seconds": 0,
            "wiki_grounded": wiki_context is not None,
        },
    }

    # Add arc info if available
    if wiki_context and wiki_context.get("arc"):
        manifest["arc"] = wiki_context["arc"]
        manifest["saga"] = wiki_context.get("saga", "")

    return manifest


# ============================================================
# CLI
# ============================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Manga Metadata Extraction Pipeline -- extracts structured "
                    "metadata from manga page images via GPT-4o vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python manga_metadata.py ./manga/0001 --chapter-id 0001
  python manga_metadata.py ./manga/0001 --chapter-id 0001 --resume
  python manga_metadata.py ./manga/0002 --chapter-id 0002 --no-wiki
  python manga_metadata.py ./manga/0001 --chapter-id 0001 --output-dir ./custom_output
""",
    )
    parser.add_argument(
        "chapter_dir",
        help="Path to directory containing chapter page images (e.g. ./manga/0001)",
    )
    parser.add_argument(
        "--chapter-id", required=True,
        help="Chapter ID, zero-padded (e.g. '0001', '0042')",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: output/manga_metadata/{chapter_id})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from cached pages (skip already-processed pages)",
    )
    parser.add_argument(
        "--no-wiki", action="store_true",
        help="Skip wiki context fetch (use generic prompt without character grounding)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Stop after processing N pages (useful for smoke testing)",
    )
    parser.add_argument(
        "--use-depth", action="store_true",
        help="Run depth estimation on panel crops (requires torch + MiDaS)",
    )
    parser.add_argument(
        "--use-segmentation", action="store_true",
        help="Run segmentation on panel crops (requires torchvision + DeepLabV3)",
    )
    parser.add_argument(
        "--no-kumiko", action="store_true",
        help="Disable Kumiko panel detection even if data is available",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    global CHARACTER_ALIASES

    args = parse_args()

    # Populate CONFIG from CLI args
    CONFIG["chapter_dir"] = Path(args.chapter_dir)
    CONFIG["chapter_id"] = args.chapter_id
    CONFIG["output_dir"] = Path(
        args.output_dir or f"output/manga_metadata/{args.chapter_id}"
    )
    CONFIG["resume"] = args.resume

    print("=" * 60)
    print("Manga Metadata Extraction Pipeline")
    print(f"Chapter: {CONFIG['chapter_id']}")
    print("=" * 60)

    # Fetch wiki context
    wiki_context = None
    if not args.no_wiki:
        print("\nFetching wiki context...")
        wiki_context = fetch_wiki_chapter_context(args.chapter_id)
        if wiki_context:
            title = wiki_context.get("chapter_title", "?")
            num_chars = len(wiki_context.get("characters", []))
            arc = wiki_context.get("arc", "")
            print(f"  Chapter: \"{title}\"")
            print(f"  Arc: {arc}")
            print(f"  Characters: {num_chars}")
        elif HAS_MANGA_WIKI_API:
            print("  Wiki context not available for this chapter")
        else:
            print("  manga_wiki_api not installed, skipping wiki context")

    CONFIG["wiki_context"] = wiki_context

    # Load Kumiko panel detection data
    kumiko_data = None
    if not args.no_kumiko:
        kumiko_data = load_kumiko_chapter_data(args.chapter_id)
        if kumiko_data:
            print(f"\n  Kumiko panels: {len(kumiko_data.get('pages', {}))} pages")
    CONFIG["kumiko_data"] = kumiko_data

    # Build dynamic system prompt and character aliases
    CONFIG["system_prompt"] = build_system_prompt(wiki_context)
    CHARACTER_ALIASES = build_character_aliases(wiki_context)

    # Validate API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set!\n"
            "  Set it with: set OPENAI_API_KEY=your-key-here"
        )

    # Load GPU models if requested
    gpu_models = None
    if args.use_depth or args.use_segmentation:
        gpu_models = load_manga_gpu_models(
            use_depth=args.use_depth,
            use_segmentation=args.use_segmentation,
        )
        if not gpu_models or (not gpu_models.get("depth_model") and not gpu_models.get("seg_model")):
            print("Warning: GPU models requested but none loaded -- continuing without them")
            gpu_models = None

    client = OpenAI()
    start_time = time.time()
    total_tokens = 0

    # Stage 1: Discover pages
    print("\n[1/4] Discovering pages...")
    pages = discover_pages(CONFIG)
    CONFIG["total_pages"] = len(pages)
    print(f"  Found {len(pages)} pages ({pages[0][1].name} to {pages[-1][1].name})")

    # Stage 2: Load progress
    print("\n[2/4] Loading progress...")
    completed, registry, page_contexts = load_progress(CONFIG)
    if completed:
        print(f"  Already processed: {len(completed)} pages")

    remaining = [(num, path) for num, path in pages if num not in completed]
    if CONFIG["resume"] and not remaining:
        print("  All pages already processed! Building manifest...")
    elif CONFIG["resume"]:
        print(f"  Remaining: {len(remaining)} pages")
    else:
        remaining = pages  # Re-process all if not resuming
        print(f"  Processing all {len(remaining)} pages")

    # Limit pages for smoke testing
    if args.max_pages and len(remaining) > args.max_pages:
        remaining = remaining[:args.max_pages]
        print(f"  --max-pages: limited to first {args.max_pages} pages")

    # Stage 3-6: Process each page
    if remaining:
        print(f"\n[3/4] Extracting metadata ({len(remaining)} pages)...")

        for i, (page_num, page_path) in enumerate(remaining):
            print(f"\n  [{i + 1}/{len(remaining)}] Page {page_num:03d} ({page_path.name})...")

            # Build context from previous pages
            prev_context = {
                pn: ctx for pn, ctx in page_contexts.items()
                if page_num - CONFIG["context_window"] <= pn < page_num
            }

            # Look up Kumiko bboxes for this page
            kumiko_bboxes = get_kumiko_bboxes(CONFIG["kumiko_data"], page_path.name)
            if kumiko_bboxes:
                print(f"    Kumiko: {len(kumiko_bboxes)} panels")

            # Extract via API
            metadata = extract_page_metadata(
                client, page_path, page_num, CONFIG, registry, prev_context,
                kumiko_bboxes=kumiko_bboxes,
            )

            page_type = metadata.get("page_type", "story")
            num_panels = len(metadata.get("panels", []))
            num_text = sum(
                len(p.get("text_elements", [])) for p in metadata.get("panels", [])
            )
            num_sfx = sum(
                len(p.get("sound_effects", [])) for p in metadata.get("panels", [])
            )
            print(f"    Type: {page_type} | Panels: {num_panels} | Text elements: {num_text} | SFX: {num_sfx}")

            # Update registry
            registry = update_character_registry(registry, metadata)

            # Store context
            page_contexts[page_num] = summarize_page_for_context(metadata)

            # Crop panels
            panels_dir = CONFIG["output_dir"] / "panels"
            num_cropped = crop_and_save_panels(metadata, page_path, panels_dir)
            if num_cropped > 0:
                print(f"    Panel crops: {num_cropped}/{num_panels}")

            # Run GPU models on panel crops (depth + segmentation)
            if gpu_models and num_cropped > 0:
                depth_count = 0
                seg_count = 0
                for panel in metadata.get("panels", []):
                    d_ok, s_ok = process_panel_gpu_models(
                        panel, panels_dir, gpu_models, resume=CONFIG["resume"]
                    )
                    if d_ok:
                        depth_count += 1
                    if s_ok:
                        seg_count += 1
                parts = []
                if depth_count:
                    parts.append(f"{depth_count} depth")
                if seg_count:
                    parts.append(f"{seg_count} seg")
                if parts:
                    print(f"    GPU models: {', '.join(parts)}")

            # Save (after cropping so panel_image_path is included in the JSON)
            api_meta = metadata.get("_api_metadata", {})
            total_tokens += api_meta.get("input_tokens", 0) + api_meta.get("output_tokens", 0)
            save_outputs(metadata, page_num, registry, CONFIG)

            # Rate limiting
            if i < len(remaining) - 1:
                time.sleep(CONFIG["delay_between_requests"])

    # Stage 7: Build manifest
    print("\n[4/4] Building chapter manifest...")
    manifest = build_manifest(CONFIG)
    manifest["processing_metadata"]["total_tokens_used"] = total_tokens
    manifest["processing_metadata"]["processing_time_seconds"] = round(time.time() - start_time, 1)

    manifest_path = CONFIG["output_dir"] / "chapter_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary
    stats = manifest["statistics"]
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed / 60:.1f} minutes")
    print(f"Output: {CONFIG['output_dir']}")
    print(f"Pages processed: {manifest['total_pages']}")
    print(f"Panels: {stats['total_panels']}")
    print(f"Dialog lines: {stats['total_dialog_lines']}")
    print(f"Narration lines: {stats['total_narration_lines']}")
    print(f"Sound effects: {stats['total_sfx']}")
    print(f"Characters found: {stats['unique_characters']}")
    print(f"API tokens used: {total_tokens:,}")
    wiki_status = "yes" if wiki_context else "no"
    print(f"Wiki grounded: {wiki_status}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
