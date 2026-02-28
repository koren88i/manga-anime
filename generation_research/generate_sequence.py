import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import time
import base64
import shutil
import argparse
import subprocess
from pathlib import Path

# Load .env from project root (if python-dotenv is installed)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ============================================================
# CONFIG
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Wan2GP valid frame counts (must be 4n+1)
VALID_FRAME_COUNTS = [17, 33, 49, 65, 81, 97, 113, 129]

# Model type -> architecture mapping for settings JSON
MODEL_ARCHITECTURES = {
    "flf2v": "flf2v_720p",
    "i2v":   "i2v",
    "fun_inp_1.3B": "fun_inp_1.3B",
}

# Default resolutions per model type
DEFAULT_RESOLUTIONS = {
    "flf2v": "1280x720",
    "i2v":   "832x480",
    "fun_inp_1.3B": "832x480",
}

# Resolution presets
RESOLUTION_PRESETS = {
    "480p": "832x480",
    "720p": "1280x720",
}


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def discover_images(images_folder):
    """Find all images in folder, sorted by name.

    Supports .jpg, .jpeg, .png, .webp, .bmp.
    Returns list of Path objects sorted alphabetically.
    """
    folder = Path(images_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    images = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name,
    )

    if len(images) < 2:
        raise ValueError(f"Need at least 2 images, found {len(images)} in {images_folder}")

    return images


# ============================================================
# AI PROMPT GENERATION (--aiprompt)
# ============================================================

# Backend-specific model name for prompt context
_BACKEND_NAMES = {
    "wan2gp": "Wan2GP",
    "kling":  "Kling AI",
}


def encode_image(image_path):
    """Read and base64-encode an image file. Returns (base64_str, media_type)."""
    ext = Path(image_path).suffix.lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, media_type


def generate_ai_prompt(image_path, base_prompt, client, backend="wan2gp"):
    """Send image + simple prompt to GPT-4o to generate a detailed video prompt.

    Returns a detailed prompt string suitable for video generation.
    """
    image_b64, media_type = encode_image(image_path)
    model_name = _BACKEND_NAMES.get(backend, backend)

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_b64}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": (
                    f'You are writing a prompt for an AI video generation model ({model_name}).\n\n'
                    f'The user wants: "{base_prompt}"\n\n'
                    f'Based on this image, write a detailed video generation prompt that describes:\n'
                    f'1. The visual scene, characters, and setting visible in the image\n'
                    f'2. The style (anime, manga, specific art style cues)\n'
                    f'3. Camera motion that would enhance the scene (slow pan, zoom, static, etc.)\n'
                    f'4. Character motion or action that should occur\n'
                    f'5. Mood, lighting, and atmosphere\n\n'
                    f'Write the prompt as a single paragraph, 2-4 sentences. Be specific and cinematic.\n'
                    f'Do NOT include technical parameters (resolution, fps, steps).\n'
                    f'Return ONLY the prompt text, nothing else.'
                ),
            },
        ],
    }]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def estimate_duration(prompt, client):
    """Ask GPT-4o to suggest a frame count based on the prompt content.

    Returns an integer frame count from VALID_FRAME_COUNTS.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                f'Based on this video prompt, suggest how many frames the video should be.\n'
                f'The video runs at ~24fps. Choose from these frame counts:\n'
                f'- 17 frames (~0.7 seconds) -- very quick cut or transition\n'
                f'- 33 frames (~1.4 seconds) -- short action or reaction\n'
                f'- 49 frames (~2 seconds) -- standard scene\n'
                f'- 65 frames (~2.7 seconds) -- longer action or dialogue\n'
                f'- 81 frames (~3.4 seconds) -- extended scene\n\n'
                f'Prompt: "{prompt}"\n\n'
                f'Return ONLY the number (17, 33, 49, 65, or 81), nothing else.'
            ),
        }],
        max_tokens=10,
        temperature=0.3,
    )

    try:
        frames = int(response.choices[0].message.content.strip())
        if frames not in VALID_FRAME_COUNTS:
            frames = min(VALID_FRAME_COUNTS[:5], key=lambda x: abs(x - frames))
        return frames
    except (ValueError, IndexError):
        return 49  # default


# ============================================================
# WAN2GP SETTINGS CONSTRUCTION
# ============================================================

def load_default_settings():
    """Load the default _settings.json from the Wan2GP installation."""
    return {
        "prompt": "",
        "resolution": "832x480",
        "video_length": 81,
        "num_inference_steps": 30,
        "seed": -1,
        "guidance_scale": 5,
        "image_start": None,
        "image_end": None,
        "negative_prompt": "",
        "output_filename": "",
    }


def build_wan2gp_settings(template_settings, prompt, first_frame_path,
                          last_frame_path, output_filename, frames,
                          resolution, model_type, seed, guidance_scale,
                          steps):
    """Build a settings JSON for one Wan2GP generation job.

    Args:
        template_settings: Dict loaded from exported Wan2GP settings JSON (or defaults)
        prompt: Text prompt for this clip
        first_frame_path: Absolute path to first frame image
        last_frame_path: Absolute path to last frame image (None for I2V-only)
        output_filename: Filename for the output video (without path)
        frames: Frame count (must be 4n+1)
        resolution: Resolution string like "832x480"
        model_type: "flf2v" or "i2v"
        seed: Random seed
        guidance_scale: Guidance scale
        steps: Inference steps

    Returns:
        Dict ready to be written as settings JSON.
    """
    settings = dict(template_settings)

    # Core generation params
    settings["prompt"] = prompt
    settings["video_length"] = frames
    settings["num_inference_steps"] = steps
    settings["seed"] = seed
    settings["guidance_scale"] = guidance_scale
    settings["resolution"] = resolution
    settings["output_filename"] = output_filename

    # Model type
    settings["model_type"] = MODEL_ARCHITECTURES.get(model_type, model_type)

    # Image inputs
    settings["image_start"] = str(first_frame_path) if first_frame_path else None
    settings["image_end"] = str(last_frame_path) if last_frame_path else None

    return settings


# ============================================================
# KLING AI API CLIENT
# ============================================================

KLING_API_BASE_URL = "https://api.klingai.com"
KLING_CREATE_TASK = "/v1/videos/image2video"
KLING_QUERY_TASK = "/v1/videos/image2video/{task_id}"

KLING_DEFAULT_MODEL = "kling-v2-6"
KLING_DEFAULT_DURATION = "5"
KLING_DEFAULT_MODE = "std"
KLING_DEFAULT_POLL_INTERVAL = 10
KLING_DEFAULT_TIMEOUT = 600
KLING_TOKEN_EXPIRE = 1800

KLING_VALID_DURATIONS = ["5", "10"]
KLING_VALID_MODELS = ["kling-v1-5", "kling-v1-6", "kling-v2-6"]


class KlingClient:
    """Kling AI video generation API client.

    Handles JWT authentication, task creation, status polling,
    and video download. Credentials from env vars KLING_ACCESS_KEY
    and KLING_SECRET_KEY.
    """

    def __init__(self, access_key=None, secret_key=None):
        self.access_key = access_key or os.environ.get("KLING_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("KLING_SECRET_KEY")
        if not self.access_key or not self.secret_key:
            raise RuntimeError(
                "Kling API keys not configured.\n"
                "Set environment variables:\n"
                "  KLING_ACCESS_KEY=your-access-key\n"
                "  KLING_SECRET_KEY=your-secret-key\n"
                "Get keys at: https://app.klingai.com/global/dev/api-key"
            )
        print(f"  Kling API client initialized (key: {self.access_key[:8]}...)")

    def _generate_jwt_token(self):
        import jwt
        now = int(time.time())
        headers = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": self.access_key,
            "exp": now + KLING_TOKEN_EXPIRE,
            "nbf": now - 5,
        }
        return jwt.encode(payload, self.secret_key, headers=headers)

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self._generate_jwt_token()}",
            "Content-Type": "application/json",
        }

    def _check_response(self, response, action):
        if response.status_code != 200:
            raise RuntimeError(
                f"Kling API {action} failed (HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(
                f"Kling API {action} failed (code {result.get('code')}): "
                f"{result.get('message', 'unknown error')}"
            )
        return result

    def create_task(self, image_start_b64, image_end_b64=None, prompt="",
                    model_name=KLING_DEFAULT_MODEL, duration=KLING_DEFAULT_DURATION,
                    mode=KLING_DEFAULT_MODE, negative_prompt=""):
        """Submit image-to-video generation task. Returns dict with task_id."""
        import requests as req

        body = {
            "model_name": model_name,
            "image": image_start_b64,
            "duration": duration,
            "mode": mode,
        }
        if image_end_b64:
            body["image_tail"] = image_end_b64
        if prompt:
            body["prompt"] = prompt
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if not model_name.startswith("kling-v2"):
            body["cfg_scale"] = 0.5

        url = f"{KLING_API_BASE_URL}{KLING_CREATE_TASK}"
        resp = req.post(url, json=body, headers=self._get_headers(), timeout=30)
        result = self._check_response(resp, "create task")
        task_data = result["data"]
        print(f"    Task created: {task_data['task_id']} (status: {task_data['task_status']})")
        return task_data

    def query_task(self, task_id):
        """Query task status. Returns task data dict."""
        import requests as req

        url = f"{KLING_API_BASE_URL}{KLING_QUERY_TASK.format(task_id=task_id)}"
        resp = req.get(url, headers=self._get_headers(), timeout=30)
        result = self._check_response(resp, "query task")
        return result["data"]

    def wait_for_task(self, task_id, timeout=KLING_DEFAULT_TIMEOUT,
                      poll_interval=KLING_DEFAULT_POLL_INTERVAL):
        """Poll until task completes. Returns task data with video URLs."""
        start = time.time()
        while True:
            elapsed = int(time.time() - start)
            if elapsed > timeout:
                raise TimeoutError(f"Kling task {task_id} timed out after {elapsed}s")

            task_data = self.query_task(task_id)
            status = task_data["task_status"]

            if status == "succeed":
                print(f"    Task completed in {format_elapsed(elapsed)}")
                return task_data
            if status == "failed":
                msg = task_data.get("task_status_msg", "unknown error")
                raise RuntimeError(f"Kling task failed: {msg}")

            print(f"    [{format_elapsed(elapsed)}] Status: {status}...")
            time.sleep(poll_interval)

    def download_video(self, video_url, save_path):
        """Download generated video. Returns Path to saved file."""
        import requests as req

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        resp = req.get(video_url, stream=True, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Download failed (HTTP {resp.status_code})")

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"    Downloaded: {save_path.name} ({size_mb:.1f} MB)")
        return save_path


def run_kling_job(kling_client, img_a, img_b, prompt, clip_path,
                  model, duration, mode, negative_prompt, i2v_only,
                  timeout, poll_interval):
    """Run a single Kling image-to-video job.

    Encodes images, submits task, polls for completion, downloads result.
    Returns Path to generated video, or None on failure.
    """
    # Encode start frame as base64 (raw, no data: prefix)
    with open(img_a, "rb") as f:
        start_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Encode end frame (unless i2v-only mode)
    end_b64 = None
    if not i2v_only and img_b:
        with open(img_b, "rb") as f:
            end_b64 = base64.b64encode(f.read()).decode("utf-8")

    video_type = "I2V (single frame)" if i2v_only else "FLF (first+last frame)"
    print(f"    Kling {video_type}, model={model}, {duration}s, mode={mode}")

    try:
        task_data = kling_client.create_task(
            image_start_b64=start_b64,
            image_end_b64=end_b64,
            prompt=prompt,
            model_name=model,
            duration=duration,
            mode=mode,
            negative_prompt=negative_prompt,
        )

        result_data = kling_client.wait_for_task(
            task_data["task_id"],
            timeout=timeout,
            poll_interval=poll_interval,
        )

        videos = result_data.get("task_result", {}).get("videos", [])
        if not videos:
            print("    ERROR: Task completed but no video returned")
            return None

        video_url = videos[0]["url"]
        return kling_client.download_video(video_url, clip_path)

    except (RuntimeError, TimeoutError) as e:
        print(f"    ERROR: {e}")
        return None
    except Exception as e:
        print(f"    ERROR: Unexpected error: {e}")
        return None


def estimate_kling_duration(prompt, client):
    """Ask GPT-4o to suggest 5 or 10 second duration for a Kling clip.

    Returns string "5" or "10".
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                f'Based on this video prompt, suggest the ideal clip duration.\n'
                f'Choose either 5 or 10 seconds:\n'
                f'- 5 seconds -- quick action, reaction, transition, or simple motion\n'
                f'- 10 seconds -- extended action, dialogue scene, complex motion, or slow pan\n\n'
                f'Prompt: "{prompt}"\n\n'
                f'Return ONLY the number (5 or 10), nothing else.'
            ),
        }],
        max_tokens=5,
        temperature=0.3,
    )

    try:
        dur = response.choices[0].message.content.strip()
        if dur in KLING_VALID_DURATIONS:
            return dur
    except (ValueError, IndexError):
        pass
    return KLING_DEFAULT_DURATION


# ============================================================
# WAN2GP INVOCATION
# ============================================================

def find_wan2gp_dir(explicit_path=None):
    """Locate the Wan2GP installation directory.

    Checks: explicit path, then sibling wan2gp/ directory.
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.is_dir() and (p / "wgp.py").exists():
            return p
        raise FileNotFoundError(f"Wan2GP not found at: {explicit_path} (missing wgp.py)")

    # Sibling directory: helloworld_claude/wan2gp/
    project_root = Path(__file__).resolve().parent.parent
    sibling = project_root.parent / "wan2gp"
    if sibling.is_dir() and (sibling / "wgp.py").exists():
        return sibling

    raise FileNotFoundError(
        "Wan2GP installation not found. Expected at: "
        f"{sibling}\n"
        "Use --wan2gp-dir to specify the location, or clone it:\n"
        "  git clone https://github.com/deepbeepmeep/Wan2GP.git wan2gp"
    )


def run_wan2gp_job(settings_path, output_dir, wan2gp_dir, conda_env,
                   profile, attention, timeout=1800):
    """Run a single Wan2GP generation job via subprocess.

    Uses start_wan2gp.bat for reliable conda activation on Windows.

    Args:
        settings_path: Path to the settings JSON file
        output_dir: Directory for output video
        wan2gp_dir: Path to Wan2GP installation
        conda_env: Name of conda environment
        profile: Memory profile (1-5)
        attention: Attention mode (sdpa, sage, flash, sage2)
        timeout: Max seconds to wait (default: 30 minutes)

    Returns:
        Path to generated video file, or None on failure.
    """
    wan2gp_dir = Path(wan2gp_dir)
    bat_path = wan2gp_dir / "start_wan2gp.bat"

    # CUDA_LAUNCH_BLOCKING=1 is required on RTX 20XX (Turing) to prevent
    # "illegal memory access" crashes during async CUDA operations in mmgp
    env = os.environ.copy()
    env["CUDA_LAUNCH_BLOCKING"] = "1"

    # Resolve all paths to absolute (bat file changes cwd to wan2gp dir)
    settings_path = Path(settings_path).resolve()
    output_dir = Path(output_dir).resolve()

    if bat_path.exists():
        # Use the batch launcher (handles venv activation + defaults)
        cmd = [
            str(bat_path),
            "--process", str(settings_path),
            "--output-dir", str(output_dir),
        ]
    else:
        # Fallback: direct venv python call
        venv_python = wan2gp_dir / "venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = "python"
        cmd = [
            str(venv_python),
            str(wan2gp_dir / "wgp.py"),
            "--process", str(settings_path),
            "--output-dir", str(output_dir),
            "--profile", str(profile),
            "--attention", attention,
            "--fp16",
        ]

    print(f"    Command: {' '.join(str(c) for c in cmd[:6])}...")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            wait = 10 * attempt
            print(f"    Retry {attempt}/{max_retries} after {wait}s cooldown...")
            time.sleep(wait)

        try:
            # Track files before run to detect new outputs
            output_path_dir = Path(output_dir)
            existing_mp4s = set(output_path_dir.glob("*.mp4"))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(wan2gp_dir),
                env=env,
            )

            if result.returncode != 0:
                stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
                print(f"    ERROR: Wan2GP exited with code {result.returncode} (attempt {attempt})")
                if "CUDA error" in stderr_tail or "illegal memory" in stderr_tail:
                    print(f"    CUDA error detected, will retry...")
                    continue
                print(f"    stderr: {stderr_tail}")
                if attempt < max_retries:
                    continue
                return None

            # Find NEW video files (not existing before this run)
            new_mp4s = set(output_path_dir.glob("*.mp4")) - existing_mp4s
            if new_mp4s:
                return sorted(new_mp4s, key=lambda p: p.stat().st_mtime)[-1]

            # Fallback: most recently modified mp4
            all_mp4s = sorted(output_path_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
            if all_mp4s:
                return all_mp4s[-1]

            print("    WARNING: Wan2GP reported success but no .mp4 found")
            return None

        except subprocess.TimeoutExpired:
            print(f"    ERROR: Wan2GP timed out after {timeout}s")
            if attempt < max_retries:
                continue
            return None
        except FileNotFoundError as e:
            print(f"    ERROR: Could not run Wan2GP: {e}")
            print("    Make sure the wan2gp venv exists.")
            return None

    return None


# ============================================================
# FFMPEG CONCATENATION
# ============================================================

def concatenate_clips(clip_paths, output_path):
    """Concatenate video clips into a single video using ffmpeg concat demuxer.

    Returns Path to output video, or None on failure.
    """
    if not clip_paths:
        print("  No clips to concatenate")
        return None

    output_path = Path(output_path)

    # Build concat list file
    list_path = output_path.parent / "concat_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for clip in clip_paths:
            # ffmpeg requires forward slashes
            escaped = str(clip.resolve()).replace("\\", "/")
            f.write(f"file '{escaped}'\n")

    # Try stream copy first (fast, no re-encoding)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(output_path),
    ]

    print(f"  Concatenating {len(clip_paths)} clips -> {output_path.name}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print("  Stream copy failed, re-encoding with libx264...")
            cmd_reencode = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                str(output_path),
            ]
            result = subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"  ERROR: ffmpeg re-encode failed: {result.stderr[-300:]}")
                return None

        # Clean up list file
        list_path.unlink(missing_ok=True)
        return output_path

    except subprocess.TimeoutExpired:
        print("  ERROR: ffmpeg concat timed out")
        return None


# ============================================================
# PROGRESS DISPLAY
# ============================================================

def format_elapsed(seconds):
    """Format seconds as Xm Ys."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def print_progress_bar(current, total, width=30):
    """Print an ASCII progress bar (Windows cp1252 safe)."""
    filled = int(width * current / total) if total > 0 else 0
    bar = "#" * filled + "-" * (width - filled)
    pct = (current / total * 100) if total > 0 else 0
    print(f"  [{bar}] {current}/{total} ({pct:.0f}%)")


# ============================================================
# CLI ARGUMENT PARSING
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video sequence from consecutive image pairs (Wan2GP or Kling AI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Wan2GP (local GPU, default)
  python generation_research/generate_sequence.py --images-folder ./manga/0001/panels --prompt "anime style scene"
  python generation_research/generate_sequence.py --images-folder ./frames --prompt "smooth transition" --aiprompt

  # Kling AI (cloud API)
  python generation_research/generate_sequence.py --backend kling --images-folder ./frames --prompt "anime scene"
  python generation_research/generate_sequence.py --backend kling --images-folder ./frames --prompt "action" --aiprompt --mode pro --duration 10
""",
    )

    # ---- Shared arguments ----
    parser.add_argument(
        "--images-folder", required=True,
        help="Path to folder containing input images (sorted alphabetically)",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Base prompt describing desired video style/content",
    )
    parser.add_argument(
        "--output-folder", default=None,
        help="Output directory (default: output/generation/<timestamp>)",
    )
    parser.add_argument(
        "--aiprompt", action="store_true",
        help="Use GPT-4o to generate detailed AI prompts from each image + base prompt",
    )
    # Hidden backward compat alias for --flag
    parser.add_argument("--flag", action="store_true", dest="aiprompt", help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-generated clips (resume interrupted run)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Timeout per clip in seconds (default: 1800 wan2gp, 600 kling)",
    )
    parser.add_argument(
        "--backend", default="wan2gp", choices=["wan2gp", "kling"],
        help="Video generation backend (default: wan2gp)",
    )

    # ---- Wan2GP-specific arguments ----
    wan2gp_group = parser.add_argument_group("Wan2GP options (--backend wan2gp)")
    wan2gp_group.add_argument(
        "--resolution", choices=["480p", "720p"], default="480p",
        help="Video resolution (default: 480p, recommended for 11GB VRAM)",
    )
    wan2gp_group.add_argument(
        "--model-type", default="fun_inp_1.3B",
        choices=["fun_inp_1.3B", "i2v", "flf2v"],
        help=(
            "Wan2GP model type (default: fun_inp_1.3B). "
            "fun_inp_1.3B = Fun InP 1.3B (start+end frames, fits 11GB VRAM). "
            "i2v = Image-to-Video 480p 14B (needs 24GB+). "
            "flf2v = First-Last-Frame 720p 14B (needs 24GB+)"
        ),
    )
    wan2gp_group.add_argument(
        "--frames", type=int, default=49,
        help="Number of frames per clip (default: 49 = ~2s at 24fps). Must be 4n+1.",
    )
    wan2gp_group.add_argument(
        "--steps", type=int, default=30,
        help="Number of inference steps (default: 30)",
    )
    wan2gp_group.add_argument(
        "--profile", type=int, default=4, choices=[1, 2, 3, 4, 5],
        help="Wan2GP memory profile (default: 4, load parts on demand)",
    )
    wan2gp_group.add_argument(
        "--attention", default="sdpa", choices=["sdpa", "sage", "flash", "sage2"],
        help="Attention mode (default: sdpa, safest for RTX 20XX)",
    )
    wan2gp_group.add_argument(
        "--wan2gp-dir", default=None,
        help="Path to Wan2GP installation (default: sibling wan2gp/ directory)",
    )
    wan2gp_group.add_argument(
        "--wan2gp-env", default="wan2gp",
        help="Conda environment name for Wan2GP (default: wan2gp)",
    )
    wan2gp_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42, -1 for random)",
    )
    wan2gp_group.add_argument(
        "--guidance-scale", type=float, default=5.0,
        help="Guidance scale (default: 5.0)",
    )
    wan2gp_group.add_argument(
        "--settings-template", default=None,
        help="Path to exported Wan2GP settings JSON to use as base template",
    )

    # ---- Kling-specific arguments ----
    kling_group = parser.add_argument_group("Kling options (--backend kling)")
    kling_group.add_argument(
        "--model", default=KLING_DEFAULT_MODEL,
        help=f"Kling model name (default: {KLING_DEFAULT_MODEL})",
    )
    kling_group.add_argument(
        "--duration", default=KLING_DEFAULT_DURATION, choices=KLING_VALID_DURATIONS,
        help=f"Clip duration in seconds (default: {KLING_DEFAULT_DURATION})",
    )
    kling_group.add_argument(
        "--mode", default=KLING_DEFAULT_MODE, choices=["std", "pro"],
        help=f"Generation mode: std (fast) or pro (higher quality) (default: {KLING_DEFAULT_MODE})",
    )
    kling_group.add_argument(
        "--negative-prompt", default="",
        help="Negative prompt (elements to avoid in generation)",
    )
    kling_group.add_argument(
        "--i2v-only", action="store_true",
        help="Use only start frame (no end frame), single-image I2V",
    )
    kling_group.add_argument(
        "--poll-interval", type=int, default=KLING_DEFAULT_POLL_INTERVAL,
        help=f"Seconds between status polls (default: {KLING_DEFAULT_POLL_INTERVAL})",
    )

    args = parser.parse_args()

    # Set backend-appropriate default timeout
    if args.timeout is None:
        args.timeout = 600 if args.backend == "kling" else 1800

    # Validate frame count (Wan2GP only)
    if args.backend == "wan2gp" and args.frames not in VALID_FRAME_COUNTS:
        closest = min(VALID_FRAME_COUNTS, key=lambda x: abs(x - args.frames))
        print(f"  WARNING: --frames {args.frames} is not valid (must be 4n+1). Using {closest}.")
        args.frames = closest

    return args


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def main():
    args = parse_args()
    run_start = time.time()
    is_kling = args.backend == "kling"

    # Resolve paths
    images_folder = Path(args.images_folder)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder = Path(args.output_folder or f"output/generation/{timestamp}")

    # ---- Backend-specific init ----
    wan2gp_dir = None
    resolution = None
    kling_client = None

    if is_kling:
        # Validate Kling dependencies
        try:
            import jwt      # noqa: F401
            import requests  # noqa: F401
        except ImportError as e:
            print(f"  ERROR: Missing dependency for Kling backend: {e}")
            print("  Install with: pip install pyjwt requests")
            return
        kling_client = KlingClient()
    else:
        wan2gp_dir = find_wan2gp_dir(args.wan2gp_dir)
        resolution = RESOLUTION_PRESETS.get(args.resolution, args.resolution)
        if args.model_type == "flf2v" and args.resolution != "720p":
            print("  NOTE: FLF2V model is 720p only. Overriding resolution to 720p.")
            resolution = "1280x720"
        if args.model_type in ("i2v", "flf2v"):
            print("  WARNING: 14B models need 24GB+ VRAM. Use fun_inp_1.3B for 11GB GPUs.")

    # Create output directories
    output_folder.mkdir(parents=True, exist_ok=True)
    clips_dir = output_folder / "clips"
    clips_dir.mkdir(exist_ok=True)
    settings_dir = output_folder / "settings"
    settings_dir.mkdir(exist_ok=True)

    # ---- Print config banner ----
    print("=" * 60)
    print("  Image Sequence Video Generation Pipeline")
    print("=" * 60)
    print(f"  Backend:     {args.backend}")
    print(f"  Images:      {images_folder}")
    if is_kling:
        print(f"  Model:       {args.model}")
        print(f"  Duration:    {args.duration}s")
        print(f"  Mode:        {args.mode}")
        flt_mode = "I2V (single frame)" if args.i2v_only else "FLF (first+last frame)"
        print(f"  Frame mode:  {flt_mode}")
        if args.negative_prompt:
            print(f"  Neg. prompt: {args.negative_prompt}")
    else:
        print(f"  Model:       {args.model_type} ({MODEL_ARCHITECTURES[args.model_type]})")
        print(f"  Resolution:  {resolution}")
        print(f"  Frames:      {args.frames} (~{args.frames / 24:.1f}s at 24fps)")
        print(f"  Steps:       {args.steps}")
        print(f"  Profile:     {args.profile}")
        print(f"  Attention:   {args.attention}")
        print(f"  Seed:        {args.seed}")
        print(f"  Wan2GP:      {wan2gp_dir}")
    print(f"  AI Prompt:   {'YES (GPT-4o)' if args.aiprompt else 'no'}")
    print(f"  Timeout:     {args.timeout}s")
    print(f"  Output:      {output_folder}")
    print(f"  Resume:      {'yes' if args.resume else 'no'}")
    print("=" * 60)

    # ---- [1/4] Discover images ----
    print("\n[1/4] Discovering images...")
    images = discover_images(images_folder)
    pairs = [(images[i], images[i + 1]) for i in range(len(images) - 1)]
    print(f"  Found {len(images)} images -> {len(pairs)} consecutive pairs")
    for i, img in enumerate(images):
        print(f"    {i + 1}. {img.name}")

    # ---- Setup OpenAI client if --aiprompt ----
    openai_client = None
    if args.aiprompt:
        try:
            from openai import OpenAI
            openai_client = OpenAI()
            print("  OpenAI client initialized for AI prompting")
        except ImportError:
            print("  ERROR: --aiprompt requires 'openai' package. pip install openai")
            return
        except Exception as e:
            print(f"  ERROR: Failed to initialize OpenAI client: {e}")
            print("  Make sure OPENAI_API_KEY is set.")
            return

    # ---- Load settings template (Wan2GP only) ----
    template = None
    if not is_kling:
        if args.settings_template:
            tpl_path = Path(args.settings_template)
            if not tpl_path.exists():
                print(f"  ERROR: Settings template not found: {tpl_path}")
                return
            with open(tpl_path, encoding="utf-8") as f:
                template = json.load(f)
            print(f"  Loaded settings template: {tpl_path.name}")
        else:
            template = load_default_settings()
            print("  Using default settings (no template provided)")
            print("  TIP: Export a settings JSON from the Wan2GP UI for best results.")
            print("       Use --settings-template <path> to load it.")

    # ---- [2/4] Generate clips ----
    print(f"\n[2/4] Generating {len(pairs)} video clips...")
    clip_paths = []
    failed_pairs = []

    for idx, (img_a, img_b) in enumerate(pairs):
        clip_name = f"clip_{idx + 1:03d}_{img_a.stem}-{img_b.stem}.mp4"
        clip_path = clips_dir / clip_name

        # Resume: skip existing clips
        if args.resume and clip_path.exists() and clip_path.stat().st_size > 0:
            print(f"\n  [{idx + 1}/{len(pairs)}] {clip_name} -- exists, skipping")
            clip_paths.append(clip_path)
            continue

        print(f"\n  [{idx + 1}/{len(pairs)}] {img_a.name} -> {img_b.name}")
        print_progress_bar(idx, len(pairs))

        # Generate prompt via GPT-4o or use base prompt
        prompt = args.prompt
        frames = args.frames
        duration = args.duration if is_kling else None

        if args.aiprompt and openai_client:
            print("    Generating AI prompt via GPT-4o...")
            try:
                prompt = generate_ai_prompt(img_a, args.prompt, openai_client,
                                            backend=args.backend)
                print(f"    [AI Prompt] {prompt}")
                if is_kling:
                    duration = estimate_kling_duration(prompt, openai_client)
                    print(f"    [AI Duration] {duration}s")
                else:
                    frames = estimate_duration(prompt, openai_client)
                    print(f"    [AI Duration] {frames} frames ({frames / 24:.1f}s)")
            except Exception as e:
                print(f"    WARNING: AI prompt failed ({e}), using base prompt")
                prompt = args.prompt
                frames = args.frames
                duration = args.duration if is_kling else None

        clip_start = time.time()

        if is_kling:
            # ---- Kling backend ----
            # Save job metadata (without base64 images)
            job_meta = {
                "backend": "kling",
                "model": args.model,
                "duration": duration,
                "mode": args.mode,
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
                "i2v_only": args.i2v_only,
                "image_start": str(img_a),
                "image_end": str(img_b) if not args.i2v_only else None,
            }
            meta_path = settings_dir / f"job_{idx + 1:03d}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(job_meta, f, indent=2, ensure_ascii=False)

            result_path = run_kling_job(
                kling_client=kling_client,
                img_a=img_a,
                img_b=img_b,
                prompt=prompt,
                clip_path=clip_path,
                model=args.model,
                duration=duration,
                mode=args.mode,
                negative_prompt=args.negative_prompt,
                i2v_only=args.i2v_only,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
        else:
            # ---- Wan2GP backend ----
            settings = build_wan2gp_settings(
                template_settings=template,
                prompt=prompt,
                first_frame_path=img_a.resolve(),
                last_frame_path=img_b.resolve(),
                output_filename=clip_path.stem,
                frames=frames,
                resolution=resolution,
                model_type=args.model_type,
                seed=args.seed if args.seed != -1 else -1,
                guidance_scale=args.guidance_scale,
                steps=args.steps,
            )

            settings_path = settings_dir / f"settings_{idx + 1:03d}.json"
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)

            print(f"    Running Wan2GP (profile {args.profile}, {resolution})...")

            result_path = run_wan2gp_job(
                settings_path=settings_path,
                output_dir=str(clips_dir),
                wan2gp_dir=str(wan2gp_dir),
                conda_env=args.wan2gp_env,
                profile=args.profile,
                attention=args.attention,
                timeout=args.timeout,
            )

        clip_elapsed = time.time() - clip_start

        if result_path:
            result_path = Path(result_path)
            # Rename to expected clip name if needed
            if result_path != clip_path:
                try:
                    shutil.move(str(result_path), str(clip_path))
                except Exception:
                    clip_path = result_path  # use whatever name backend gave

            clip_paths.append(clip_path)
            print(f"    Done in {format_elapsed(clip_elapsed)} -> {clip_path.name}")
        else:
            failed_pairs.append((idx + 1, img_a.name, img_b.name))
            print(f"    FAILED after {format_elapsed(clip_elapsed)}")

        # ETA estimate
        completed = idx + 1
        avg_time = (time.time() - run_start) / completed
        remaining = (len(pairs) - completed) * avg_time
        print(f"    ETA: ~{format_elapsed(remaining)} remaining")

    # ---- [3/4] Report ----
    print(f"\n[3/4] Generation complete: {len(clip_paths)}/{len(pairs)} clips succeeded")
    if failed_pairs:
        print("  Failed pairs:")
        for num, a, b in failed_pairs:
            print(f"    Pair {num}: {a} -> {b}")

    if not clip_paths:
        print("\n  No clips generated. Cannot concatenate.")
        if is_kling:
            print("  Check KLING_ACCESS_KEY/KLING_SECRET_KEY and API credits.")
        else:
            print("  Check Wan2GP installation and settings.")
        return

    # ---- [4/4] Concatenate ----
    print("\n[4/4] Concatenating clips into final video...")
    final_path = output_folder / "final_sequence.mp4"
    result = concatenate_clips(clip_paths, final_path)

    total_elapsed = time.time() - run_start
    print(f"\n{'=' * 60}")
    if result:
        print(f"  SUCCESS")
        print(f"  Final video: {result}")
        print(f"  Clips:       {len(clip_paths)}/{len(pairs)}")
    else:
        print(f"  Concatenation failed. Individual clips available in:")
        print(f"    {clips_dir}")
    print(f"  Total time:  {format_elapsed(total_elapsed)}")
    print(f"  Output dir:  {output_folder}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
