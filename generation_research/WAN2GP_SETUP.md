# Wan2GP Setup & Image Sequence Pipeline

## GPU Compatibility: RTX 2080 Ti

| Spec | Value |
|------|-------|
| GPU | NVIDIA RTX 2080 Ti |
| VRAM | 11 GB |
| Architecture | Turing (SM 7.5) |
| BF16 | NOT supported (use `--fp16`) |
| Max CUDA | 12.8 (but Wan2GP recommends 12.6 for Turing) |
| SageAttention | v1.0.6 only (v2.x is NOT compatible) |
| Flash Attention | May require compilation; SDPA is safest |

### CUDA/PyTorch Versions

RTX 20XX (Turing) must use these versions per Wan2GP docs:

```
Python:     3.10.9
PyTorch:    2.6.0+cu126
CUDA:       12.6
Triton:     <3.3 (Windows)
```

**WARNING**: CUDA 12.8 + PyTorch 2.7.1 is for RTX 40XX-50XX only. Using it on Turing may cause SageAttention kernel failures.

---

## Installation

### 1. Clone Wan2GP

```bash
cd c:\Users\User\helloworld_claude
git clone https://github.com/deepbeepmeep/Wan2GP.git wan2gp
```

### 2. Create Conda Environment

```bash
conda create -n wan2gp python=3.10.9 -y
conda activate wan2gp
```

### 3. Install PyTorch (CUDA 12.6)

```bash
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

### 4. Install Wan2GP Dependencies

```bash
cd c:\Users\User\helloworld_claude\wan2gp
pip install -r requirements.txt
```

### 5. Install Triton + SageAttention (Optional)

```bash
pip install -U "triton-windows<3.3"
pip install sageattention==1.0.6
```

SageAttention gives ~30% speedup but is optional. SDPA (default) works without it.

### 6. Verify

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX 2080 Ti
```

---

## Memory Profile Recommendations (11GB VRAM)

| Profile | Strategy | Suitable for 11GB? | Speed |
|---------|----------|---------------------|-------|
| 1 | Full model in VRAM + reserve RAM | No (needs 24GB+) | Fastest |
| 2 | Parts as needed + reserve RAM | Possibly (tight) | Fast |
| 3 | Full model in VRAM | No (needs 24GB) | Fast |
| **4** | **Parts loaded on demand** | **Yes (recommended)** | **Medium** |
| 5 | Minimum RAM usage | Yes (fallback) | Slow |

Use **Profile 4** by default. Switch to Profile 5 if you get OOM errors.

---

## Model Selection

### For Image Sequence Generation (FLF-style)

| Model | Architecture | Resolution | Start+End Frame | VRAM Requirement | Notes |
|-------|-------------|------------|-----------------|------------------|-------|
| **I2V 480p 14B** | `i2v` | 832x480 | Yes (unofficial but works) | 11GB w/ Profile 4 | **Recommended for 11GB** |
| FLF2V 720p 14B | `flf2v_720p` | 1280x720 | Yes (official) | 11GB w/ Profile 5 | Slower, higher quality |
| T2V 1.3B | `t2v_1.3B` | 832x480 | No (text only) | ~6GB | Fast, no image conditioning |

**Recommendation**: Use `i2v` at 480p. It supports both start and end frame images (the description notes "End Image is not supported in the original model but seems to work well") and runs comfortably on 11GB at Profile 4.

---

## Launching the Web UI

```bash
start_wan2gp.bat
```

This activates the `wan2gp` conda env and runs with safe defaults:
- `--attention sdpa` (safest for Turing)
- `--profile 4` (11GB compatible)
- `--fp16` (required, no bf16 on Turing)

The UI opens at http://localhost:7860.

Extra args are passed through:
```bash
start_wan2gp.bat --attention sage    # try SageAttention
start_wan2gp.bat --profile 5         # less VRAM usage
start_wan2gp.bat --server-port 7865  # different port
```

---

## Exporting a Settings Template

For best results with `generate_sequence.py`, export a tested settings file:

1. Launch the web UI: `start_wan2gp.bat`
2. Select the I2V 480p model
3. Upload a test image as "Start Image"
4. Optionally upload an "End Image"
5. Enter a test prompt and run a generation
6. Once it works, click **"Export Settings"** in the UI
7. Save the `.json` file (e.g., `i2v_settings.json`)
8. Use it with the pipeline: `--settings-template i2v_settings.json`

---

## Running the Pipeline

### Basic usage

```bash
python generation_research/generate_sequence.py \
  --images-folder ./manga/0001/panels \
  --prompt "One Piece anime style, dynamic action scene"
```

### With AI prompt enhancement

```bash
python generation_research/generate_sequence.py \
  --images-folder ./manga/0001/panels \
  --prompt "turn this manga page into an anime shot" \
  --flag
```

The `--flag` option sends each image to GPT-4o which generates a detailed cinematic prompt and suggests an appropriate duration.

### With settings template (recommended)

```bash
python generation_research/generate_sequence.py \
  --images-folder ./frames \
  --prompt "smooth anime transition" \
  --settings-template ./wan2gp/i2v_settings.json \
  --flag
```

### FLF2V mode at 720p

```bash
python generation_research/generate_sequence.py \
  --images-folder ./frames \
  --prompt "anime scene transition" \
  --model-type flf2v \
  --resolution 720p \
  --profile 5
```

### Resume an interrupted run

```bash
python generation_research/generate_sequence.py \
  --images-folder ./frames \
  --prompt "anime style" \
  --output-folder ./output/generation/20260225_143022 \
  --resume
```

### All options

```
--images-folder PATH     Input folder with images (1.png..10.png, sorted alphabetically)
--prompt TEXT             Base prompt for video generation
--output-folder PATH     Output directory (default: output/generation/<timestamp>)
--flag                   Use GPT-4o to enhance prompts + suggest duration
--resolution 480p|720p   Resolution (default: 480p)
--model-type i2v|flf2v   Model (default: i2v, supports start+end frames at 480p)
--frames N               Frames per clip (default: 49 = ~2s, must be 4n+1)
--steps N                Inference steps (default: 30)
--profile 1-5            Memory profile (default: 4)
--attention MODE         sdpa|sage|flash|sage2 (default: sdpa)
--seed N                 Random seed (default: 42, -1 for random)
--guidance-scale F       Guidance scale (default: 5.0)
--settings-template PATH Exported Wan2GP settings JSON
--wan2gp-dir PATH        Wan2GP installation path
--wan2gp-env NAME        Conda env name (default: wan2gp)
--resume                 Skip existing clips
--timeout N              Seconds per clip (default: 1800)
```

---

## Output Structure

```
output/generation/<timestamp>/
  clips/
    clip_001_1-2.mp4        # Transition from image 1 to 2
    clip_002_2-3.mp4        # Transition from image 2 to 3
    ...
  settings/
    settings_001.json       # Wan2GP settings used for each clip
    settings_002.json
    ...
  final_sequence.mp4        # All clips concatenated
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM / CUDA out of memory | Use `--profile 5`, reduce `--frames`, use 480p |
| SageAttention crash | Use `--attention sdpa` (default, always safe) |
| BF16 errors | Ensure `--fp16` is set (start_wan2gp.bat does this) |
| Very slow generation | Expected: 10-30 min per 2s clip on 11GB at Profile 4 |
| No .mp4 in output | Check Wan2GP stderr, verify model downloaded, try a UI run first |
| Conda not found | Add conda to PATH, or edit start_wan2gp.bat with full conda path |
| OpenAI errors with --flag | Set OPENAI_API_KEY env var, check API quota |

---

## Expected Performance (RTX 2080 Ti, Profile 4)

| Resolution | Model | Frames | Approx Time | Notes |
|-----------|-------|--------|-------------|-------|
| 480p | I2V 14B | 49 (~2s) | ~10-20 min | Recommended |
| 480p | I2V 14B | 81 (~3.4s) | ~15-30 min | Longer clips |
| 720p | FLF2V 14B | 49 (~2s) | ~20-40 min | Profile 5 recommended |
| 480p | T2V 1.3B | 81 (~3.4s) | ~2-5 min | No image conditioning |

For a 10-image sequence (9 clips at 480p/49 frames): ~2-3 hours total.
