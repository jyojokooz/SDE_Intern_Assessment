# ============================================================
# Q3: END-TO-END TRY-ON INFERENCE
# Model: CatVTON (non-commercial research license — note this
# in your README) — chosen because it's the lightest of the
# three options and most Colab-T4-friendly, per the assignment's
# own recommendation.
#
# UPDATED: reads the true binary upper-clothes mask from
# processed/clothes_mask/ (written by the updated Q2 script)
# instead of processed/person_mask/, which only ever contained
# the full multi-label parsing visualization (not a clean
# inpainting mask). Mask resizing uses NEAREST (not LANCZOS) to
# keep the mask strictly binary after resize.
#
# UPDATED (2): disabled the SD inpainting safety checker. It was
# producing false-positive "Potential NSFW content" flags on
# person-agnostic try-on inputs (lots of visible skin where the
# garment was masked out is normal for this task, not NSFW), and
# silently returning solid black images instead of real results
# for pair_01 and pair_04. See the monkeypatch block below.
#
# BEFORE RUNNING: confirm GPU is attached —
#   Runtime -> Change runtime type -> T4 GPU -> Save -> reconnect
#   then check in a cell:  !nvidia-smi
# This pipeline is extremely slow on CPU (many minutes per image
# vs ~30-60s on a T4).
# ============================================================

!pip install -q rembg transformers torch torchvision pillow numpy
!git clone -q https://github.com/Zheng-Chong/CatVTON.git
!pip install -q diffusers accelerate omegaconf peft torchvision

import os
import sys
import csv
import torch
from PIL import Image

# ------------------------------------------------------------
# Preflight check: make sure Q2's outputs actually exist before
# cloning/loading CatVTON (a multi-GB download + load). Prevents
# wasting several minutes only to have every pair reported as
# "skipped_missing_input" at the end.
# ------------------------------------------------------------
_base_check = "sample_files"
_required_dirs = {
    "person_agnostic": os.path.join(_base_check, "processed", "person_agnostic"),
    "clothes_mask": os.path.join(_base_check, "processed", "clothes_mask"),
    "garment": os.path.join(_base_check, "processed", "garment"),
}
_missing = [name for name, d in _required_dirs.items() if not os.path.isdir(d)]
if _missing:
    raise FileNotFoundError(
        f"Missing processed/ subfolder(s): {_missing}. Run the updated Q2 script "
        f"first — it must produce processed/person_agnostic/, processed/clothes_mask/, "
        f"and processed/garment/ before Q3 can run."
    )
for _name, _d in _required_dirs.items():
    _n = len([f for f in os.listdir(_d) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"  {_d}: {_n} file(s) found")
    if _n == 0:
        raise RuntimeError(f"{_d} exists but is empty — re-run Q2.")

# Sanity check before loading — warn loudly if no GPU is attached.
if not torch.cuda.is_available():
    print("WARNING: torch.cuda.is_available() is False — running on CPU.")
    print("This will be very slow. Go to Runtime -> Change runtime type -> T4 GPU,")
    print("save, reconnect, and re-run this cell before continuing.")

# >>> NEW: disable the SD safety checker before the pipeline is built.
# The checker is instantiated as part of the base_ckpt components, so
# this must run BEFORE CatVTONPipeline(...) below. It flags normal
# try-on outputs (visible skin from the masked-out garment region) as
# NSFW and silently swaps in a black image instead of raising an error,
# which is why pair_01 / pair_04 came back solid black last run.
try:
    from diffusers.pipelines.stable_diffusion import safety_checker as _sc

    def _no_nsfw_check(self, images, clip_input):
        num_images = len(images) if hasattr(images, "__len__") else 1
        return images, [False] * num_images

    _sc.StableDiffusionSafetyChecker.forward = _no_nsfw_check
    print("Safety checker monkeypatch applied (NSFW false-positive filter disabled).")
except Exception as _e:
    print(f"Could not patch safety checker (continuing anyway): {_e}")
# <<< NEW

sys.path.append("CatVTON")
from model.pipeline import CatVTONPipeline

print("Loading CatVTON pipeline...")
pipeline = CatVTONPipeline(
    base_ckpt="runwayml/stable-diffusion-inpainting",
    attn_ckpt="zhengchong/CatVTON",
    attn_ckpt_version="mix",
    weight_dtype=torch.float16,
    use_tf32=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print("Pipeline loaded.\n")
print("UNet device:", next(pipeline.unet.parameters()).device)

base_path = "sample_files"
agnostic_dir = os.path.join(base_path, "processed", "person_agnostic")
mask_dir = os.path.join(base_path, "processed", "clothes_mask")  # updated: was "person_mask"
garment_dir = os.path.join(base_path, "processed", "garment")
out_dir = os.path.join(base_path, "tryon_results")
os.makedirs(out_dir, exist_ok=True)

# CONSTRAINT LOG (fill this in as you go, then copy into your README):
# - If you hit CUDA OutOfMemoryError: drop resolution below to (384, 512)
#   or reduce num_inference_steps.
# - fp16 is already used above via weight_dtype to keep VRAM down.
# - Mask source fixed: processed/person_mask/ holds the full multi-label
#   parsing map (documentation only); processed/clothes_mask/ holds the
#   true binary upper-clothes mask used for inpainting.
# - SD safety checker disabled: it produced false-positive NSFW flags on
#   normal person-agnostic try-on inputs and returned black images for
#   pair_01 and pair_04. Patched StableDiffusionSafetyChecker.forward to
#   a no-op before pipeline construction. Noted here for transparency.

def resize_for_memory(img, target_size=(512, 768), resample=Image.LANCZOS):
    return img.resize(target_size, resample)

def find_file(directory, stem):
    """Find a file in `directory` matching `stem` regardless of extension."""
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None

# Read the official pairs from the manifest, plus your two sourced pairs.
# Adjust stems here to match whatever you named your sourced images
# (e.g. "my_person_1" / "my_garment_1").
pairs_to_run = [
    {"id": "pair_01", "person_stem": "person_01", "garment_stem": "garment_01"},
    {"id": "pair_02", "person_stem": "person_02", "garment_stem": "garment_02"},
    {"id": "pair_03", "person_stem": "person_03", "garment_stem": "garment_03"},
    {"id": "pair_04", "person_stem": "my_person_1", "garment_stem": "my_garment_1"},
    {"id": "pair_05", "person_stem": "my_person_2", "garment_stem": "my_garment_2"},
]

results_log = []

for pair in pairs_to_run:
    print(f"Running {pair['id']}...")
    agnostic_path = find_file(agnostic_dir, pair["person_stem"])
    mask_path = find_file(mask_dir, pair["person_stem"])
    garment_path = find_file(garment_dir, pair["garment_stem"])

    if not all([agnostic_path, mask_path, garment_path]):
        print(f"  SKIPPED — missing preprocessed file for {pair['id']}. "
              f"Run Q2 first (with the clothes_mask fix), or check your "
              f"sourced-image filenames.")
        results_log.append({"pair_id": pair["id"], "status": "skipped_missing_input"})
        continue

    try:
        agnostic_img = resize_for_memory(Image.open(agnostic_path).convert("RGB"))
        # NEAREST for the mask so resizing doesn't introduce soft/gray
        # edge pixels into what needs to stay a strict binary mask.
        mask_img = resize_for_memory(
            Image.open(mask_path).convert("L"), resample=Image.NEAREST
        )
        garment_img = resize_for_memory(Image.open(garment_path).convert("RGB"))

        with torch.no_grad():
            result = pipeline(
                image=agnostic_img,
                condition_image=garment_img,
                mask=mask_img,
                num_inference_steps=30,
            )
        result_img = result[0] if isinstance(result, (list, tuple)) else result

        out_path = os.path.join(out_dir, f"{pair['id']}_result.png")
        result_img.save(out_path)
        print(f"  Saved -> {out_path}")
        results_log.append({"pair_id": pair["id"], "status": "success", "output": out_path})

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  OOM on {pair['id']} — retrying at lower resolution (384x512)...")
        try:
            agnostic_img = resize_for_memory(Image.open(agnostic_path).convert("RGB"), (384, 512))
            mask_img = resize_for_memory(
                Image.open(mask_path).convert("L"), (384, 512), resample=Image.NEAREST
            )
            garment_img = resize_for_memory(Image.open(garment_path).convert("RGB"), (384, 512))
            with torch.no_grad():
                result = pipeline(image=agnostic_img, condition_image=garment_img,
                                   mask=mask_img, num_inference_steps=20)
            result_img = result[0] if isinstance(result, (list, tuple)) else result
            out_path = os.path.join(out_dir, f"{pair['id']}_result.png")
            result_img.save(out_path)
            print(f"  Saved (reduced res) -> {out_path}")
            results_log.append({"pair_id": pair["id"], "status": "success_reduced_res", "output": out_path})
        except Exception as e2:
            print(f"  Failed even at reduced resolution: {e2}")
            results_log.append({"pair_id": pair["id"], "status": f"failed: {e2}"})

    except Exception as e:
        print(f"  Error: {e}")
        results_log.append({"pair_id": pair["id"], "status": f"failed: {e}"})

print("\nQ3 complete.")
print("Copy sample_files/tryon_results/ into your Q3/ folder before pushing to GitHub.")
print("Copy the constraint log below into your README:")
for r in results_log:
    print(" ", r)

# ------------------------------------------------------------
# Zip results for download
# ------------------------------------------------------------
!zip -r tryon_results.zip sample_files/tryon_results