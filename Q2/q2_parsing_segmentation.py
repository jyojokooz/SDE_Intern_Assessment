# ============================================================
# Q2: HUMAN PARSING & GARMENT SEGMENTATION PIPELINE
# Human parsing: SegFormer fine-tuned for clothes parsing
# (mattmdjaga/segformer_b2_clothes) — lighter than SCHP, runs
# comfortably on a T4, MIT-style open weights.
# Garment background removal: rembg (u2net) — lightweight,
# handles flat garment photos well including strappy/sleeveless
# garments (garment_03).
#
# UPDATED: now saves TWO separate mask outputs per person image:
#   - processed/clothes_mask/   -> true binary upper-clothes mask
#                                  (0/255 only). This is what Q3
#                                  needs to feed CatVTON for inpainting.
#   - processed/person_mask/    -> full multi-label parsing map,
#                                  kept only for documentation/
#                                  debugging (e.g. inspecting the
#                                  crossed-arms edge case visually).
# ============================================================

!pip install -q transformers torch torchvision pillow numpy scipy rembg onnxruntime

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

# rembg's `remove`/`new_session` import onnxruntime internally and will
# raise SystemExit (not a normal exception) if onnxruntime is missing —
# import it in its own try/except so we get a clear, catchable error
# instead of an IPython traceback crash if the pip install above didn't
# take effect for any reason (e.g. a stale cached environment).
try:
    from rembg import remove, new_session
except SystemExit:
    raise RuntimeError(
        "rembg failed to import because onnxruntime is missing. Run: "
        "!pip install -q onnxruntime  (or onnxruntime-gpu on a GPU runtime), "
        "then Runtime -> Restart session, and re-run this cell from the top."
    )

# ------------------------------------------------------------
# Sanity check: make sure Q1's unzip actually produced the
# expected folders before loading any models. Prevents wasting
# time loading SegFormer/rembg only to find 0 images later.
# ------------------------------------------------------------
_base_check = "sample_files"
_required_dirs = [
    os.path.join(_base_check, "test_pairs", "garment"),
    os.path.join(_base_check, "test_pairs", "person"),
]
_missing = [d for d in _required_dirs if not os.path.isdir(d)]
if _missing:
    raise FileNotFoundError(
        f"Missing expected folder(s): {_missing}. Make sure sample_files.zip has "
        f"been extracted (run the Q1 setup cell first, or !unzip -q -o sample_files.zip) "
        f"before running Q2."
    )
for _d in _required_dirs:
    _n = len([f for f in os.listdir(_d) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"  {_d}: {_n} image(s) found")
    if _n == 0:
        raise RuntimeError(f"{_d} exists but has no images — check the extracted contents.")

print("\nLoading segmentation models...")

device = "cuda" if torch.cuda.is_available() else "cpu"
rmbg_session = new_session("u2net")
seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device)
seg_model.eval()
print("Models loaded.\n")

# Label 4 = "Upper-clothes" in this checkpoint's label map.
UPPER_CLOTHES_LABEL = 4

base_path = "sample_files"
garment_dir = os.path.join(base_path, "test_pairs", "garment")
person_dir = os.path.join(base_path, "test_pairs", "person")
edge_case_dir = os.path.join(base_path, "edge_cases")

out_garment = os.path.join(base_path, "processed", "garment")
out_person_agnostic = os.path.join(base_path, "processed", "person_agnostic")
out_person_mask = os.path.join(base_path, "processed", "person_mask")
out_clothes_mask = os.path.join(base_path, "processed", "clothes_mask")  # NEW
os.makedirs(out_garment, exist_ok=True)
os.makedirs(out_person_agnostic, exist_ok=True)
os.makedirs(out_person_mask, exist_ok=True)
os.makedirs(out_clothes_mask, exist_ok=True)  # NEW

def parse_and_mask_person(image_path, out_name):
    """Runs human parsing + builds the agnostic (clothing-masked) image."""
    image = Image.open(image_path).convert("RGB")
    inputs = seg_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    pred_seg = F.interpolate(
        outputs.logits.cpu(), size=image.size[::-1],
        mode="bilinear", align_corners=False
    ).argmax(dim=1)[0].numpy()

    clothes_mask = (pred_seg == UPPER_CLOTHES_LABEL)

    # NEW: Save the actual binary mask — this is what CatVTON (Q3) needs
    # for inpainting: 0 = keep pixel, 255 = region to regenerate.
    Image.fromarray((clothes_mask * 255).astype(np.uint8)).save(
        os.path.join(out_clothes_mask, out_name)
    )

    # Save the raw parsing map (all labels, for documentation/debugging only —
    # NOT used as an inpainting mask; values span the full label range, not
    # just 0/255).
    parsing_vis = (pred_seg * (255 // max(pred_seg.max(), 1))).astype(np.uint8)
    Image.fromarray(parsing_vis).save(os.path.join(out_person_mask, out_name))

    # Build agnostic image: gray out the upper-clothes region
    agnostic_img = np.array(image)
    agnostic_img[clothes_mask] = [128, 128, 128]
    Image.fromarray(agnostic_img).save(os.path.join(out_person_agnostic, out_name))

    coverage = clothes_mask.mean()
    return coverage

# --- Garment segmentation (background removal) ---
print("Processing garments...")
for f in sorted(os.listdir(garment_dir)):
    if f.lower().endswith((".png", ".jpg", ".jpeg")):
        save_name = f.rsplit(".", 1)[0] + ".png"
        img = Image.open(os.path.join(garment_dir, f)).convert("RGBA")
        result = remove(img, session=rmbg_session)
        result.save(os.path.join(out_garment, save_name), "PNG")
        print(f"  {f} -> {save_name}")

# --- Person parsing + agnostic mask (test_pairs) ---
print("\nProcessing person images (test_pairs)...")
for f in sorted(os.listdir(person_dir)):
    if f.lower().endswith((".png", ".jpg", ".jpeg")):
        coverage = parse_and_mask_person(os.path.join(person_dir, f), f)
        print(f"  {f}: upper-clothes coverage = {coverage:.1%}")
        # person_02, person_03 have hair over shoulders — check the
        # parsing map visually afterward to confirm hair wasn't
        # misclassified as clothing (common failure mode with this model).

# --- Crossed-arms edge case (required deliverable) ---
crossed_arms_path = os.path.join(edge_case_dir, "person_crossed_arms.jpg")
if os.path.exists(crossed_arms_path):
    print("\nProcessing crossed-arms edge case...")
    coverage = parse_and_mask_person(crossed_arms_path, "person_crossed_arms.jpg")
    print(f"  person_crossed_arms.jpg: upper-clothes coverage = {coverage:.1%}")
    print("  NOTE FOR README: inspect processed/person_mask/person_crossed_arms.jpg —")
    print("  crossed arms often get partially masked as clothing or left unmasked,")
    print("  which affects agnostic image quality for Q3. Document what you observe.")
    print("  Also inspect processed/clothes_mask/person_crossed_arms.jpg — this is")
    print("  the actual binary mask Q3 will use for inpainting.")

print("\nQ2 complete. Outputs saved under sample_files/processed/.")
print("processed/clothes_mask/ holds the binary inpainting masks (used by Q3).")
print("processed/person_mask/ holds the full parsing maps (documentation only).")
print("Copy the relevant outputs into your Q2/ folder before pushing to GitHub.")

# ------------------------------------------------------------
# Zip everything up for download / for your Q2/ folder
# ------------------------------------------------------------
!zip -r processed_garment.zip sample_files/processed/garment
!zip -r processed_person_agnostic.zip sample_files/processed/person_agnostic
!zip -r processed_person_mask.zip sample_files/processed/person_mask
!zip -r processed_clothes_mask.zip sample_files/processed/clothes_mask