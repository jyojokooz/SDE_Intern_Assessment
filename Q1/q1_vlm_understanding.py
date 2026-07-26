# ============================================================
# PHASE 0: SETUP (run this first, once per session)
# ============================================================
# UPDATED: now reads sample_files.zip from Google Drive instead
# of Colab's local file browser. Local-upload files are wiped
# every time the runtime disconnects/restarts, so you had to
# re-upload the zip every session. Google Drive persists across
# sessions — you only upload the zip there ONCE, ever. Mounting
# Drive each session just needs one auth click.
#
# ONE-TIME SETUP (do this once, outside Colab):
#   1. Go to drive.google.com
#   2. Create a folder, e.g. MyDrive/xap_project/
#   3. Upload sample_files.zip into that folder
#   That's it — you never need to re-upload it again.
#
# Still has: zip-integrity check before proceeding, and a
# post-unzip sanity check on directory structure, so a bad/
# corrupted zip fails fast BEFORE the multi-GB model loads,
# instead of silently producing "Found 0 images to process"
# after a 10+ minute wasted model load.
# ============================================================

import os
import zipfile

# --- Mount Google Drive (prompts a one-click auth popup) ---
from google.colab import drive
drive.mount('/content/drive')

# --- EDIT THIS PATH to match where you uploaded the zip in Drive ---
zip_path = "/content/drive/MyDrive/xap_project/sample_files.zip"

if not os.path.exists(zip_path):
    raise FileNotFoundError(
        f"'{zip_path}' not found in Google Drive. Go to drive.google.com, "
        f"upload sample_files.zip into MyDrive/xap_project/ (or update `zip_path` "
        f"above to match wherever you put it), wait for the Drive upload to finish, "
        f"then re-run this cell."
    )

size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Found {zip_path} ({size_mb:.1f} MB)")

if size_mb < 1:
    raise ValueError(
        f"'{zip_path}' is only {size_mb:.2f} MB — this looks like an incomplete/failed "
        f"upload, not the real sample_files.zip. Re-upload it to Drive fully and "
        f"re-run this cell."
    )

if not zipfile.is_zipfile(zip_path):
    raise ValueError(
        f"'{zip_path}' exists ({size_mb:.1f} MB) but is not a valid zip file "
        f"(corrupted or truncated). Re-download sample_files.zip from the assignment "
        f"source and re-upload it fully to Drive before running this cell again."
    )

print("Zip file looks valid. Extracting to local disk (/content/sample_files/)...")
# NOTE: we extract to local Colab disk (not Drive) for fast read/write during
# the session — only the source zip itself needs to live on Drive.
import subprocess
result = subprocess.run(["unzip", "-q", "-o", zip_path], capture_output=True, text=True)
if result.returncode != 0:
    print(result.stdout)
    print(result.stderr)
    raise RuntimeError("unzip failed even though the zip passed integrity check — see output above.")

# Sanity check the expected structure actually exists now.
base_path = "sample_files"
expected_dirs = [
    os.path.join(base_path, "test_pairs", "person"),
    os.path.join(base_path, "test_pairs", "garment"),
    os.path.join(base_path, "edge_cases"),
]
missing = [d for d in expected_dirs if not os.path.isdir(d)]
if missing:
    raise FileNotFoundError(
        f"Extraction finished but these expected folders are missing: {missing}. "
        f"Check that sample_files.zip has the expected structure (test_pairs/person, "
        f"test_pairs/garment, edge_cases) at its root, not nested inside another folder."
    )

for d in expected_dirs:
    n = len([f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"  {d}: {n} image(s) found")

print("sample_files/ structure looks correct. Proceeding to load the model.\n")

!pip install -q torch torchvision transformers accelerate bitsandbytes pillow

import json
import re
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

# ============================================================
# Q1: GARMENT & BODY UNDERSTANDING WITH A VISION-LANGUAGE MODEL
# Model choice: LLaVA-NeXT (llava-v1.6-mistral-7b) — Apache 2.0
# license, strong instruction-following for structured JSON,
# runs in 4-bit on a free T4 (~5GB VRAM). Justify this choice
# in your README: alternatives like Florence-2 are faster/lighter
# but less reliable at following a strict JSON schema; MiniCPM-V
# 2.6 is a good alternative if you hit VRAM issues with LLaVA.
# ============================================================

print("Loading LLaVA-NeXT model... this takes a few minutes.")
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("Model loaded.\n")

JSON_INSTRUCTIONS = """
Analyze the image. Output ONLY a valid JSON object matching the exact structure below.
Do not include any other text, explanation, or markdown formatting.
Use double quotes for all strings. Do not use trailing commas.
If the image is only a flat garment (no person), set all person_attributes to null.
If no person is visible in the image at all, set "person_detected" to false.

{
  "person_detected": true or false,
  "garment_attributes": {
    "garment_type": "T-shirt / Dress / Tank Top / etc.",
    "sleeve_length": "Short / Long / Sleeveless / None",
    "neckline": "Round / V-neck / Collar / None",
    "primary_color": "color name",
    "pattern": "Solid / Striped / Graphic / etc."
  },
  "person_attributes": {
    "pose_category": "front-facing / side / seated / null",
    "upper_body_visible": true or false,
    "lower_body_visible": true or false
  }
}
"""

images_to_process = []

dirs_to_check = [
    os.path.join(base_path, "test_pairs", "person"),
    os.path.join(base_path, "test_pairs", "garment"),
    os.path.join(base_path, "edge_cases"),
]

# NOTE: process ALL edge_cases images, including no_person.jpg —
# the assignment needs no_person.jpg classified too (person_detected: false)
# for your Q5 guardrail demo to make sense with this file.
for d in dirs_to_check:
    if os.path.exists(d):
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                images_to_process.append(os.path.join(d, f))

print(f"Found {len(images_to_process)} images to process.")

if len(images_to_process) == 0:
    raise RuntimeError(
        "No images found even though sample_files/ exists and passed the structure "
        "check above. Double-check the folder contents with: !find sample_files -type f"
    )

def load_rgb(img_path):
    """Handle transparent PNGs safely by flattening onto white."""
    image = Image.open(img_path)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.convert("RGBA").split()[3])
        return background
    return image.convert("RGB")

final_results = {}

for img_path in images_to_process:
    filename = os.path.basename(img_path)
    print(f"Analyzing {filename}...", end=" ")

    answer = None
    try:
        image = load_rgb(img_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": JSON_INSTRUCTIONS},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200)
        res = processor.decode(output[0], skip_special_tokens=True)
        answer = res.split("[/INST]")[-1].strip()

        clean_json = answer.replace("```json", "").replace("```", "").strip()
        clean_json = re.sub(r",\s*}", "}", clean_json)
        clean_json = re.sub(r",\s*]", "]", clean_json)

        parsed_data = json.loads(clean_json)
        final_results[filename] = parsed_data
        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        final_results[filename] = {"error": "Failed to parse", "raw_output": answer}

# --- Save output to Drive too, so results survive a runtime reset ---
output_file = "/content/drive/MyDrive/xap_project/sample_output_q1.json"
with open(output_file, "w") as f:
    json.dump(final_results, f, indent=4)

print(f"\nQ1 complete. Saved to {output_file}")
print("Copy this file into your Q1/ folder before pushing to GitHub.")

# Keep `model` and `processor` in memory — Q5 can reuse this VLM later
# in the same Colab session if you run all cells top to bottom.