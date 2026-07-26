# ============================================================
# Q4: AUTOMATED QUALITY EVALUATION OF TRY-ON RESULTS
# (a) garment fidelity -> OpenCLIP cosine similarity
# (b) identity preservation -> InsightFace face embedding similarity
# (c) VLM-as-judge -> reuses your Q1 model (LLaVA-NeXT) with a
#     rubric prompt covering fit realism, artifacts, texture transfer
# ============================================================

!pip install -q open_clip_torch insightface onnxruntime-gpu pandas

import os
import re
import json
import gc
import numpy as np
import pandas as pd
import torch
from PIL import Image
from numpy.linalg import norm

base_path = "sample_files"
garment_dir = os.path.join(base_path, "processed", "garment")
person_dir = os.path.join(base_path, "test_pairs", "person")
tryon_dir = os.path.join(base_path, "tryon_results")
csv_path = os.path.join(base_path, "evaluation_template_q4.csv")

# ------------------------------------------------------------
# Preflight check: make sure Q3's outputs and the CSV template
# actually exist before loading InsightFace/OpenCLIP/LLaVA (three
# separate multi-hundred-MB to multi-GB model loads). Prevents
# wasting a long run only to fill the CSV with "Missing File".
# ------------------------------------------------------------
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"'{csv_path}' not found. Make sure sample_files.zip was extracted "
        f"(it ships with evaluation_template_q4.csv already in place)."
    )
if not os.path.isdir(tryon_dir):
    raise FileNotFoundError(
        f"'{tryon_dir}' not found. Run Q3 first — it must produce "
        f"sample_files/tryon_results/pair_0X_result.png for each pair."
    )
_n_results = len([f for f in os.listdir(tryon_dir) if f.lower().endswith(".png")])
print(f"  {tryon_dir}: {_n_results} result image(s) found")
if _n_results == 0:
    raise RuntimeError(f"{tryon_dir} exists but has no result images — re-run Q3.")

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(csv_path)
# Expected columns per the template: pair_id, garment_fidelity_score,
# identity_preservation_score, vlm_judge_score, vlm_judge_reasons,
# artifacts_observed, tryon_model
for col in ["garment_fidelity_score", "identity_preservation_score",
            "vlm_judge_score", "vlm_judge_reasons", "artifacts_observed"]:
    if col not in df.columns:
        df[col] = ""

def find_file(directory, stem):
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None

pairs_to_evaluate = [
    {"id": "pair_01", "person_stem": "person_01", "garment_stem": "garment_01", "result": "pair_01_result.png"},
    {"id": "pair_02", "person_stem": "person_02", "garment_stem": "garment_02", "result": "pair_02_result.png"},
    {"id": "pair_03", "person_stem": "person_03", "garment_stem": "garment_03", "result": "pair_03_result.png"},
    {"id": "pair_04", "person_stem": "my_person_1", "garment_stem": "my_garment_1", "result": "pair_04_result.png"},
    {"id": "pair_05", "person_stem": "my_person_2", "garment_stem": "my_garment_2", "result": "pair_05_result.png"},
]

# ------------------------------------------------------------
# STAGE 1: IDENTITY PRESERVATION (InsightFace)
# ------------------------------------------------------------
print("STAGE 1: Loading InsightFace for identity preservation...")
try:
    from insightface.app import FaceAnalysis
except ImportError as e:
    raise ImportError(
        f"Failed to import insightface/onnxruntime ({e}). Run: "
        f"!pip install -q onnxruntime-gpu insightface  then restart the runtime "
        f"and re-run this cell from the top."
    )

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0 if device == "cuda" else -1)

for pair in pairs_to_evaluate:
    person_path = find_file(person_dir, pair["person_stem"])
    res_path = os.path.join(tryon_dir, pair["result"])

    if not person_path or not os.path.exists(res_path):
        df.loc[df["pair_id"] == pair["id"], "identity_preservation_score"] = "Missing File"
        continue

    try:
        img_orig = np.array(Image.open(person_path).convert("RGB"))[:, :, ::-1]
        img_res = np.array(Image.open(res_path).convert("RGB"))[:, :, ::-1]
        faces_orig = app.get(img_orig)
        faces_res = app.get(img_res)

        if faces_orig and faces_res:
            emb1, emb2 = faces_orig[0].embedding, faces_res[0].embedding
            sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
            score = round(float(sim), 3)
        else:
            score = "No face detected"
    except Exception as e:
        score = f"Error: {e}"

    df.loc[df["pair_id"] == pair["id"], "identity_preservation_score"] = score
    print(f"  -> {pair['id']} Identity Score: {score}")

del app
gc.collect()
torch.cuda.empty_cache()

# ------------------------------------------------------------
# STAGE 2: GARMENT FIDELITY (OpenCLIP)
# ------------------------------------------------------------
print("\nSTAGE 2: Loading OpenCLIP for garment fidelity...")
import open_clip

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(device)

for pair in pairs_to_evaluate:
    garment_path = find_file(garment_dir, pair["garment_stem"])
    res_path = os.path.join(tryon_dir, pair["result"])

    if not garment_path or not os.path.exists(res_path):
        df.loc[df["pair_id"] == pair["id"], "garment_fidelity_score"] = "Missing File"
        continue

    try:
        img_garment = preprocess(Image.open(garment_path).convert("RGB")).unsqueeze(0).to(device)
        img_res = preprocess(Image.open(res_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            feat_g = clip_model.encode_image(img_garment)
            feat_r = clip_model.encode_image(img_res)
            feat_g /= feat_g.norm(dim=-1, keepdim=True)
            feat_r /= feat_r.norm(dim=-1, keepdim=True)
            sim = (feat_g @ feat_r.T).item()
            score = round(sim, 3)
    except Exception as e:
        score = f"Error: {e}"

    df.loc[df["pair_id"] == pair["id"], "garment_fidelity_score"] = score
    print(f"  -> {pair['id']} Garment Score: {score}")

del clip_model, preprocess
gc.collect()
torch.cuda.empty_cache()

# ------------------------------------------------------------
# STAGE 3: VLM-AS-JUDGE (reuses your Q1 model choice)
# ------------------------------------------------------------
print("\nSTAGE 3: Loading LLaVA-NeXT for VLM realism judge...")
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    quantization_config=quant_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# This is your written judge rubric — copy it into your README as required.
JUDGE_PROMPT = """
You are an expert fashion AI judge. Analyze this virtual try-on image.
Score it from 1 to 10 based on:
1. Fit realism — does the garment drape naturally on the body?
2. Artifacts — any blurring, warping, extra limbs, or texture bleeding?
3. Texture transfer — did the garment's pattern/color transfer accurately?
Output ONLY a valid JSON object matching this structure:
{
  "vlm_judge_score": 8,
  "vlm_judge_reasons": "Short explanation covering fit, artifacts, and texture.",
  "artifacts_observed": "None / describe any weird AI artifacts like blurry hands."
}
"""

for pair in pairs_to_evaluate:
    res_path = os.path.join(tryon_dir, pair["result"])

    if not os.path.exists(res_path):
        df.loc[df["pair_id"] == pair["id"], "vlm_judge_reasons"] = "Missing File"
        continue

    try:
        image = Image.open(res_path).convert("RGB")
        conversation = [{"role": "user", "content": [{"type": "text", "text": JUDGE_PROMPT}, {"type": "image"}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(vlm_model.device)

        with torch.no_grad():
            output = vlm_model.generate(**inputs, max_new_tokens=150)
        res = processor.decode(output[0], skip_special_tokens=True)
        answer = res.split("[/INST]")[-1].strip()

        clean_json = answer.replace("```json", "").replace("```", "").strip()
        clean_json = re.sub(r",\s*}", "}", clean_json)
        parsed_data = json.loads(clean_json)

        df.loc[df["pair_id"] == pair["id"], "vlm_judge_score"] = parsed_data.get("vlm_judge_score", "")
        df.loc[df["pair_id"] == pair["id"], "vlm_judge_reasons"] = parsed_data.get("vlm_judge_reasons", "")
        df.loc[df["pair_id"] == pair["id"], "artifacts_observed"] = parsed_data.get("artifacts_observed", "")
        print(f"  -> {pair['id']} VLM Score: {parsed_data.get('vlm_judge_score')}/10")

    except Exception as e:
        print(f"  -> {pair['id']} VLM Error: {e}")
        df.loc[df["pair_id"] == pair["id"], "vlm_judge_reasons"] = "Failed to parse JSON"

df["tryon_model"] = "CatVTON"
df.to_csv(csv_path, index=False)
print(f"\nQ4 complete. Scores saved to {csv_path}")
print("Commit this completed CSV to your repo root as required.")

# Keep `processor` and `vlm_model` in memory — Q5 can reuse this VLM
# later in the same Colab session.