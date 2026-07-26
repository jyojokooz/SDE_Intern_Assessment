# ============================================================
# Q5: Mini Try-On Web Demo — LIVE INFERENCE VERSION
# Paste this in place of your final cell (the one starting
# "!pip install -q gradio"). Assumes model, processor, seg_model,
# seg_processor, rmbg_session, and pipeline are already loaded
# from your earlier Q1/Q2/Q3 cells in this same Colab session.
#
# UPDATED: fixed a CUDA OutOfMemoryError that happened on the
# VLM judge step (last step of scoring). Cause: LLaVA-NeXT,
# SegFormer, CatVTON, and OpenCLIP were all sitting in GPU VRAM
# at the same time on a 15GB T4 — by the time score_vlm_judge()
# ran model.generate() again, there wasn't enough free VRAM left
# for even a 186MB allocation. Fix: score_garment_fidelity() now
# frees CLIP's GPU memory (del + empty_cache + gc.collect) right
# after computing the fidelity score, before identity/VLM-judge
# scoring run, so the VLM judge call gets the freed headroom back.
# ============================================================

!pip install -q gradio

import os
import re
import json
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Preflight check: this script assumes `model`, `processor`
# (from Q1), `seg_model`, `seg_processor`, `rmbg_session` (from
# Q2), and `pipeline` (from Q3) are all already loaded in this
# same Colab session's memory. If you restarted the runtime or
# are running this in a fresh session, these won't exist and
# every button click will fail with a NameError deep inside
# Gradio instead of a clear message up front.
# ------------------------------------------------------------
_required_vars = ["model", "processor", "seg_model", "seg_processor", "rmbg_session", "pipeline"]
_missing_vars = [v for v in _required_vars if v not in globals()]
if _missing_vars:
    raise NameError(
        f"Missing variable(s) in memory: {_missing_vars}. This script reuses models "
        f"already loaded by your Q1 (model, processor), Q2 (seg_model, seg_processor, "
        f"rmbg_session), and Q3 (pipeline) cells. Run those cells first, in this same "
        f"session, before running Q5 — a runtime restart clears all of them."
    )
print("All required models found in memory:", ", ".join(_required_vars))

# ------------------------------------------------------------
# 1. LIVE Q1: VLM attribute + person-detection call
# ------------------------------------------------------------
JUDGE_INSTRUCTIONS = """
Analyze the image. First decide if a real photographed person is visible in it at all
(not a garment flat-lay, not an empty background). Output ONLY a valid JSON object,
no other text, matching this structure exactly:

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

If no person is visible, set "person_detected" to false and all person_attributes to null.
If the image is only a flat garment, set garment_attributes normally and person_attributes to null.
"""

def run_vlm_json(image: Image.Image, instructions: str, max_new_tokens=200):
    """Runs the already-loaded Q1 VLM (model/processor) on a single image and
    returns a parsed dict, or {'error': ...} if parsing failed."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instructions},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    res = processor.decode(output[0], skip_special_tokens=True)
    answer = res.split("[/INST]")[-1].strip()

    clean_json = answer.replace("```json", "").replace("```", "").strip()
    clean_json = re.sub(r',\s*}', '}', clean_json)
    clean_json = re.sub(r',\s*]', ']', clean_json)

    try:
        return json.loads(clean_json)
    except Exception as e:
        return {"error": f"parse_failed: {e}", "raw": answer}


# ------------------------------------------------------------
# 2. LIVE Q2: preprocessing (rembg on garment, human parsing +
#    agnostic mask on person)
# ------------------------------------------------------------
from rembg import remove as rembg_remove

def preprocess_garment(garment_img: Image.Image) -> Image.Image:
    return rembg_remove(garment_img.convert("RGBA"), session=rmbg_session)

def preprocess_person(person_img: Image.Image):
    image = person_img.convert("RGB")
    inputs = seg_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    pred_seg = F.interpolate(
        outputs.logits.cpu(), size=image.size[::-1],
        mode="bilinear", align_corners=False
    ).argmax(dim=1)[0].numpy()

    clothes_mask = (pred_seg == 4)  # upper-clothes label for this checkpoint
    agnostic_img = np.array(image)
    agnostic_img[clothes_mask] = [128, 128, 128]
    agnostic_img = Image.fromarray(agnostic_img)
    mask_img = Image.fromarray((clothes_mask * 255).astype(np.uint8))
    return agnostic_img, mask_img


# ------------------------------------------------------------
# 3. LIVE Q3: CatVTON inference
#    (assumes `pipeline` = CatVTONPipeline is already loaded)
# ------------------------------------------------------------
def resize_for_memory(img, target_size=(512, 768), resample=Image.LANCZOS):
    return img.resize(target_size, resample)

def run_tryon(agnostic_img, mask_img, garment_rgba):
    agnostic_img = resize_for_memory(agnostic_img)
    # NEAREST keeps the mask strictly binary after resize.
    mask_img = resize_for_memory(mask_img, resample=Image.NEAREST)
    garment_rgb = garment_rgba.convert("RGB")
    garment_rgb = resize_for_memory(garment_rgb)

    with torch.no_grad():
        result = pipeline(
            image=agnostic_img,
            condition_image=garment_rgb,
            mask=mask_img,
            num_inference_steps=30,
        )
    # CatVTON returns a list of PIL images
    return result[0] if isinstance(result, (list, tuple)) else result


# ------------------------------------------------------------
# 4. LIVE Q4: scoring (lazy-loaded so we don't hold 3 extra
#    models in VRAM unless the button is actually pressed)
# ------------------------------------------------------------
_clip_model = None
_clip_preprocess = None
_face_app = None

def get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import open_clip
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        _clip_model = _clip_model.to(device)
    return _clip_model, _clip_preprocess

def get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name="buffalo_l")
        _face_app.prepare(ctx_id=0 if device == "cuda" else -1)
    return _face_app

def score_garment_fidelity(garment_img, result_img):
    clip_model, preprocess = get_clip()
    try:
        img_g = preprocess(garment_img.convert("RGB")).unsqueeze(0).to(device)
        img_r = preprocess(result_img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_g = clip_model.encode_image(img_g)
            feat_r = clip_model.encode_image(img_r)
            feat_g /= feat_g.norm(dim=-1, keepdim=True)
            feat_r /= feat_r.norm(dim=-1, keepdim=True)
            sim = (feat_g @ feat_r.T).item()
        return round(sim, 3)
    except Exception as e:
        return f"Error: {e}"
    finally:
        # NEW: always free CLIP's VRAM right after use — the VLM judge
        # step runs later in the same request and needs the most VRAM
        # headroom of the three scoring steps. Without this, LLaVA-NeXT +
        # SegFormer + CatVTON + CLIP all sitting in VRAM at once on a
        # 15GB T4 leaves no room for the judge's generate() call, causing
        # a CUDA OutOfMemoryError.
        global _clip_model
        if _clip_model is not None:
            del _clip_model
            _clip_model = None
            torch.cuda.empty_cache()
            gc.collect()

def score_identity(person_img, result_img):
    try:
        face_app = get_face_app()
        faces_orig = face_app.get(np.array(person_img.convert("RGB"))[:, :, ::-1])
        faces_res = face_app.get(np.array(result_img.convert("RGB"))[:, :, ::-1])
        if not faces_orig or not faces_res:
            return "No face detected"
        emb1, emb2 = faces_orig[0].embedding, faces_res[0].embedding
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return round(float(sim), 3)
    except Exception as e:
        return f"Error: {e}"

def score_vlm_judge(result_img):
    # Extra safety: clear any leftover CUDA cache right before the VLM
    # judge call, since this is the most VRAM-hungry step and runs last.
    if device == "cuda":
        torch.cuda.empty_cache()

    prompt_text = """
You are an expert fashion AI judge. Analyze this virtual try-on image.
Output ONLY a valid JSON object matching this structure:
{
  "vlm_judge_score": 8,
  "vlm_judge_reasons": "Short explanation of realism, lighting, and textures.",
  "artifacts_observed": "None / describe any weird AI artifacts."
}
"""
    return run_vlm_json(result_img, prompt_text, max_new_tokens=150)


# ------------------------------------------------------------
# 5. Guardrail + main pipeline, wired together LIVE
# ------------------------------------------------------------
def process_tryon(person_path, garment_path):
    if person_path is None or garment_path is None:
        raise gr.Error("Please upload both a Person and a Garment image.")

    start_time = time.time()
    person_img = Image.open(person_path)
    garment_img = Image.open(garment_path)

    # --- Live guardrail check (Q1 VLM), no filename lookup ---
    vlm_attrs = run_vlm_json(person_img, JUDGE_INSTRUCTIONS)

    if vlm_attrs.get("error"):
        # VLM output didn't parse — fail safe by warning, not silently proceeding
        gr.Warning("Could not confidently analyze this image; results may be unreliable.")
        pose = ""
    else:
        if vlm_attrs.get("person_detected") is False:
            raise gr.Error("Rejected: No person detected in the uploaded image.")

        person_attrs = vlm_attrs.get("person_attributes") or {}
        pose = str(person_attrs.get("pose_category", "")).lower()
        if "seated" in pose or "side" in pose:
            gr.Warning(f"Pose detected as '{pose}'. Results may have artifacts.")

    # --- Estimated processing time (shown immediately, before generation) ---
    est_seconds = 25  # rough T4 estimate for CatVTON @ 30 steps, 512x768
    gr.Info(f"Estimated processing time: ~{est_seconds}s")

    # --- Live Q2: preprocessing ---
    garment_rgba = preprocess_garment(garment_img)
    agnostic_img, mask_img = preprocess_person(person_img)

    # --- Live Q3: try-on inference ---
    try:
        result_img = run_tryon(agnostic_img, mask_img, garment_rgba)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise gr.Error("GPU out of memory. Try a smaller image or restart the runtime.")

    # --- Live Q4: scoring ---
    fidelity = score_garment_fidelity(garment_rgba, result_img)
    identity = score_identity(person_img, result_img)
    try:
        judge = score_vlm_judge(result_img)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        judge = {"vlm_judge_score": "N/A", "vlm_judge_reasons": "GPU ran out of memory during judging.", "artifacts_observed": "N/A"}

    elapsed = round(time.time() - start_time, 1)

    garment_attrs = vlm_attrs.get("garment_attributes", {}) if not vlm_attrs.get("error") else {}
    info_text = (
        f"👕 Garment: {garment_attrs.get('primary_color', 'Unknown')} "
        f"{garment_attrs.get('garment_type', 'Unknown')}\n"
        f"🧍 Pose: {pose or 'unknown'}\n"
        f"⭐ Garment Fidelity (CLIP cosine sim): {fidelity}\n"
        f"🧑 Identity Preservation (face embedding sim): {identity}\n"
        f"🤖 VLM Judge Score: {judge.get('vlm_judge_score', 'N/A')} / 10\n"
        f"   Reason: {judge.get('vlm_judge_reasons', 'N/A')}\n"
        f"   Artifacts: {judge.get('artifacts_observed', 'N/A')}\n"
        f"⏱️ Actual processing time: {elapsed}s"
    )

    return result_img, info_text


# ------------------------------------------------------------
# 6. UI
# ------------------------------------------------------------
with gr.Blocks(title="Virtual Try-On Demo") as demo:
    gr.Markdown("# 👕 AI Virtual Try-On Pipeline")
    gr.Markdown(
        "Safety guardrails are active and run live on every upload — "
        "no precomputed lookups. Upload a person and a garment to test."
    )

    with gr.Row():
        with gr.Column():
            person_in = gr.Image(type="filepath", label="Upload Person Image")
            garment_in = gr.Image(type="filepath", label="Upload Garment Image")
            btn = gr.Button("Generate Try-On", variant="primary")

        with gr.Column():
            output_img = gr.Image(label="Try-On Result", type="pil")
            output_info = gr.Textbox(label="Attributes & Metrics", lines=9)

    btn.click(fn=process_tryon, inputs=[person_in, garment_in], outputs=[output_img, output_info])

demo.launch(share=True, debug=True)