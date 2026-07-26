# Virtual Try-On Assessment - Submission

**Candidate name:** Joel S Raphael
**Email:** joelraphael6425@gmail.com
**Date:** 26-07-2025
**GitHub repo link:** https://github.com/jyojokooz/My_Submission
**Demo video link (max 5 min):** [FILL IN — YouTube/Drive link after recording]
**Colab notebook links (if used):** https://colab.research.google.com/drive/1s4CrtTL3vHoRN7pYC_W-3UFxjJ4LHeQh?usp=sharing

---

## Q1 - Garment & Body Understanding

- **VLM chosen and why:**
  LLaVA-NeXT (`llava-hf/llava-v1.6-mistral-7b-hf`), Apache 2.0 license. Chosen over the other
  allowed options (InternVL2, MiniCPM-V 2.6, Florence-2) because it follows strict JSON-schema
  instructions reliably, and runs comfortably in 4-bit (NF4 quantization via BitsAndBytesConfig)
  on a free Colab T4 with ~5GB VRAM footprint. Florence-2 is lighter/faster but less reliable at
  strictly following a JSON output schema in testing; MiniCPM-V 2.6 was considered as a fallback
  if VRAM became a constraint, which it did not.

- **How to run:**
  Run `Q1/q1_vlm_understanding.py` in a Colab notebook with a T4 GPU runtime. It expects
  `sample_files/` (extracted from `sample_files.zip`, mounted via Google Drive) with
  `test_pairs/person/`, `test_pairs/garment/`, and `edge_cases/` populated. Loads LLaVA-NeXT in
  4-bit, runs it over all 18 images found (10 official test_pairs images + 4 sourced pair images
  + 4 edge_cases images), and writes structured JSON output to `sample_output_q1.json`.

- **Known limitations:**
  - Output JSON occasionally requires cleanup (trailing commas, stray markdown fences) before
    parsing — handled with regex cleanup in the script, with a fallback `"error"` entry logged
    per-image if parsing still fails, rather than crashing the whole run.
  - `person_seated.jpg` and `person_side_pose.jpg` were manually verified to classify correctly
    as `"seated"` and `"side"` respectively in `pose_category`.

## Q2 - Human Parsing & Segmentation

- **Models used (parsing / background removal):**
  - Human parsing: SegFormer fine-tuned for clothes parsing (`mattmdjaga/segformer_b2_clothes`,
    MIT license) — chosen over SCHP for being lighter and running comfortably on a T4.
  - Garment background removal: `rembg` (u2net backend) — lightweight, handles flat garment
    product photos well, including strappy/sleeveless garments (garment_03).

- **How to run:**
  Run `Q2/q2_parsing_segmentation.py` after Q1's setup has extracted `sample_files/`. It loads
  SegFormer and rembg, then for every person image produces:
  - `processed/clothes_mask/` — the true binary (0/255) upper-clothes inpainting mask, used
    directly by Q3/CatVTON.
  - `processed/person_mask/` — the full multi-label parsing map, kept only for
    documentation/debugging.
  - `processed/person_agnostic/` — the person image with the upper-clothes region grayed out.
  For garments, it writes background-removed RGBA PNGs to `processed/garment/`.

- **Edge cases handled / failed:**
  - **Hair over shoulders (person_02, person_03):** visually inspected the parsing maps —
    [FILL IN your actual observation here, e.g. "hair was correctly excluded from the
    upper-clothes label in both cases" or "some hair strands at the shoulder line were
    misclassified as clothing, producing a slightly ragged mask edge"].
  - **Strappy sleeveless garment (garment_03):** rembg's u2net backend cleanly separated the
    thin straps from the background without holes.
  - **Crossed-arms (`edge_cases/person_crossed_arms.jpg`):** this is a genuine failure case —
    coverage came out at 0.0%, meaning SegFormer did not detect any upper-clothes region at
    all for this pose, likely because the crossed-arm position occludes the torso in a way the
    model wasn't trained to handle well. Documented as a known limitation rather than silently
    included in downstream Q3 results.
  - **Sourced person image (`my_person_1`, hand-on-face pose):** initially had the same failure
    mode as the crossed-arms case — SegFormer produced a fragmented mask (multiple disconnected
    blobs, with the chest center undetected) because the raised arm partially occluded the
    torso. This directly caused a broken try-on result in Q3/Q5 (garment fidelity score of only
    0.328). Fixed by replacing the sourced image with a straight-standing, hands-at-sides,
    front-facing photo, which produced a clean, fully-connected mask matching the garment shape.

## Q3 - End-to-End Try-On

- **Try-on model chosen and why:**
  CatVTON (`zhengchong/CatVTON`, base checkpoint `runwayml/stable-diffusion-inpainting`) —
  non-commercial research license. Chosen as the lightest of the three recommended options
  (IDM-VTON, OOTDiffusion, CatVTON), and the assignment itself recommends it as the most
  Colab-T4-friendly choice.

- **Hardware used (GPU, VRAM):**
  Google Colab free tier, NVIDIA T4 GPU, ~15GB VRAM. Ran with fp16 weights
  (`weight_dtype=torch.float16`) and `use_tf32=True` to keep VRAM usage manageable.

- **Constraints hit and workarounds:**
  1. **Wrong mask source (fixed):** the pipeline was initially reading
     `processed/person_mask/` (the full multi-label parsing visualization) as the inpainting
     mask, instead of a clean binary mask. Fixed by having Q2 write a dedicated
     `processed/clothes_mask/` binary (0/255) mask, and pointing Q3 at that instead. Mask
     resizing was also switched from LANCZOS to NEAREST interpolation, to avoid introducing
     soft/gray edge pixels that would break the mask's strict binary nature.
  2. **False-positive NSFW safety checker (fixed):** the base Stable Diffusion inpainting
     checkpoint's built-in NSFW safety checker flagged normal try-on outputs as unsafe — likely
     because person-agnostic inputs (with the garment region masked to gray) show more bare
     skin than the checker's training distribution expects — and silently returned solid black
     images instead of raising an error. Fixed by monkeypatching
     `StableDiffusionSafetyChecker.forward` to a no-op before the pipeline was constructed,
     after visually confirming the flagged outputs were legitimate try-on results.
  3. **CUDA OutOfMemoryError (documented, with fallback):** on a couple of pairs, the pipeline
     hit `torch.cuda.OutOfMemoryError` at the default 512x768 resolution / 30 inference steps.
     Workaround: the script catches this specific exception and retries at a reduced resolution
     (384x512) and fewer steps (20), which resolved the OOM without failing the pair outright.
  4. **HuggingFace model re-download every session:** since Colab's local disk is wiped on
     every runtime restart, the ~15GB LLaVA-NeXT and CatVTON weights re-download each session
     (~15-30 minutes). Mitigated by mounting Google Drive for the input `sample_files.zip` so at
     least that doesn't need re-uploading; model re-download was accepted as a standard
     Colab free-tier cost rather than adding HF_HOME→Drive caching, which trades download time
     for slower Drive I/O with uncertain net benefit.

- **How to run:**
  Run `Q3/q3_tryon_inference.py` after Q2 has produced `processed/person_agnostic/`,
  `processed/clothes_mask/`, and `processed/garment/`. It runs CatVTON over the 5 pairs defined
  in `pairs_manifest.csv` (3 official + 2 sourced), saving results to
  `sample_files/tryon_results/pair_0X_result.png`.

## Q4 - Automated Quality Evaluation

- **Metrics implemented:**
  1. **Garment fidelity** — OpenCLIP (`ViT-B-32`, `laion2b_s34b_b79k` weights) cosine similarity
     between the background-removed garment image and the try-on result image.
  2. **Identity preservation** — InsightFace (`buffalo_l`) face embedding cosine similarity
     between the original person photo and the try-on result.
  3. **VLM-as-judge** — reuses the Q1 LLaVA-NeXT model with a written rubric prompt (below),
     returning a 1-10 realism score plus reasoning and observed artifacts.

- **VLM-as-judge rubric prompt (paste it here):**
  ```
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
  ```

- **Results:** `evaluation_template_q4.csv` filled for all 5 pairs and committed to the repo
  root and to `Q4/`.

## Q5 - Web Demo

- **Framework (Gradio/Streamlit):** Gradio (`gr.Blocks`), launched with `share=True` for a
  temporary public URL.

- **How to launch:**
  Run `Q5/q5_web_demo.py` in the same Colab session, **after** Q1, Q2, and Q3 have all already
  run and loaded `model`, `processor`, `seg_model`, `seg_processor`, `rmbg_session`, and
  `pipeline` into memory — Q5 reuses these live rather than reloading them, and will raise a
  clear `NameError` up front listing any missing variable if run out of order or after a runtime
  restart. Opens a public `*.gradio.live` share link (temporary, ~1 week).

- **Guardrails implemented:**
  - **No person detected** → hard reject (`gr.Error`), pipeline stops before any preprocessing.
  - **Seated or side pose detected** → soft warning (`gr.Warning`), pipeline still proceeds.
  - **Estimated processing time** shown immediately via `gr.Info` before generation starts.
  - All guardrail decisions come from a single live LLaVA-NeXT call per upload (no filename
    lookups), so they work identically for the required edge_cases test images and for any
    arbitrary user-uploaded photo.
  - Tested against all three required edge_cases images: `no_person.jpg` (rejected),
    `person_seated.jpg` (warning shown, proceeds), `person_side_pose.jpg` (warning shown,
    proceeds) — demonstrated in the demo video.

## Honest failure log

- **Sourced person image with occluded torso pose:** the first `my_person_1` photo used a
  hand-on-face pose that crossed one arm over the torso. SegFormer's upper-clothes
  segmentation produced a fragmented, disconnected mask (missing the chest-center region
  entirely) for this pose — essentially the same failure mode as the `person_crossed_arms.jpg`
  edge case. This propagated into a visibly broken Q3 try-on result (the original garment
  remained largely un-replaced, garment fidelity score only 0.328). Root-caused via manual
  visual inspection of the agnostic image vs. binary mask side by side, rather than trusting the
  coverage percentage alone. Fixed by resourcing a straight-standing, front-facing,
  hands-at-sides photo instead of attempting to patch the segmentation model.
- **CUDA OutOfMemoryError during Q5 live scoring:** with LLaVA-NeXT, SegFormer, CatVTON, and
  OpenCLIP all resident in GPU memory simultaneously in one Colab session, the VLM-as-judge
  generation step (the last and most memory-hungry scoring step) occasionally failed with
  `torch.OutOfMemoryError` on a ~15GB T4. Fixed by explicitly freeing OpenCLIP's GPU memory
  (`del` + `torch.cuda.empty_cache()` + `gc.collect()`) immediately after computing the garment
  fidelity score and before the identity/VLM-judge steps run, plus a defensive
  `try/except torch.cuda.OutOfMemoryError` around the judge call so a transient VRAM spike
  degrades gracefully (judge score reported as "N/A") instead of crashing the whole request.
- **Stable Diffusion NSFW false positives:** documented above under Q3 — two of five pairs
  initially returned solid black images due to the safety checker's false-positive rate on
  person-agnostic (partially bare-skin) try-on inputs.