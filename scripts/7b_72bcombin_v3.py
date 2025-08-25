"""
RoboSense Track1: Route tasks to Qwen2.5-VL-7B / 72B by subtask type and run inference.

- 7B handles: Perception-MCQs, Perception-VQAs-Object-Description, Prediction
- 72B handles: Perception-VQAs-Scene-Description, Planning-VQAs-Scene-Description, Planning-VQAs-Object-Description
- Use item.category + question template for robust routing
- Load models on demand with caching to avoid frequent reloads
- Automatically fallback to 7B when 72B runs OOM
- Add task hints (System + Task Hint) to constrain output style and length
- Fixed camera order + limit number of images per sample
"""

import os
import re
import json
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm

import argparse

# ========== Command-line arguments ==========
parser = argparse.ArgumentParser(description="Run Qwen2.5-VL-7B/72B routing inference on RoboSense dataset.")
parser.add_argument(
    "--output_file",
    type=str,
    default="7B_72B_V3_results.json",
    help="Output JSON file to save inference results."
)
args = parser.parse_args()

# ========== Paths ==========
json_file = "robosense_track1_release_converted.json"  # Input dataset
output_file = args.output_file                         # Output results
image_base_dir = "data/nuscenes"                       # Root directory for images (img_paths are relative)

# ========== Configurable parameters ==========
# Global default (overridden per subtask if needed)
MAX_NEW_TOKENS_DEFAULT = 128
DO_SAMPLE = False                 # Recommended False during evaluation for reproducibility
SEED = 888                        # Fixed random seed (if sampling enabled)
USE_FLASH_ATTN = True             # Set True if environment supports it
DTYPE = torch.bfloat16            # Can switch to torch.float16 if needed
DEVICE_MAP = "auto"               # Auto split across multiple GPUs
ENABLE_FALLBACK_72B_TO_7B = True  # If 72B OOM, fallback to 7B
MAX_IMAGES_PER_PROMPT = 6         # Max images per sample

# Fixed camera order (skip missing, do not shuffle)
CAM_ORDER = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"
]

# ========== Model repositories ==========
MODEL_7B  = "/workspace/Qwen2.5-VL-7B-Instruct"
MODEL_72B = "/workspace/Qwen2.5-VL-72B-Instruct"
USE_LORA_7B = True
LORA_7B_DIR = "/workspace/qwen7b_drivebench_lora_4gpus/checkpoint-736"

def _find_latest_checkpoint(dir_path: str) -> str | None:
    """Find the latest checkpoint directory (checkpoint-xxxx) under dir_path."""
    if not os.path.isdir(dir_path):
        return None
    pat = re.compile(r"^checkpoint-(\d+)$")
    best = None
    best_step = -1
    for name in os.listdir(dir_path):
        m = pat.match(name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best = os.path.join(dir_path, name)
    return best

def _resolve_lora_dir(lora_dir: str) -> str:
    """
    1) If lora_dir itself contains adapter_model.safetensors -> return it
    2) Otherwise treat it as LoRA root dir, pick the latest checkpoint-*
    """
    if os.path.isdir(lora_dir):
        if os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors")) and \
           os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
            return lora_dir
        latest = _find_latest_checkpoint(lora_dir)
        if latest and os.path.exists(os.path.join(latest, "adapter_model.safetensors")):
            return latest
    return lora_dir  # fallback

def resolve_local_hf_repo(path_or_name: str) -> str:
    """
    - If it's an online HF name (no path separators), return as is.
    - If local dir:
        * If config.json exists in root, use this dir.
        * If it's a HF cache repo root (contains snapshots/), pick latest snapshot.
    """
    if not os.path.sep in path_or_name and not path_or_name.startswith(("hf://", "./", "../")):
        return path_or_name

    path = path_or_name
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "config.json")):
            return path
        snaps_dir = os.path.join(path, "snapshots")
        if os.path.isdir(snaps_dir):
            snaps = sorted(
                [d for d in os.listdir(snaps_dir) if os.path.isdir(os.path.join(snaps_dir, d))]
            )
            if snaps:
                return os.path.join(snaps_dir, snaps[-1])
    return path_or_name

# ========== Subtask constants ==========
SUB_PER_MCQS = "Perception-MCQs"
SUB_PER_VQA_OBJ = "Perception-VQAs-Object-Description"
SUB_PER_VQA_SCENE = "Perception-VQAs-Scene-Description"
SUB_PRED = "Prediction"
SUB_PLAN_VQA_SCENE = "Planning-VQAs-Scene-Description"
SUB_PLAN_VQA_OBJ = "Planning-VQAs-Object-Description"

# Routing responsibility
TASKS_7B = {SUB_PER_MCQS, SUB_PER_VQA_OBJ, SUB_PRED}
TASKS_72B = {SUB_PER_VQA_SCENE, SUB_PLAN_VQA_SCENE, SUB_PLAN_VQA_OBJ}

# ========== Precompiled regex (case-insensitive) ==========
RE_MCQS = re.compile(r"(please select the correct answer|following options|\bA\.\s|\bB\.\s|\bC\.\s|\bD\.\s)", re.I)
RE_VISUAL_OBJ = re.compile(r"what\s+is\s+the\s+visual\s+description\s+of\s*<c\d+,\s*cam_", re.I)
RE_SCENE_IMPORTANT = re.compile(r"important\s+objects\s+in\s+the\s+current\s+scene", re.I)

RE_PREDICTION = re.compile(r"(be in the moving direction|change its motion state|would\s*<c|will\s*<c)", re.I)

RE_PLAN_SCENE = re.compile(
    r"(safe actions|dangerous actions|comment on this scene|issues worth noting|under what conditions|priority of the objects)",
    re.I,
)
RE_PLAN_OBJECT = re.compile(
    r"(based on\s*<c\d+|collision\s+with\s*<c\d+|what actions.*<c\d+)",
    re.I,
)

# ========== Task prompt (System + Task Hint) ==========
SYSTEM_PROMPT = """You are a helpful autonomous driving assistant that can answer questions about images and videos. 
You are providing images from multi-view sensors ordered as [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT]. 
The object coordinates are provided in the format of <id, camera_view, x, y>. 
The coordinate is the center of the bounding box where the image resolution is 1600x900.
"""

def task_hint_for_subtask(subtask: str) -> str:
    """Provide task hint per subtask. For Perception-MCQs / Perception-VQAs-Object-Description / Planning-VQAs-Object-Description -> no hint."""
    if subtask in (SUB_PER_MCQS, SUB_PER_VQA_OBJ, SUB_PLAN_VQA_OBJ):
        return ""
    if subtask == SUB_PER_VQA_SCENE:
        return "TASK=Perception-VQAs-Scene-Description. List up to 5 important objects and give a one-sentence summary."
    if subtask == SUB_PRED:
        return "TASK=Prediction. Answer concisely whether the object will move in the given direction or change state (yes/no) and one short reason."
    if subtask == SUB_PLAN_VQA_SCENE:
        return "TASK=Planning-VQAs-Scene-Description. Give up to 3 safe actions and up to 3 dangerous actions for the scene, plus one short note."
    return ""

def max_tokens_for_subtask(subtask: str) -> int:
    """No extra max_tokens restriction for 3 tasks; others retain limits."""
    if subtask in (SUB_PER_MCQS, SUB_PER_VQA_OBJ, SUB_PLAN_VQA_OBJ):
        return MAX_NEW_TOKENS_DEFAULT
    if subtask == SUB_PRED:
        return 128
    if subtask == SUB_PER_VQA_SCENE:
        return 256
    if subtask == SUB_PLAN_VQA_SCENE:
        return 512
    return MAX_NEW_TOKENS_DEFAULT

# ========== Model cache ==========
_model_cache = {}

def set_seed(seed: int):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_and_processor(which: str):
    """Return (model, processor), which in {"7B", "72B"}, with caching."""
    if which in _model_cache:
        return _model_cache[which]

    repo_raw = MODEL_7B if which == "7B" else MODEL_72B
    repo = resolve_local_hf_repo(repo_raw)

    kwargs = dict(torch_dtype=DTYPE, device_map=DEVICE_MAP)
    if USE_FLASH_ATTN:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo, **kwargs)

    proc_from = repo
    if which == "7B" and USE_LORA_7B:
        from peft import PeftModel
        lora_repo_raw = LORA_7B_DIR
        lora_repo = _resolve_lora_dir(lora_repo_raw)
        model = PeftModel.from_pretrained(model, lora_repo)
        model.eval()
        proc_from = repo
        if os.path.exists(os.path.join(lora_repo, "tokenizer_config.json")) or \
           os.path.exists(os.path.join(lora_repo, "chat_template.json")):
            proc_from = lora_repo

    processor = AutoProcessor.from_pretrained(proc_from)
    _model_cache[which] = (model, processor)
    return model, processor

def classify_subtask(item) -> str:
    """Classify subtask by category + question template."""
    q = (item.get("question") or "").strip()
    cat = (item.get("category") or "").strip().lower()

    if cat == "prediction":
        return SUB_PRED
    if cat == "planning":
        if RE_PLAN_OBJECT.search(q):
            return SUB_PLAN_VQA_OBJ
        if RE_PLAN_SCENE.search(q):
            return SUB_PLAN_VQA_SCENE
        return SUB_PLAN_VQA_SCENE

    if RE_MCQS.search(q):
        return SUB_PER_MCQS
    if RE_VISUAL_OBJ.search(q):
        return SUB_PER_VQA_OBJ
    if RE_SCENE_IMPORTANT.search(q):
        return SUB_PER_VQA_SCENE

    return SUB_PER_VQA_SCENE

def route_model(item) -> str:
    """Return '7B' or '72B', and record subtask."""
    sub = classify_subtask(item)
    item["subtask"] = sub
    return "7B" if sub in TASKS_7B else "72B"

def load_images_in_order(image_paths_dict: dict, base_dir: str, max_images: int):
    """Load images in fixed camera order; limit max_images."""
    images = []
    if not image_paths_dict:
        return images
    for cam_name in CAM_ORDER:
        if cam_name not in image_paths_dict:
            continue
        rel_img_path = image_paths_dict[cam_name]
        img_path = os.path.join(base_dir, rel_img_path)
        if os.path.exists(img_path):
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"❌ Failed to load image: {img_path} -> {e}")
        else:
            print(f"⚠️ Image not found: {img_path}")
        if len(images) >= max_images:
            break
    return images

def build_messages_for_item(question: str, images, subtask: str):
    """Build messages including system + task hint + images + question"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    content = []
    hint = task_hint_for_subtask(subtask)
    if hint:
        content.append({"type": "text", "text": hint})
    content.extend([{"type": "image", "image": img} for img in images])
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})
    return messages

def run_inference_on_item(item, which_model: str):
    """Run inference on one item using specified model; return answer text."""
    model, processor = get_model_and_processor(which_model)
    question = (item.get("question") or "").strip()
    image_paths_dict = item.get("img_paths", {})
    images = load_images_in_order(image_paths_dict, image_base_dir, MAX_IMAGES_PER_PROMPT)
    if not images:
        return "❌ Skipped due to missing images."
    if not question:
        return "❌ Skipped due to missing question."
    subtask = item.get("subtask", "")
    messages = build_messages_for_item(question, images, subtask)

    with torch.no_grad():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        max_new_tokens = max_tokens_for_subtask(subtask)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    return output_text

def main():
    set_seed(SEED if DO_SAMPLE else None)
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    output_data = []
    oom_count = 0
    for idx, item in tqdm(enumerate(data), total=len(data), desc="Running Inference"):
        which_model = route_model(item)
        item["model_routed"] = which_model
        try:
            answer = run_inference_on_item(item, which_model)
            item["answer"] = answer
            item["model_used"] = which_model
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            oom_count += 1
            print(f"❌ CUDA OOM on frame {item.get('frame_token')} with {which_model}")
            if ENABLE_FALLBACK_72B_TO_7B and which_model == "72B":
                try:
                    answer = run_inference_on_item(item, "7B")
                    item["answer"] = answer
                    item["model_used"] = "7B_fallback"
                except Exception as e2:
                    item["answer"] = f"❌ Skipped due to CUDA OOM and fallback failed: {repr(e2)}"
                    item["model_used"] = which_model
            else:
                item["answer"] = f"❌ Skipped due to CUDA OOM: {repr(e)}"
                item["model_used"] = which_model
        except Exception as e:
            item["answer"] = f"❌ Error during inference: {repr(e)}"
            item["model_used"] = which_model
        output_data.append(item)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Inference finished, results saved to: {output_file}")
    print(f"⚠️ Samples skipped due to CUDA OOM: {oom_count}")

if __name__ == "__main__":
    main()
