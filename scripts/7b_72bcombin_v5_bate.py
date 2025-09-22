# -*- coding: utf-8 -*-
"""
RoboSense Track1: Route tasks to Qwen2.5-VL-7B / 72B by subtask type and run inference.
RAG: external knowledge base support with hint/override modes

- 7B handles: Perception-MCQs, Perception-VQAs-Object-Description, Prediction
- 72B handles: Perception-VQAs-Scene-Description, Planning-VQAs-Scene-Description, Planning-VQAs-Object-Description
- Robust routing using item.category and question patterns
- Lazy loading and caching of models including LoRA to avoid frequent reloads
- Automatic fallback to 7B when 72B hits OOM optionally
- System prompt and task hints constrain output style and length
- Fixed camera order and per-sample image count limit
- RAG: retrieve from kb_file JSON; hint mode injects context, override mode returns KB answer on exact match
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from collections import defaultdict
import difflib
from PIL import Image

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Precompiled regex (case-insensitive)
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

# Subtask constants
SUB_PER_MCQS = "Perception-MCQs"
SUB_PER_VQA_OBJ = "Perception-VQAs-Object-Description"
SUB_PER_VQA_SCENE = "Perception-VQAs-Scene-Description"
SUB_PRED = "Prediction"
SUB_PLAN_VQA_SCENE = "Planning-VQAs-Scene-Description"
SUB_PLAN_VQA_OBJ = "Planning-VQAs-Object-Description"

# Model assignment
TASKS_7B = {SUB_PER_MCQS, SUB_PER_VQA_OBJ, SUB_PRED}
TASKS_72B = {SUB_PER_VQA_SCENE, SUB_PLAN_VQA_SCENE, SUB_PLAN_VQA_OBJ}

# Fixed camera order
CAM_ORDER = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"
]

# System prompt
SYSTEM_PROMPT = """You are a helpful autonomous driving assistant that can answer questions about images and videos. You are providing images from multi-view sensors ordered as [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT]. The object coordinates are provided in the format of <id, camera_view, x, y>. The coordinate is the center of the bounding box where the image resolution is 1600x900.
Instructions:
1. Answer Requirements:
• For multiple-choice questions, provide the selected answer choice along with an explanation.
• For “is” or “is not” questions, respond with a “Yes” or “No”, along with an explanation.
• For open-ended perception and prediction questions, related objects to which the camera.
2. Key Information for Driving Context:
• When answering, focus on object attributes (e.g., categories, statuses, visual descriptions) and motions (e.g., speed, action, acceleration) relevant to driving decision-making
Use the images and coordinate information to respond accurately to questions related to perception, prediction, planning, or behavior, based on the question requirements.
"""

def task_hint_for_subtask(subtask: str) -> str:
    if subtask in (SUB_PER_MCQS, SUB_PER_VQA_OBJ, SUB_PLAN_VQA_OBJ):
        return ""
    if subtask == SUB_PER_VQA_SCENE:
        return "TASK=Perception-VQAs-Scene-Description. List up to 5 important objects and give a one-sentence summary."
    if subtask == SUB_PRED:
        return "TASK=Prediction. Answer concisely whether the object will move in the given direction or change state (yes/no) and one short reason."
    if subtask == SUB_PLAN_VQA_SCENE:
        return "TASK=Planning-VQAs-Scene-Description. Give up to 3 safe actions and up to 3 dangerous actions for the scene, plus one short note."
    return ""

def max_tokens_for_subtask(subtask: str, default_tokens: int) -> int:
    if subtask in (SUB_PER_MCQS, SUB_PER_VQA_OBJ, SUB_PLAN_VQA_OBJ):
        return default_tokens
    if subtask == SUB_PRED:
        return 128
    if subtask == SUB_PER_VQA_SCENE:
        return 256
    if subtask == SUB_PLAN_VQA_SCENE:
        return 512
    return default_tokens

# LoRA and repo path helpers
def _find_latest_checkpoint(dir_path: str):
    """Find the subdirectory like checkpoint-1234 with the largest step."""
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

def _resolve_lora_dir(lora_dir: str):
    """
    If lora_dir itself contains adapter_model.safetensors and adapter_config.json, use it directly.
    Otherwise treat lora_dir as a root directory and pick the latest checkpoint-*.
    """
    if os.path.isdir(lora_dir):
        if (os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors"))
            and os.path.exists(os.path.join(lora_dir, "adapter_config.json"))):
            return lora_dir
        latest = _find_latest_checkpoint(lora_dir)
        if latest and os.path.exists(os.path.join(latest, "adapter_model.safetensors")):
            return latest
    return lora_dir

def resolve_local_hf_repo(path_or_name: str) -> str:
    """
    If it looks like an online HF repo id, return as-is.
    If it is a local directory: 
      if config.json exists at root, use it.
      if it is a HF cache root containing snapshots/, pick snapshots/<latest>.
    """
    if (os.path.sep not in path_or_name) and (not path_or_name.startswith(("hf://", "./", "../"))):
        return path_or_name
    path = path_or_name
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "config.json")):
            return path
        snaps_dir = os.path.join(path, "snapshots")
        if os.path.isdir(snaps_dir):
            snaps = sorted([d for d in os.listdir(snaps_dir) if os.path.isdir(os.path.join(snaps_dir, d))])
            if snaps:
                return os.path.join(snaps_dir, snaps[-1])
    return path_or_name

# Routing and message building with optional RAG context
def classify_subtask(item) -> str:
    """Classify subtask using category and question patterns."""
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
    """Return '7B' or '72B' and record subtask."""
    sub = classify_subtask(item)
    item["subtask"] = sub
    return "7B" if sub in TASKS_7B else "72B"

def load_images_in_order(image_paths_dict: dict, base_dir: str, max_images: int):
    """Load images in fixed camera order; limit to max_images."""
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

def build_messages_for_item(question: str, images, subtask: str, context_txt: str = ""):
    """
    Build messages with system prompt, optional RAG context, task hint, images, and question.
    RAG context, if any, is placed at the beginning of the user text content.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    content = []

    if context_txt:
        content.append({"type": "text", "text": context_txt})
        content.append({"type": "text", "text": "Use the above context if relevant. Keep answers concise and precise.\n"})

    hint = task_hint_for_subtask(subtask)
    if hint:
        content.append({"type": "text", "text": hint})

    content.extend([{"type": "image", "image": img} for img in images])
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})
    return messages

# Model cache
_model_cache = {}

def set_seed(seed: int):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_and_processor(which: str,
                            model_7b: str,
                            model_72b: str,
                            use_lora_7b: bool,
                            lora_7b_dir: str,
                            use_lora_72b: bool,
                            lora_72b_dir: str,
                            dtype: torch.dtype,
                            device_map: str,
                            use_flash_attn: bool):
    """Return (model, processor) with caching for '7B' or '72B'."""
    if which in _model_cache:
        return _model_cache[which]

    from peft import PeftModel

    repo_raw = model_7b if which == "7B" else model_72b
    repo = resolve_local_hf_repo(repo_raw)

    kwargs = dict(torch_dtype=dtype, device_map=device_map)
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo, **kwargs)

    proc_from = repo
    if which == "7B" and use_lora_7b:
        lora_repo = _resolve_lora_dir(lora_7b_dir)
        model = PeftModel.from_pretrained(model, lora_repo)
        model.eval()
        if (os.path.exists(os.path.join(lora_repo, "tokenizer_config.json"))
            or os.path.exists(os.path.join(lora_repo, "chat_template.json"))):
            proc_from = lora_repo

    if which == "72B" and use_lora_72b:
        lora_repo_72 = _resolve_lora_dir(lora_72b_dir)
        model = PeftModel.from_pretrained(model, lora_repo_72)
        model.eval()
        if (os.path.exists(os.path.join(lora_repo_72, "tokenizer_config.json"))
            or os.path.exists(os.path.join(lora_repo_72, "chat_template.json"))):
            proc_from = lora_repo_72

    processor = AutoProcessor.from_pretrained(proc_from)

    _model_cache[which] = (model, processor)
    return model, processor

# RAG utilities
def _norm_txt(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str) -> set:
    s = s.lower()
    s = re.sub(r"<c\d+\s*,\s*([A-Z_]+)\s*,\s*[\d\.]+\s*,\s*[\d\.]+\s*>", r"\1", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return set(t for t in s.split() if t)

@dataclass
class KBEntry:
    scene_token: str
    frame_token: str
    question: str
    answer: str
    raw: dict
    q_tokens: set
    qa_tokens: set

def load_kb(kb_path: str):
    if (not kb_path) or (not os.path.exists(kb_path)):
        print(f"[KB] file not found: {kb_path}")
        return [], {}, defaultdict(list)

    with open(kb_path, "r", encoding="utf-8") as f:
        kb_json = json.load(f)

    entries, exact_idx, scene_idx = [], {}, defaultdict(list)
    anns = kb_json.get("annotations") if isinstance(kb_json, dict) else kb_json
    for rec in anns:
        q = _norm_txt(rec.get("question", ""))
        a = _norm_txt(rec.get("user_annotations", {}).get("user_annotation") or rec.get("answer", ""))
        st = rec.get("scene_token") or ""
        ft = rec.get("frame_token") or ""
        if not q:
            continue
        e = KBEntry(
            scene_token=st, frame_token=ft, question=q, answer=a, raw=rec,
            q_tokens=_tokenize(q), qa_tokens=_tokenize(q + " " + a)
        )
        entries.append(e)
        exact_idx[(st, ft, q)] = e
        scene_idx[(st, ft)].append(e)
        scene_idx[(st, "")].append(e)
        scene_idx[("", "")].append(e)
    print(f"[KB] loaded {len(entries)} entries from {kb_path}")
    return entries, exact_idx, scene_idx

def _jaccard(a: set, b: set) -> float:
    if not a or not b: 
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / (union + 1e-9)

def _fuzzy(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def retrieve_kb(entries, exact_idx, scene_idx, item, topk=4):
    q = _norm_txt(item.get("question", "") or item.get("prompt", ""))
    st = item.get("scene_token") or ""
    ft = item.get("frame_token") or ""
    key = (st, ft, q)

    if key in exact_idx:
        return exact_idx[key], [exact_idx[key]]

    cands = []
    for bucket in [(st, ft), (st, ""), ("", "")]:
        if (bucket in scene_idx) and scene_idx[bucket]:
            cands = scene_idx[bucket]
            break
    q_tokens = _tokenize(q)
    scored = []
    for e in cands:
        s1 = _jaccard(q_tokens, e.q_tokens)
        s2 = _jaccard(q_tokens, e.qa_tokens)
        s3 = _fuzzy(q, e.question) * 0.5
        bonus = 0.2 if (e.frame_token == ft and ft) else (0.1 if (e.scene_token == st and st) else 0.0)
        score = 0.6*s1 + 0.3*s2 + 0.1*s3 + bonus
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return None, [e for _, e in scored[:topk]]

def build_context_text(item, kb_hits):
    """Build a compact structured context from KB hits."""
    if not kb_hits: 
        return ""
    lines = []
    lines.append("Context (human-verified notes):")
    for i, e in enumerate(kb_hits, 1):
        q = e.question
        a = e.answer
        loc = []
        if e.scene_token: 
            loc.append(f"scene={e.scene_token[:8]}")
        if e.frame_token: 
            loc.append(f"frame={e.frame_token[:8]}")
        tag = f" [{', '.join(loc)}]" if loc else ""
        if a:
            lines.append(f"{i}. {a}{tag}")
        else:
            lines.append(f"{i}. {q}{tag}")
    return "\n".join(lines)

# Single-item inference with RAG support
def run_inference_on_item(item,
                          which_model: str,
                          model_7b: str,
                          model_72b: str,
                          use_lora_7b: bool,
                          lora_7b_dir: str,
                          use_lora_72b: bool,
                          lora_72b_dir: str,
                          dtype: torch.dtype,
                          device_map: str,
                          use_flash_attn: bool,
                          image_base_dir: str,
                          max_images_per_prompt: int,
                          max_new_tokens_default: int,
                          do_sample: bool,
                          rag_mode: str = "off",
                          kb_entries=None,
                          kb_exact=None,
                          kb_scene=None,
                          kb_topk: int = 4):
    """Run inference for one item and return answer text."""
    model, processor = get_model_and_processor(
        which=which_model,
        model_7b=model_7b,
        model_72b=model_72b,
        use_lora_7b=use_lora_7b,
        lora_7b_dir=lora_7b_dir,
        use_lora_72b=use_lora_72b,
        lora_72b_dir=lora_72b_dir,
        dtype=dtype,
        device_map=device_map,
        use_flash_attn=use_flash_attn,
    )

    question = (item.get("question") or "").strip() or (item.get("prompt") or "").strip()
    image_paths_dict = item.get("img_paths", {}) or item.get("image_path", {})

    images = load_images_in_order(image_paths_dict, image_base_dir, max_images_per_prompt)

    if not images:
        return "❌ Skipped due to missing images."
    if not question:
        return "❌ Skipped due to missing question."

    context_txt = ""
    if rag_mode != "off" and kb_entries is not None:
        kb_exact_hit, kb_hits = retrieve_kb(kb_entries, kb_exact or {}, kb_scene or {}, item, topk=kb_topk)
        if rag_mode == "override" and kb_exact_hit and kb_exact_hit.answer:
            return kb_exact_hit.answer
        context_txt = build_context_text(item, kb_hits)

    subtask = item.get("subtask", "")
    messages = build_messages_for_item(question, images, subtask, context_txt=context_txt)

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

        max_new_tokens = max_tokens_for_subtask(subtask, max_new_tokens_default)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    return output_text

# CLI and main
def main():
    parser = argparse.ArgumentParser(description="Route tasks to Qwen2.5-VL-7B/72B and run inference with optional RAG.")
    parser.add_argument("--json_file", default="robosense_track1_release_converted.json", help="Input JSON file.")
    parser.add_argument("--output_file", default="7B_72B_V3_results.json", help="Output JSON file.")
    parser.add_argument("--image_base_dir", default="data/nuscenes", help="Root dir that contains images.")
    parser.add_argument("--max_images_per_prompt", type=int, default=6)

    parser.add_argument("--model_7b", default="/workspace/Qwen2.5-VL-7B-Instruct",
                        help="7B base model path or HF repo id")
    parser.add_argument("--model_72b", default="/workspace/Qwen2.5-VL-72B-Instruct",
                        help="72B base model path or HF repo id")

    parser.add_argument("--use_lora_7b", action="store_true", default=True, help="Enable LoRA for 7B")
    parser.add_argument("--lora_7b_dir", default="/workspace/qwen7b_drivebench_lora_dbg",
                        help="7B LoRA directory (root or a specific checkpoint)")
    parser.add_argument("--use_lora_72b", action="store_true", default=True, help="Enable LoRA for 72B")
    parser.add_argument("--lora_72b_dir", default="/workspace/qwen72b_drivebench_lora",
                        help="72B LoRA directory (root or a specific checkpoint)")

    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device_map", default="auto", help="e.g., 'auto' or an explicit mapping")

    parser.add_argument("--use_flash_attn", action="store_true", default=True,
                        help="Use FlashAttention-2 if available")
    parser.add_argument("--enable_fallback_72b_to_7b", action="store_true", default=True,
                        help="Fallback to 7B when 72B hits CUDA OOM")

    parser.add_argument("--max_new_tokens_default", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=888)

    parser.add_argument("--limit", type=int, default=0, help="If >0, only process the first N samples")

    parser.add_argument("--kb_file", type=str, default="/mnt/data/human_annotations.json",
                        help="External KB JSON file with human annotations or rules")
    parser.add_argument("--rag_mode", type=str, default="hint",
                        choices=["off", "hint", "override"],
                        help="off: no RAG; hint: inject KB context; override: return KB answer on exact match")
    parser.add_argument("--kb_topk", type=int, default=4, help="Max KB evidence entries to inject in context")

    args = parser.parse_args()

    DTYPE = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    set_seed(args.seed if args.do_sample else None)

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    kb_entries, kb_exact, kb_scene = load_kb(args.kb_file)

    output_data = []
    oom_count = 0

    for idx, item in tqdm(enumerate(data), total=len(data), desc="Running Inference"):
        which_model = route_model(item)
        item["model_routed"] = which_model

        try:
            answer = run_inference_on_item(
                item=item,
                which_model=which_model,
                model_7b=args.model_7b,
                model_72b=args.model_72b,
                use_lora_7b=args.use_lora_7b,
                lora_7b_dir=args.lora_7b_dir,
                use_lora_72b=args.use_lora_72b,
                lora_72b_dir=args.lora_72b_dir,
                dtype=DTYPE,
                device_map=args.device_map,
                use_flash_attn=args.use_flash_attn,
                image_base_dir=args.image_base_dir,
                max_images_per_prompt=args.max_images_per_prompt,
                max_new_tokens_default=args.max_new_tokens_default,
                do_sample=args.do_sample,
                rag_mode=args.rag_mode,
                kb_entries=kb_entries,
                kb_exact=kb_exact,
                kb_scene=kb_scene,
                kb_topk=args.kb_topk,
            )
            item["answer"] = answer
            item["model_used"] = which_model

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            oom_count += 1
            print(f"❌ CUDA OOM on frame {item.get('frame_token')} with {which_model}")

            if args.enable_fallback_72b_to_7b and which_model == "72B":
                try:
                    answer = run_inference_on_item(
                        item=item,
                        which_model="7B",
                        model_7b=args.model_7b,
                        model_72b=args.model_72b,
                        use_lora_7b=args.use_lora_7b,
                        lora_7b_dir=args.lora_7b_dir,
                        use_lora_72b=args.use_lora_72b,
                        lora_72b_dir=args.lora_72b_dir,
                        dtype=DTYPE,
                        device_map=args.device_map,
                        use_flash_attn=args.use_flash_attn,
                        image_base_dir=args.image_base_dir,
                        max_images_per_prompt=args.max_images_per_prompt,
                        max_new_tokens_default=args.max_new_tokens_default,
                        do_sample=args.do_sample,
                        rag_mode=args.rag_mode,
                        kb_entries=kb_entries,
                        kb_exact=kb_exact,
                        kb_scene=kb_scene,
                        kb_topk=args.kb_topk,
                    )
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

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Inference finished. Results saved to: {args.output_file}")
    print(f"⚠️ Number of samples skipped due to CUDA OOM: {oom_count}")

if __name__ == "__main__":
    main()
