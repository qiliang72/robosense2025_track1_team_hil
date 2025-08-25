import os, re, json, argparse, difflib
from dataclasses import dataclass
from collections import defaultdict
from PIL import Image
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# -------------------- CLI --------------------
parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B + LoRA inference (RoboSense Track1) with RAG")
parser.add_argument("--output_file", type=str, default="7B_LORA_results.json")
parser.add_argument("--json_file", type=str, default="robosense_track1_release_converted.json")
parser.add_argument("--image_base_dir", type=str, default="data/nuscenes")

# Base model & LoRA
parser.add_argument("--base_model", type=str,
                    default="/workspace/Qwen2.5-VL-7B-Instruct",
                    help="Support online model name / local repo root / local snapshots subdirectory")
parser.add_argument("--lora_dir", type=str,
                    default="/workspace/qwen7b_drivebench_lora_4gpus",
                    help="Support root directory or specific checkpoint")

# Generation options
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--merge_lora", action="store_true",
                    help="Merge LoRA into base model before inference (higher throughput, slightly more memory)")
parser.add_argument("--max_images_per_prompt", type=int, default=6)

# RAG options
parser.add_argument("--kb_file", type=str, default="/mnt/data/human_annotations.json",
                    help="External knowledge base JSON (human annotations / rules / notes, etc.)")
parser.add_argument("--rag_mode", type=str, default="hint",
                    choices=["off", "hint", "override"],
                    help="RAG mode: off=disabled; hint=add context hint; override=use KB answer directly if exact match")
parser.add_argument("--kb_topk", type=int, default=4,
                    help="Maximum number of KB evidence entries to concatenate per sample")
args = parser.parse_args()

# -------------------- utils --------------------
def resolve_local_hf_repo(path_or_name: str) -> str:
    """
    - If online model name: return directly.
    - If local directory:
        * If config.json exists: use directly.
        * If it's a HF cache repo root (contains snapshots/), jump to snapshots/<latest>.
    """
    if (os.path.sep not in path_or_name) and (not path_or_name.startswith(("hf://", "./", "../"))):
        return path_or_name

    p = path_or_name
    if os.path.isdir(p):
        if os.path.exists(os.path.join(p, "config.json")):
            return p
        snaps = os.path.join(p, "snapshots")
        if os.path.isdir(snaps):
            subdirs = [d for d in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, d))]
            if subdirs:
                subdirs.sort()
                return os.path.join(snaps, subdirs[-1])
    return path_or_name

def find_latest_checkpoint(lora_root: str) -> str | None:
    """Find the checkpoint-xxxx subdirectory with the largest step under lora_root."""
    if not os.path.isdir(lora_root):
        return None
    pat = re.compile(r"^checkpoint-(\d+)$")
    best, best_step = None, -1
    for name in os.listdir(lora_root):
        m = pat.match(name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step, best = step, os.path.join(lora_root, name)
    return best

def resolve_lora_dir(lora_dir: str) -> str:
    """
    - If lora_dir itself contains adapter_model.safetensors -> return it.
    - Otherwise treat lora_dir as root dir, select the latest checkpoint-xxxx automatically.
    """
    if os.path.isdir(lora_dir):
        if (os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors"))
            and os.path.exists(os.path.join(lora_dir, "adapter_config.json"))):
            return lora_dir
        latest = find_latest_checkpoint(lora_dir)
        if latest and os.path.exists(os.path.join(latest, "adapter_model.safetensors")):
            return latest
    return lora_dir

# Fixed camera order
CAM_ORDER = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"
]

SYSTEM_PROMPT = """You are a helpful autonomous driving assistant that can answer questions about images and videos. 
You are providing images from multi-view sensors ordered as [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT]. 
The object coordinates are provided in the format of <id, camera_view, x, y>. 
The coordinate is the center of the bounding box where the image resolution is 1600x900.
"""

# -------------------- KB (RAG) --------------------
def _norm_txt(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str) -> set:
    s = s.lower()
    # Remove numeric details inside <...>, keep ID/camera only to reduce noise
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
    anns = kb_json.get("annotations") if isinstance(kb_json, dict) else kb_json  # support list structure
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
        scene_idx[(st, "")].append(e)     # same scene
        scene_idx[("", "")].append(e)     # global fallback
    print(f"[KB] loaded {len(entries)} entries from {kb_path}")
    return entries, exact_idx, scene_idx

def _jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / (union + 1e-9)

def _fuzzy(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def retrieve_kb(entries, exact_idx, scene_idx, item, topk=4):
    q = _norm_txt(item.get("question", "") or item.get("prompt", ""))
    st = item.get("scene_token") or ""
    ft = item.get("frame_token") or ""
    key = (st, ft, q)

    # 1) Exact match
    if key in exact_idx:
        return exact_idx[key], [exact_idx[key]]

    # 2) Lightweight retrieval (same frame / same scene / global)
    cands = []
    for bucket in [(st, ft), (st, ""), ("", "")]:
        if (bucket in scene_idx) and scene_idx[bucket]:
            cands = scene_idx[bucket]; break
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
    """Concatenate KB evidence into structured context, keeping it concise."""
    if not kb_hits: return ""
    lines = []
    lines.append("Context (human-verified notes):")
    for i, e in enumerate(kb_hits, 1):
        q = e.question
        a = e.answer
        loc = []
        if e.scene_token: loc.append(f"scene={e.scene_token[:8]}")
        if e.frame_token: loc.append(f"frame={e.frame_token[:8]}")
        tag = f" [{', '.join(loc)}]" if loc else ""
        if a:
            lines.append(f"{i}. {a}{tag}")
        else:
            lines.append(f"{i}. {q}{tag}")
    return "\n".join(lines)

# -------------------- load model --------------------
attn_impl = "sdpa"
try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
    print("Using FlashAttention-2.")
except Exception:
    print("flash-attn not installed; falling back to SDPA.")

base_repo = resolve_local_hf_repo(args.base_model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_repo,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl,
    device_map="auto"
)

lora_path = resolve_lora_dir(args.lora_dir)
print(f"[LoRA] using: {lora_path}")
model = PeftModel.from_pretrained(model, lora_path)
if args.merge_lora:
    print("[LoRA] merge_and_unload() for faster inference")
    model = model.merge_and_unload()
model.eval()

# Processor: prefer LoRA dir if it has tokenizer/chat_template
proc_from = lora_path if (os.path.exists(os.path.join(lora_path, "tokenizer_config.json"))
                          or os.path.exists(os.path.join(lora_path, "chat_template.json"))) else base_repo
processor = AutoProcessor.from_pretrained(proc_from, use_fast=False)

# -------------------- data --------------------
with open(args.json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

def load_images_in_order(paths_dict: dict):
    imgs = []
    if not paths_dict:
        return imgs
    for cam in CAM_ORDER:
        if cam not in paths_dict:
            continue
        rel = paths_dict[cam]
        p = os.path.join(args.image_base_dir, rel)
        if os.path.exists(p):
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"❌ Failed to load image: {p} -> {e}")
        else:
            print(f"⚠️ Image not found: {p}")
        if len(imgs) >= args.max_images_per_prompt:
            break
    return imgs

# -------------------- init KB --------------------
kb_entries, kb_exact, kb_scene = load_kb(args.kb_file)

# -------------------- inference --------------------
output_data, oom_count = [], 0

for idx, item in tqdm(enumerate(data), total=len(data), desc="Running Inference"):
    question = item.get("question", "") or item.get("prompt", "")
    image_paths = item.get("img_paths") or item.get("image_path") or {}
    images = load_images_in_order(image_paths)

    if not images or not question:
        item["answer"] = "❌ Skipped due to missing images or question."
        output_data.append(item)
        continue

    # ===== RAG begin =====
    context_txt = ""
    if args.rag_mode != "off":
        kb_exact_hit, kb_hits = retrieve_kb(kb_entries, kb_exact, kb_scene, item, topk=args.kb_topk)

        # override: if exact match, directly use KB answer
        if args.rag_mode == "override" and kb_exact_hit and kb_exact_hit.answer:
            item["answer"] = kb_exact_hit.answer
            output_data.append(item)
            continue

        # hint: concatenate KB evidence into "Context"
        context_txt = build_context_text(item, kb_hits)
    # ===== RAG end =====

    # Build dialogue: system prompt + context + question
    user_text_blocks = []
    if context_txt:
        user_text_blocks.append({"type": "text", "text": context_txt})
        user_text_blocks.append({"type": "text", "text": "Use the above context if relevant. Keep answers concise and precise.\n"})
    user_text_blocks += [{"type": "text", "text": question}]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": img} for img in images] + user_text_blocks},
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[images], padding=True, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample
            )
        gen_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        output_text = processor.batch_decode(
            gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        item["answer"] = output_text

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        oom_count += 1
        item["answer"] = "❌ Skipped due to CUDA OOM."
    except Exception as e:
        item["answer"] = f"❌ Error during inference: {repr(e)}"

    output_data.append(item)

with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Inference finished, results saved to: {args.output_file}")
print(f"⚠️ Samples skipped due to CUDA OOM: {oom_count}")
