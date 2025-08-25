import os, re, json, argparse
from PIL import Image
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# -------------------- CLI --------------------
parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B + LoRA inference (RoboSense Track1)")
parser.add_argument("--output_file", type=str, default="7B_LORA_results.json")
parser.add_argument("--json_file", type=str, default="robosense_track1_release_converted.json")
parser.add_argument("--image_base_dir", type=str, default="data/nuscenes")

# Base model & LoRA
parser.add_argument("--base_model", type=str,
                    default="/workspace/Qwen2.5-VL-7B-Instruct",
                    help="Can be an online model name, a local repo root, or a snapshots subdirectory")
parser.add_argument("--lora_dir", type=str,
                    default="/workspace/qwen7b_drivebench_lora_4gpus",
                    help="Can be a root directory or a specific checkpoint")

# Options
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--merge_lora", action="store_true",
                    help="Merge LoRA into the base model before inference (higher throughput, slightly higher memory)")
parser.add_argument("--max_images_per_prompt", type=int, default=6)
args = parser.parse_args()

# -------------------- utils --------------------
def resolve_local_hf_repo(path_or_name: str) -> str:
    """
    - If it's an online model name: return as is.
    - If it's a local directory:
        * If config.json exists -> use directly.
        * If it's a HF cache repo root (contains snapshots/), pick snapshots/<latest>.
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
    - If lora_dir contains adapter_model.safetensors -> use directly.
    - Otherwise, treat it as a root directory and select the latest checkpoint-xxxx.
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

SYSTEM_PROMPT = (
    "You are a helpful autonomous driving assistant that can answer questions about multi-view images.\n"
    "Views are ordered as [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT].\n"
    "Objects are referenced as <id, camera_view, x, y> where (x,y) are CENTER coordinates normalized to [0,1]\n"
    "for that camera's width/height. Do not assume a fixed pixel resolution."
)

# -------------------- load model --------------------
attn_impl = "sdpa"
try:
    import flash_attn  # noqa
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

# Processor: prefer LoRA directory if it has tokenizer/chat_template
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": img} for img in images] +
                                    [{"type": "text", "text": question}]},
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
