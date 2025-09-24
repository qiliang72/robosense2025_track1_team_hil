# Qwen2.5-VL 7B/72B LoRA Inference (DriveBench/RoboSense Track1)

This project provides an inference script designed for **multi-view autonomous driving QA/prediction tasks**. The script **automatically routes subtasks** to either **Qwen2.5-VL-7B** or **Qwen2.5-VL-72B**, with support for **LoRA adapters** and **RAG (external knowledge integration)**. The expected directory structure is:

```
qwen72b_drivebench_lora/      # LoRA weights for Qwen2.5-VL-72B (root or checkpoint-*)
qwen7b_drivebench_lora_dbg/   # LoRA weights for Qwen2.5-VL-7B (root or checkpoint-*)
results/                      # Directory to store inference result JSON files
scripts/                      # Directory containing inference scripts (e.g., this .py file)
```

---

## Features

- **Automatic Task Routing**: Based on `item.category` and question patterns:  
  - 7B: Perception-MCQs, Perception-VQAs-Object-Description, Prediction  
  - 72B: Perception-VQAs-Scene-Description, Planning-VQAs-Scene-Description, Planning-VQAs-Object-Description
- **Multi-image Input with Fixed Order**: Default camera order is  
  `[CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT]` (max `--max_images_per_prompt`).
- **Task-specific Prompts**: Controls output style and length depending on task (MCQs / Yes-No / Open-ended).
- **LoRA Support**: Loads LoRA adapters for both 7B and 72B from local directories (root or `checkpoint-*`).
- **Caching & Fallback**: Models cached by size; 72B can fallback to 7B in case of OOM.
- **RAG (Knowledge Base)**: Supports `off` / `hint` / `override` modes, injecting or overriding answers with human-annotated knowledge.
- **Repeatable & Deterministic Runs**: Controlled via `--limit`, `--seed`, and `--do_sample`.

---

## Requirements

- Python 3.9+
- PyTorch (GPU recommended, bf16/fp16)
- `transformers`, `peft`, `tqdm`, `Pillow`
- Qwen2.5-VL model + processor (`Qwen2_5_VLForConditionalGeneration`, `AutoProcessor`)
- Optional: FlashAttention-2 (`--use_flash_attn`)

> **Note**: Requires locally available 7B/72B base models and corresponding LoRA weights.

---

## Input JSON Format

The script reads from `--json_file` (list of samples). Example:

```json
{
  "question": "… (or use 'prompt')",
  "category": "perception / prediction / planning",
  "img_paths": {
    "CAM_FRONT": "xxx/front.jpg",
    "CAM_FRONT_RIGHT": "xxx/fr.jpg",
    "CAM_FRONT_LEFT": "xxx/fl.jpg",
    "CAM_BACK": "xxx/back.jpg",
    "CAM_BACK_RIGHT": "xxx/br.jpg",
    "CAM_BACK_LEFT": "xxx/bl.jpg"
  },
  "scene_token": "optional",
  "frame_token": "optional"
}
```

- Either `question` or `prompt` is required.  
- Image paths are joined with `--image_base_dir`. Missing images are skipped.  
- Routing is determined by `category` + question pattern.

---

## Directory & File Details

- **LoRA Directories**  
  - `qwen72b_drivebench_lora/`: LoRA for 72B. Script auto-selects latest `checkpoint-*`.  
  - `qwen7b_drivebench_lora_dbg/`: LoRA for 7B.
- **Output**  
  - `results/*.json`: Each run saves results with `answer`, `model_routed`, `model_used`.

---

## Quick Start

Directory structure:

```
.
├─ qwen72b_drivebench_lora/
├─ qwen7b_drivebench_lora_dbg/
├─ results/
└─ scripts/
   └─ 7b_72bcombin_v5_bate.py
```

Run:

```bash
cd scripts

python 7b_72bcombin_v5_bate.py \
  --json_file /path/to/robosense_track1_release_converted.json \
  --output_file ../results/run_$(date +%Y%m%d_%H%M%S).json \
  --image_base_dir /path/to/data/nuscenes \
  --model_7b /workspace/Qwen2.5-VL-7B-Instruct \
  --model_72b /workspace/Qwen2.5-VL-72B-Instruct \
  --use_lora_7b \
  --lora_7b_dir ../qwen7b_drivebench_lora_dbg \
  --use_lora_72b \
  --lora_72b_dir ../qwen72b_drivebench_lora \
  --dtype bf16 \
  --device_map auto \
  --use_flash_attn \
  --enable_fallback_72b_to_7b \
  --max_images_per_prompt 6 \
  --max_new_tokens_default 512 \
  --do_sample False \
  --kb_file /mnt/data/human_annotations.json \
  --rag_mode hint \
  --kb_topk 4
```

Results will be saved under `results/`.

---

## Key Arguments

- `--json_file`: Input dataset JSON.  
- `--output_file`: Output JSON file.  
- `--image_base_dir`: Base directory for images.  
- `--model_7b`, `--model_72b`: Base model paths or HF repos.  
- `--use_lora_7b`, `--use_lora_72b`: Enable LoRA adapters.  
- `--lora_7b_dir`, `--lora_72b_dir`: LoRA directories.  
- `--dtype {bf16, fp16}`: Precision.  
- `--device_map`: Device mapping (e.g. `auto`).  
- `--use_flash_attn`: Enable FlashAttention-2.  
- `--enable_fallback_72b_to_7b`: Use 7B if 72B OOM.  
- `--max_images_per_prompt`: Max number of images per input.  
- `--max_new_tokens_default`: Default max generation length.  
- `--do_sample`: Enable sampling.  
- `--limit`: Process first N entries only.  
- `--kb_file`: External knowledge JSON file.  
- `--rag_mode {off,hint,override}`: RAG mode.  
- `--kb_topk`: Number of KB entries injected.

---

## Subtasks & Answer Styles

- **Perception-MCQs**: Option + explanation.  
- **Yes/No questions**: Yes/No + reason.  
- **Prediction**: Movement/state + brief reason.  
- **Scene/Object-Description**: Summarized per prompt rules.

---

## Output Format

Each result item adds fields:

```json
{
  "... original fields ...": "...",
  "subtask": "auto-detected",
  "model_routed": "7B or 72B",
  "model_used": "actual model (with fallback info)",
  "answer": "generated text"
}
```

Errors or missing data are also recorded in `answer`.

---

## FAQ

1) **How should I provide LoRA directories?**  
Point to directories containing `adapter_model.safetensors` / `adapter_config.json`. If multiple checkpoints exist, latest is chosen.

2) **What if 72B runs OOM?**  
- Reduce `--max_images_per_prompt` / `--max_new_tokens_default`  
- Try disabling `--use_flash_attn`  
- Enable `--enable_fallback_72b_to_7b`  
- Adjust `--device_map`

3) **What format for RAG KB?**  
`--kb_file` JSON with `annotations` list. Each entry includes `question`, optionally `answer`, `scene_token`, `frame_token`.  
- `hint`: injects into context  
- `override`: replaces answer on exact match

---

## License

For research purposes only. Ensure proper licenses for models, datasets, and LoRA weights used.
