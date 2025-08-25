import os, json, argparse
from typing import List, Dict
from PIL import Image, ImageFile

# Avoid crash from a few truncated or corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from inspect import signature

# Enable TF32 (significant speedup on H100/A100, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============== Dataset and Collator ==============
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

class SFTJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = list(read_jsonl(jsonl_path))
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

def build_collator(processor, max_len: int):
    tok = processor.tokenizer

    def collate(batch):
        texts, image_batches, assistant_texts_toklens, metas = [], [], [], []
        for ex in batch:
            msgs = ex["messages"]

            # Find the last assistant message to prepare prefix
            last_ass_idx = None
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i]["role"] == "assistant":
                    last_ass_idx = i
                    break

            if last_ass_idx is None:
                msgs_prefix = msgs
                ass_text = ""
            else:
                msgs_prefix = msgs[:last_ass_idx]
                ass_text = msgs[last_ass_idx]["content"]

            # Use chat_template to build prefix text (includes all <image> placeholders)
            prefix_text = processor.apply_chat_template(
                msgs_prefix, tokenize=False, add_generation_prompt=False
            )
            prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
            prefix_len = len(prefix_ids)

            # Budget = max_len - prefix length
            budget = max_len - prefix_len
            if budget < 0:
                truncated_ass_text = ""
                ass_ids = []
            else:
                ass_ids_full = tok(ass_text, add_special_tokens=False).input_ids
                # Truncate only assistant part (keep the last `budget` tokens)
                if len(ass_ids_full) > budget:
                    ass_ids = ass_ids_full[-budget:]
                else:
                    ass_ids = ass_ids_full
                truncated_ass_text = tok.decode(ass_ids, skip_special_tokens=False)

            # Reassemble final messages
            msgs_final = msgs_prefix + [{"role": "assistant", "content": truncated_ass_text}]

            # Build final text (now length <= max_len)
            text = processor.apply_chat_template(
                msgs_final, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            # Collect user images
            images = []
            for m in msgs_final:
                if m["role"] == "user" and isinstance(m["content"], list):
                    for c in m["content"]:
                        if c.get("type") == "image":
                            try:
                                images.append(Image.open(c["path"]).convert("RGB"))
                            except Exception as e:
                                print(f"[WARN] failed to open image {c['path']}: {e}")
            image_batches.append(images)

            assistant_texts_toklens.append(len(ass_ids))
            metas.append(ex.get("meta", {}))

        # Do not let processor truncate again here
        enc = processor(
            text=texts,
            images=image_batches,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        labels = input_ids.clone().fill_(-100)

        # Mark labels only for the last assistant tokens
        B, L = labels.shape
        for i in range(B):
            n = min(assistant_texts_toklens[i], L)
            if n > 0:
                labels[i, L - n:] = input_ids[i, L - n:]

        enc["labels"] = labels
        return enc

    return collate


# ============== Main ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train_file", default="train.jsonl")
    ap.add_argument("--eval_file", default="dev.jsonl")
    ap.add_argument("--out_dir", default="qwen7b_drivebench_lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)    # per GPU batch size
    ap.add_argument("--eval_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_seq_len", type=int, default=3072)
    ap.add_argument("--use_flash_attn", action="store_true",
                    help="try FlashAttention-2 (fallback to SDPA if unavailable)")
    ap.add_argument("--qlora", action="store_true", help="use 4-bit QLoRA")

    # Resume training options
    ap.add_argument("--resume_adapter", type=str, default=None,
                    help="path to existing LoRA folder (e.g., qwen7b_drivebench_lora_dbg)")
    ap.add_argument("--resume_ckpt", type=str, default=None,
                    help="path to checkpoint dir to resume Trainer state (e.g., .../checkpoint-1471)")

    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank != -1:
        # For distributed training, bind to corresponding GPU first
        torch.cuda.set_device(local_rank)

    # FlashAttention fallback
    attn_impl = "sdpa"
    if args.use_flash_attn:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            if local_rank in (-1, 0):
                print("Using FlashAttention-2.")
        except Exception:
            if local_rank in (-1, 0):
                print("flash-attn not installed; falling back to SDPA.")
            attn_impl = "sdpa"

    # Load model
    kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
    if local_rank == -1:
        # Single-process can use device_map=auto; in distributed mode, must not use device_map
        kwargs["device_map"] = "auto"

    if args.qlora:
        kwargs.update(dict(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ))

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kwargs)

    # Prefer processor from resume_adapter (to keep chat_template/added_tokens consistent)
    if args.resume_adapter and os.path.isdir(args.resume_adapter):
        try:
            processor = AutoProcessor.from_pretrained(args.resume_adapter, use_fast=False)
            if local_rank in (-1, 0):
                print(f"[Processor] Loaded from adapter dir: {args.resume_adapter}")
        except Exception:
            processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
            if local_rank in (-1, 0):
                print("[Processor] Fallback to base model processor.")
    else:
        processor = AutoProcessor.from_pretrained(args.model, use_fast=False)

    # Disable caching during training; allow grads to flow for checkpointing
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Unfreeze multimodal projector layers (other weights use LoRA)
    for n, p in model.named_parameters():
        if "mm_projector" in n or "multi_modal_projector" in n:
            p.requires_grad = True

    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    # Load or create LoRA
    if args.resume_adapter and os.path.isdir(args.resume_adapter):
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        if local_rank in (-1, 0):
            print(f"[LoRA] Loaded existing adapter from: {args.resume_adapter}")
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        )
        model = get_peft_model(model, lora_cfg)
        if local_rank in (-1, 0):
            print("[LoRA] Created new adapter with current hyperparameters.")

    # Print trainable params
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    if local_rank in (-1, 0):
        print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")
    assert trainable > 0, "No trainable parameters found! Check LoRA target_modules."

    # Dataset
    train_ds = SFTJsonlDataset(args.train_file)
    eval_ds = SFTJsonlDataset(args.eval_file) if os.path.exists(args.eval_file) else None
    collate_fn = build_collator(processor, args.max_seq_len)

    # Compatibility with older transformers: evaluation_strategy optional
    eval_kwargs = {}
    try:
        if "evaluation_strategy" in signature(TrainingArguments.__init__).parameters and eval_ds:
            eval_kwargs = dict(evaluation_strategy="steps", eval_steps=500)
    except Exception:
        pass

    # Training arguments (stable for /dev/shm and distributed)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        gradient_checkpointing=True,   # requires enable_input_require_grads
        remove_unused_columns=False,   # must be False for multimodal
        dataloader_num_workers=0,      # safer for /dev/shm
        dataloader_pin_memory=False,   # more stable
        seed=args.seed,
        **eval_kwargs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
    )

    # Training: resume from checkpoint if provided
    if args.resume_ckpt and os.path.isdir(args.resume_ckpt):
        if local_rank in (-1, 0):
            print(f"[Resume] Resuming Trainer state from: {args.resume_ckpt}")
        trainer.train(resume_from_checkpoint=args.resume_ckpt)
    else:
        # Can also use resume_from_checkpoint=True to automatically pick latest
        trainer.train()

    # Final evaluation (in case not run online)
    if eval_ds and local_rank in (-1, 0):
        try:
            metrics = trainer.evaluate()
            print("Eval metrics:", metrics)
        except Exception as e:
            print("Eval skipped due to:", repr(e))

    # Save only on rank 0
    if local_rank in (-1, 0):
        model.save_pretrained(args.out_dir)
        processor.save_pretrained(args.out_dir)
        print(f"Done. LoRA adapter saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
