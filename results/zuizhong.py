import json
from pathlib import Path

# 输入文件路径
src_path = Path("7b(drivebench)_72b_V3_results.json")
tgt_path = Path("7B_LORA_results_override_final.json")

# 输出文件路径
out_path = Path("final_results.json")

# 读取源文件（包含 subtask == "Planning-VQAs-Scene-Description" 的答案）
with src_path.open("r", encoding="utf-8") as f:
    src = json.load(f)

# 读取目标文件（需要更新的）
with tgt_path.open("r", encoding="utf-8") as f:
    tgt = json.load(f)

# 三元键 (scene_token, frame_token, question)
KEY_FIELDS = ("scene_token", "frame_token", "question")
want_subtask = "Planning-VQAs-Scene-Description"

# 建立映射：key -> answer
src_map = {}
for item in src:
    if item.get("subtask") == want_subtask:
        key = tuple(item.get(k) for k in KEY_FIELDS)
        if None not in key:
            src_map[key] = item.get("answer")

# 遍历目标文件，匹配并替换 answer
updated = 0
for item in tgt:
    key = tuple(item.get(k) for k in KEY_FIELDS)
    if key in src_map:
        item["answer"] = src_map[key]
        updated += 1

# 保存结果
with out_path.open("w", encoding="utf-8") as f:
    json.dump(tgt, f, ensure_ascii=False, indent=2)

print(f"源文件匹配条数: {len(src_map)}")
print(f"更新条数: {updated}")
print(f"已保存到: {out_path}")