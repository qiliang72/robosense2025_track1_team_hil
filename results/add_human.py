import json

# 文件路径
anno_file = "annotations_export_20250815_031420.json"
results_file = "merged_results1.json"
output_file = "merged_results2.json"

# 读取人工标注文件
with open(anno_file, "r", encoding="utf-8") as f:
    anno_data = json.load(f)

# 读取原始结果文件
with open(results_file, "r", encoding="utf-8") as f:
    results_data = json.load(f)

# 建立 (scene_token, frame_token, question) -> answer 的映射
anno_map = {}
for item in anno_data.get("annotations", []):
    key = (item["scene_token"], item["frame_token"], item["question"])
    # 优先用 user_annotations.user_annotation，如果没有则用原 answer
    new_answer = item.get("user_annotations", {}).get("user_annotation") or item.get("answer")
    anno_map[key] = new_answer

# 遍历结果文件并替换 answer
updated_count = 0
for item in results_data:
    key = (item["scene_token"], item["frame_token"], item["question"])
    if key in anno_map:
        item["answer"] = anno_map[key]
        updated_count += 1

# 保存新文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)

print(f"替换完成，共更新 {updated_count} 条记录。")
print(f"已生成新文件：{output_file}")

