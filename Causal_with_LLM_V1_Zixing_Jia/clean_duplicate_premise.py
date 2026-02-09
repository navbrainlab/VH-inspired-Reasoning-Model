# ===== 清理重复 premise 对应多个 index 的记录 =====
# 规则：同一个 premise 只保留最小的 index 对应的所有回答记录
# 默认输出新文件，可选 --inplace 覆盖原文件（会备份 .bak）

import argparse
import json
import os
import re
from typing import Dict, List, Tuple


def index_key(index_value: str) -> Tuple[int, str]:
    """
    解析 index 后缀数字作为比较键。
    若无法解析数字，退回字符串比较。
    """
    match = re.search(r"(\d+)$", index_value)
    if match:
        return (int(match.group(1)), index_value)
    return (10**18, index_value)


def load_records(input_path: str) -> List[dict]:
    records: List[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_min_index_by_premise(records: List[dict]) -> Dict[str, str]:
    min_index_by_premise: Dict[str, str] = {}
    for r in records:
        premise = r["premise"]
        idx = r["index"]
        if premise not in min_index_by_premise:
            min_index_by_premise[premise] = idx
            continue
        current = min_index_by_premise[premise]
        if index_key(idx) < index_key(current):
            min_index_by_premise[premise] = idx
    return min_index_by_premise


def filter_records(records: List[dict], min_index_by_premise: Dict[str, str]) -> List[dict]:
    filtered: List[dict] = []
    for r in records:
        premise = r["premise"]
        idx = r["index"]
        if min_index_by_premise.get(premise) == idx:
            filtered.append(r)
    return filtered


def write_records(output_path: str, records: List[dict]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove duplicate premise entries by keeping the smallest index.")
    parser.add_argument(
        "--input",
        default="/home/amax/Zixing_Jia/Vector-HASH-tinkering-main/Gemma_anwer_causal/T_1.25_Gemma_Answer/gemma_causal_answers_100.jsonl",
        help="Path to input jsonl file.",
    )
    parser.add_argument(
        "--output",
        default="/home/amax/Zixing_Jia/Vector-HASH-tinkering-main/Gemma_anwer_causal/T_1.25_Gemma_Answer/gemma_causal_answers_100_dedup.jsonl",
        help="Path to output jsonl file.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file (creates a .bak backup).",
    )

    args = parser.parse_args()

    records = load_records(args.input)
    min_index_by_premise = build_min_index_by_premise(records)
    filtered = filter_records(records, min_index_by_premise)

    removed = len(records) - len(filtered)

    if args.inplace:
        backup_path = args.input + ".bak"
        tmp_output = args.input + ".tmp"
        write_records(tmp_output, filtered)
        os.replace(args.input, backup_path)
        os.replace(tmp_output, args.input)
        output_path = args.input
    else:
        write_records(args.output, filtered)
        output_path = args.output

    print("=" * 60)
    print("Dedup completed")
    print(f"Input records : {len(records)}")
    print(f"Output records: {len(filtered)}")
    print(f"Removed        : {removed}")
    print(f"Output file    : {output_path}")
    if args.inplace:
        print(f"Backup file    : {backup_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
