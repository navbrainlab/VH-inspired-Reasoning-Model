"""
从原始向量数据中筛选压缩后的训练数据

重要说明：
  - 本脚本不重新聚类，直接从原始数据中筛选
  - 根据 clustering_summary_compressed.json 确定需要保留哪些 (premise, cluster_id) 组合
  - 从 causal_A_vectors.npy, causal_B_vectors.npy, causal_metadata.json 中筛选对应的样本

输入文件（由 causal_data_processing.py 生成）：
  - causal_A_vectors.npy: 原始 A 向量
  - causal_B_vectors.npy: 原始 B 向量
  - causal_metadata.json: 原始元数据

输入文件（由 compress_clusters.py 生成）：
  - clustering_summary_compressed.json: 压缩后的聚类摘要

输出文件：
  - causal_A_vectors_compressed.npy: 压缩后的 A 向量
  - causal_B_vectors_compressed.npy: 压缩后的 B 向量
  - causal_metadata_compressed.json: 压缩后的元数据

使用方法：
1. 先运行 causal_data_processing.py 生成原始数据
2. 运行 compress_clusters.py 压缩聚类摘要
3. 运行本脚本筛选向量数据
"""

import os
import json
import numpy as np
from collections import defaultdict

# ================== 配置参数 ==================
BASE_DIR = "/home/amax/Zixing_Jia/Vector-HASH-tinkering-main/Gemma_anwer_causal/Causal_Data"

# 输入文件
ORIGINAL_A_VECTORS = os.path.join(BASE_DIR, "causal_A_vectors.npy")
ORIGINAL_B_VECTORS = os.path.join(BASE_DIR, "causal_B_vectors.npy")
ORIGINAL_METADATA = os.path.join(BASE_DIR, "causal_metadata.json")
COMPRESSED_SUMMARY = os.path.join(BASE_DIR, "clustering_summary_compressed.json")

# 输出文件
OUTPUT_A_VECTORS = os.path.join(BASE_DIR, "causal_A_vectors_compressed.npy")
OUTPUT_B_VECTORS = os.path.join(BASE_DIR, "causal_B_vectors_compressed.npy")
OUTPUT_METADATA = os.path.join(BASE_DIR, "causal_metadata_compressed.json")

# ================== Step 1: 加载原始数据 ==================
print("=" * 60)
print("Step 1: 加载原始向量数据和元数据")
print("=" * 60)

print(f"📁 加载 A 向量: {ORIGINAL_A_VECTORS}")
A_vectors = np.load(ORIGINAL_A_VECTORS)
print(f"   Shape: {A_vectors.shape}")

print(f"📁 加载 B 向量: {ORIGINAL_B_VECTORS}")
B_vectors = np.load(ORIGINAL_B_VECTORS)
print(f"   Shape: {B_vectors.shape}")

print(f"📁 加载元数据: {ORIGINAL_METADATA}")
with open(ORIGINAL_METADATA, 'r', encoding='utf-8') as f:
    metadata = json.load(f)
print(f"   样本数: {len(metadata)}")

# 验证数据一致性
assert A_vectors.shape[0] == len(metadata), "A 向量数量与元数据不匹配！"
assert B_vectors.shape[0] == len(metadata), "B 向量数量与元数据不匹配！"
print("✅ 数据一致性验证通过")

# ================== Step 2: 加载压缩后的聚类摘要 ==================
print("\n" + "=" * 60)
print("Step 2: 加载压缩后的聚类摘要")
print("=" * 60)

print(f"📁 加载压缩后的聚类摘要: {COMPRESSED_SUMMARY}")
with open(COMPRESSED_SUMMARY, 'r', encoding='utf-8') as f:
    compressed_summary = json.load(f)
print(f"   Premise 数: {len(compressed_summary)}")

# 构建有效的 (premise, cluster_id) 集合
valid_pairs = set()
for premise, data in compressed_summary.items():
    for cluster_id in data['clusters'].keys():
        valid_pairs.add((premise, cluster_id))

total_clusters = sum(d['n_clusters'] for d in compressed_summary.values())
print(f"   总 Cluster 数: {total_clusters}")
print(f"   有效 (premise, cluster_id) 对数: {len(valid_pairs)}")

# ================== Step 3: 筛选数据 ==================
print("\n" + "=" * 60)
print("Step 3: 筛选压缩后的数据")
print("=" * 60)

# 找出需要保留的样本索引
keep_indices = []
for i, m in enumerate(metadata):
    premise = m['premise']
    cluster_id = str(m['cluster_id'])  # 注意：metadata 中是 int，summary 中是 str
    
    if (premise, cluster_id) in valid_pairs:
        keep_indices.append(i)

print(f"原始样本数: {len(metadata)}")
print(f"保留样本数: {len(keep_indices)}")
print(f"移除样本数: {len(metadata) - len(keep_indices)}")

# 筛选向量和元数据
compressed_A = A_vectors[keep_indices]
compressed_B = B_vectors[keep_indices]
compressed_metadata = [metadata[i] for i in keep_indices]

print(f"\n压缩后 A 向量 Shape: {compressed_A.shape}")
print(f"压缩后 B 向量 Shape: {compressed_B.shape}")
print(f"压缩后元数据数量: {len(compressed_metadata)}")

# ================== Step 4: 验证数据一致性 ==================
print("\n" + "=" * 60)
print("Step 4: 验证压缩后的数据")
print("=" * 60)

# 统计 ASKFOR 分布
askfor_counts = defaultdict(int)
for m in compressed_metadata:
    askfor_counts[m['ASKFOR']] += 1
print(f"ASKFOR 分布: {dict(askfor_counts)}")

# 统计唯一的 INDEX 数量
unique_indices = set(m['INDEX'] for m in compressed_metadata)
print(f"唯一 INDEX 数量: {len(unique_indices)}")

# 统计唯一的 premise 数量
unique_premises = set(m['premise'] for m in compressed_metadata)
print(f"唯一 Premise 数量: {len(unique_premises)}")

# 验证与压缩摘要的一致性
# 检查每个 (premise, cluster_id) 的样本数是否与 compressed_summary 中的 count 一致
pair_counts = defaultdict(int)
for m in compressed_metadata:
    key = (m['premise'], str(m['cluster_id']))
    pair_counts[key] += 1

mismatches = []
for premise, data in compressed_summary.items():
    for cluster_id, cluster_info in data['clusters'].items():
        expected_count = cluster_info['count']
        actual_count = pair_counts.get((premise, cluster_id), 0)
        if expected_count != actual_count:
            mismatches.append({
                'premise': premise[:50] + '...',
                'cluster_id': cluster_id,
                'expected': expected_count,
                'actual': actual_count
            })

if mismatches:
    print(f"\n⚠️ 发现 {len(mismatches)} 个不一致:")
    for m in mismatches[:5]:
        print(f"   - {m}")
else:
    print("✅ 所有 (premise, cluster_id) 的样本数与压缩摘要一致")

# ================== Step 5: 保存压缩后的数据 ==================
print("\n" + "=" * 60)
print("Step 5: 保存压缩后的数据")
print("=" * 60)

# 保存向量
np.save(OUTPUT_A_VECTORS, compressed_A)
print(f"✅ 保存 A 向量: {OUTPUT_A_VECTORS}")

np.save(OUTPUT_B_VECTORS, compressed_B)
print(f"✅ 保存 B 向量: {OUTPUT_B_VECTORS}")

# 保存元数据
with open(OUTPUT_METADATA, 'w', encoding='utf-8') as f:
    json.dump(compressed_metadata, f, ensure_ascii=False, indent=2)
print(f"✅ 保存元数据: {OUTPUT_METADATA}")

# ================== Step 6: 最终验证 ==================
print("\n" + "=" * 60)
print("Step 6: 最终验证")
print("=" * 60)

# 重新加载并验证
A_loaded = np.load(OUTPUT_A_VECTORS)
B_loaded = np.load(OUTPUT_B_VECTORS)
with open(OUTPUT_METADATA, 'r', encoding='utf-8') as f:
    meta_loaded = json.load(f)

print(f"A 向量 Shape: {A_loaded.shape}")
print(f"B 向量 Shape: {B_loaded.shape}")
print(f"元数据数量: {len(meta_loaded)}")

assert A_loaded.shape[0] == len(meta_loaded), "A 向量数量与元数据不匹配！"
assert B_loaded.shape[0] == len(meta_loaded), "B 向量数量与元数据不匹配！"

print("\n✅ 压缩数据生成完成！")

# 输出摘要
print("\n" + "=" * 60)
print("压缩摘要")
print("=" * 60)
print(f"{'指标':<25} {'压缩前':>12} {'压缩后':>12}")
print("-" * 49)
print(f"{'样本数':<25} {len(metadata):>12} {len(meta_loaded):>12}")
print(f"{'唯一 Premise 数':<25} {len(set(m['premise'] for m in metadata)):>12} {len(unique_premises):>12}")
print(f"{'向量维度':<25} {A_vectors.shape[1]:>12} {A_loaded.shape[1]:>12}")
