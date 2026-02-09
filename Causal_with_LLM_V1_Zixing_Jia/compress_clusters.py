"""
压缩 Cluster 数量，满足以下约束：

约束条件：
1. 每个 Premise 的 Cluster 数量 <= 7
2. Premise 数量 + 总 Cluster 数量 <= 4096
3. 去除重复的 Premise（相同 premise 文本只保留一个）

输入：clustering_summary.json (由 causal_data_processing.py 生成)
输出：clustering_summary_compressed.json (压缩后的聚类摘要)

使用方法：
1. 先运行 causal_data_processing.py 生成原始数据
2. 运行本脚本进行压缩
3. 运行 generate_compressed_data.py 从原始向量数据中筛选
"""

import os
import json
import copy
from collections import defaultdict

# ================== 配置参数 ==================
BASE_DIR = "/home/amax/Zixing_Jia/Vector-HASH-tinkering-main/Gemma_anwer_causal/Causal_Data"
INPUT_FILE = os.path.join(BASE_DIR, "clustering_summary.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "clustering_summary_compressed.json")

# 约束参数
MAX_CLUSTERS_PER_PREMISE = 7
MAX_TOTAL_POSITIONS = 4096  # Premise 数量 + Cluster 数量的上限

# ================== Step 0: 加载数据 ==================
print("=" * 60)
print("Step 0: 加载数据...")
print("=" * 60)

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 深拷贝，避免修改原数据
data = copy.deepcopy(data)

# 初始统计
total_premises_initial = len(data)
total_clusters_initial = sum(d['n_clusters'] for d in data.values())
print(f"初始总 Premise 数: {total_premises_initial}")
print(f"初始总 Cluster 数: {total_clusters_initial}")
print(f"初始总位置数 (Premise + Cluster): {total_premises_initial + total_clusters_initial}")


# ==================== Step 1: 去除重复的 Premise ====================
print("\n" + "=" * 60)
print("Step 1: 去除重复的 Premise（相同 premise 文本只保留一个）")
print("=" * 60)

# 检查是否有重复的 premise 文本
# 数据结构：data[premise_text] = {...}，所以理论上不会有重复
# 但可能有 INDEX 不同但 premise 文本相同的情况
# 这种情况下需要合并或去重

# 统计每个 premise 文本对应的 INDEX
premise_to_indices = defaultdict(list)
for premise, info in data.items():
    premise_to_indices[premise].append(info['index'])

# 找出重复的 premise
duplicate_premises = {p: indices for p, indices in premise_to_indices.items() if len(indices) > 1}

if duplicate_premises:
    print(f"⚠️ 发现 {len(duplicate_premises)} 个重复的 Premise 文本")
    for premise, indices in list(duplicate_premises.items())[:5]:
        print(f"   - '{premise[:50]}...' 对应 INDEX: {indices}")
    # 保留第一个 INDEX 对应的记录
    # 由于 data 以 premise 为 key，实际上不会有重复，这里只是验证
else:
    print("✅ 没有发现重复的 Premise 文本")

print(f"当前 Premise 数: {len(data)}")

# ==================== Step 1.5: 限制每个 Premise 的 Cluster 数量 <= 7 ====================
print("\n" + "=" * 60)
print(f"Step 1.5: 限制每个 Premise 的 Cluster 数量 <= {MAX_CLUSTERS_PER_PREMISE}")
print("=" * 60)

# 找出 Cluster 数量大于 7 的 Premise
premises_over_limit = []
for premise, info in data.items():
    if info['n_clusters'] > MAX_CLUSTERS_PER_PREMISE:
        premises_over_limit.append({
            'premise': premise,
            'n_clusters': info['n_clusters'],
            'ask_for': info['ask_for']
        })

print(f"\nCluster 数量 > {MAX_CLUSTERS_PER_PREMISE} 的 Premise 数量: {len(premises_over_limit)}")

if premises_over_limit:
    # 按 Cluster 数量降序排列
    premises_over_limit.sort(key=lambda x: x['n_clusters'], reverse=True)
    
    print(f"\n前 10 个示例:")
    for i, p in enumerate(premises_over_limit[:10]):
        print(f"  {i+1}. [{p['ask_for']}] n_clusters={p['n_clusters']}: {p['premise'][:50]}...")
    
    # 去除多余的 Cluster（保留样本数最多的 7 个）
    removed_count_step1 = 0
    for p in premises_over_limit:
        premise = p['premise']
        clusters = data[premise]['clusters']
        
        # 按 count 降序排列
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # 保留前 MAX_CLUSTERS_PER_PREMISE 个
        clusters_to_keep = dict(sorted_clusters[:MAX_CLUSTERS_PER_PREMISE])
        clusters_to_remove = dict(sorted_clusters[MAX_CLUSTERS_PER_PREMISE:])
        
        removed_count_step1 += len(clusters_to_remove)
        
        # 更新数据
        data[premise]['clusters'] = clusters_to_keep
        data[premise]['n_clusters'] = len(clusters_to_keep)
    
    print(f"\nStep 1 完成: 移除了 {removed_count_step1} 个 Cluster")
else:
    print("无需处理")

# Step 1 后统计
total_clusters_after_step1 = sum(d['n_clusters'] for d in data.values())
print(f"Step 1 后总 Cluster 数: {total_clusters_after_step1}")


# ==================== Step 2: 限制 Premise 数 + Cluster 数 <= 4096 ====================
print("\n" + "=" * 60)
print(f"Step 2: 限制总位置数 (Premise + Cluster) <= {MAX_TOTAL_POSITIONS}")
print("=" * 60)

current_premises = len(data)
current_clusters = total_clusters_after_step1
current_total = current_premises + current_clusters

print(f"当前 Premise 数: {current_premises}")
print(f"当前 Cluster 数: {current_clusters}")
print(f"当前总位置数: {current_total}")

if current_total <= MAX_TOTAL_POSITIONS:
    print(f"当前总位置数已经 <= {MAX_TOTAL_POSITIONS}，无需压缩")
else:
    # 收集所有 Cluster 信息
    all_clusters = []
    for premise, info in data.items():
        for cluster_id, cluster_info in info['clusters'].items():
            all_clusters.append({
                'premise': premise,
                'cluster_id': cluster_id,
                'count': cluster_info['count'],
                'ask_for': info['ask_for']
            })
    
    # 按 count 升序排列（优先删除小样本）
    all_clusters.sort(key=lambda x: x['count'])
    
    # 统计各 count 值的 Cluster 数量
    count_distribution = defaultdict(list)
    for c in all_clusters:
        count_distribution[c['count']].append(c)
    
    print(f"\nCluster count 分布 (前 10 个):")
    for count in sorted(count_distribution.keys())[:10]:
        print(f"  count={count}: {len(count_distribution[count])} 个 Cluster")
    
    # 逐步去除小样本 Cluster
    to_remove = current_total - MAX_TOTAL_POSITIONS
    print(f"\n需要移除的 Cluster 数量: {to_remove}")
    
    removed_clusters = []
    current_count_threshold = 0
    
    for count in sorted(count_distribution.keys()):
        current_count_threshold = count
        clusters_at_this_count = count_distribution[count]
        
        if len(removed_clusters) + len(clusters_at_this_count) <= to_remove:
            # 全部移除
            removed_clusters.extend(clusters_at_this_count)
            print(f"  移除 count={count} 的全部 {len(clusters_at_this_count)} 个 Cluster (累计 {len(removed_clusters)})")
        else:
            # 部分移除
            need_more = to_remove - len(removed_clusters)
            # 随机选择（或按某种规则选择）
            removed_clusters.extend(clusters_at_this_count[:need_more])
            print(f"  移除 count={count} 的部分 {need_more} 个 Cluster (累计 {len(removed_clusters)})")
            break
    
    print(f"\n总共标记移除 {len(removed_clusters)} 个 Cluster")
    print(f"移除的 Cluster 的 count 范围: {min(c['count'] for c in removed_clusters)} - {max(c['count'] for c in removed_clusters)}")
    
    # 执行移除
    for c in removed_clusters:
        premise = c['premise']
        cluster_id = c['cluster_id']
        if cluster_id in data[premise]['clusters']:
            del data[premise]['clusters'][cluster_id]
            data[premise]['n_clusters'] -= 1
    
    # 检查是否有 Premise 的所有 Cluster 都被移除了
    empty_premises = [p for p, info in data.items() if info['n_clusters'] == 0]
    if empty_premises:
        print(f"\n⚠️ 警告: {len(empty_premises)} 个 Premise 的所有 Cluster 都被移除了")
        for p in empty_premises:
            del data[p]

# Step 2 后统计
total_clusters_after_step2 = sum(d['n_clusters'] for d in data.values())
print(f"\nStep 2 后总 Cluster 数: {total_clusters_after_step2}")


# ==================== Step 3: 重新统计 Cause/Effect 类的 Cluster 数量 ====================
print("\n" + "=" * 60)
print("Step 3: 最终统计")
print("=" * 60)

final_total_premises = len(data)
final_total_clusters = sum(d['n_clusters'] for d in data.values())
final_cause_premises = sum(1 for d in data.values() if d['ask_for'] == 'cause')
final_effect_premises = sum(1 for d in data.values() if d['ask_for'] == 'effect')
final_cause_clusters = sum(d['n_clusters'] for d in data.values() if d['ask_for'] == 'cause')
final_effect_clusters = sum(d['n_clusters'] for d in data.values() if d['ask_for'] == 'effect')

print(f"\n最终统计:")
print(f"  总 Premise 数: {final_total_premises}")
print(f"  总 Cluster 数: {final_total_clusters}")
print(f"  - Cause 类 Premise: {final_cause_premises}")
print(f"  - Cause 类 Cluster: {final_cause_clusters}")
print(f"  - Effect 类 Premise: {final_effect_premises}")
print(f"  - Effect 类 Cluster: {final_effect_clusters}")

# 验证
final_total_positions = final_total_premises + final_total_clusters
print(f"\n验证:")
print(f"  总位置数 (Premise + Cluster) = {final_total_positions} {'<=' if final_total_positions <= MAX_TOTAL_POSITIONS else '>'} {MAX_TOTAL_POSITIONS}: {'✓ 通过' if final_total_positions <= MAX_TOTAL_POSITIONS else '✗ 失败'}")

# 检查每个 Premise 的 Cluster 数量
max_clusters_per_premise = max(d['n_clusters'] for d in data.values())
print(f"  最大 Cluster/Premise = {max_clusters_per_premise} {'<=' if max_clusters_per_premise <= MAX_CLUSTERS_PER_PREMISE else '>'} {MAX_CLUSTERS_PER_PREMISE}: {'✓ 通过' if max_clusters_per_premise <= MAX_CLUSTERS_PER_PREMISE else '✗ 失败'}")


# ==================== 保存结果 ====================
print("\n" + "=" * 60)
print("保存结果...")
print("=" * 60)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 压缩后的数据已保存到: {OUTPUT_FILE}")

# 对比摘要
print("\n" + "=" * 60)
print("压缩前后对比")
print("=" * 60)
print(f"{'指标':<30} {'压缩前':>12} {'压缩后':>12} {'变化':>12}")
print("-" * 66)
print(f"{'总 Premise 数':<30} {total_premises_initial:>12} {final_total_premises:>12} {final_total_premises - total_premises_initial:>+12}")
print(f"{'总 Cluster 数':<30} {total_clusters_initial:>12} {final_total_clusters:>12} {final_total_clusters - total_clusters_initial:>+12}")
print(f"{'总位置数 (Premise + Cluster)':<30} {total_premises_initial + total_clusters_initial:>12} {final_total_positions:>12} {final_total_positions - total_premises_initial - total_clusters_initial:>+12}")
print(f"{'Cause 类 Cluster':<30} {'-':>12} {final_cause_clusters:>12}")
print(f"{'Effect 类 Cluster':<30} {'-':>12} {final_effect_clusters:>12}")

print("\n✅ 完成!")
