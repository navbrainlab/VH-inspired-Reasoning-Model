# ===== 因果数据处理与编码学习 =====
# 功能：
# 1. 加载Gemma回答数据并使用NV-Embed向量化
# 2. 使用HDBSCAN聚类
# 3. 使用centroid方法找到类代表向量
# 4. 构造训练数据集 {A, B, ASKFOR, INDEX}
# 5. 模仿Causal_V2进行因果事件编码学习

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ================== 配置参数 ==================
# Linux 路径
BASE_DIR = "/home/amax/Zixing_Jia/Vector-HASH-tinkering-main/Gemma_anwer_causal"
DATA_PATH = os.path.join(BASE_DIR, "T_1.25_Gemma_Answer/gemma_causal_answers_100.jsonl")
MODEL_PATH = "/home/amax/Gemma_and_NVembed/NV/snapshots/main"
OUTPUT_DIR = os.path.join(BASE_DIR, "Causal_Data")

# HDBSCAN参数
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3

# ================== 创建输出目录 ==================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== 加载NV-Embed模型 ==================
print("=" * 60)
print("Step 1: 加载NV-Embed-v2模型")
print("=" * 60)

from transformers import AutoModel, AutoTokenizer

print(f"模型路径: {MODEL_PATH}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

print("⏳ 正在加载模型...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"\n模型加载成功！运行设备: {device}")

# ================== 定义编码函数 ==================
def encode_texts(texts, max_length=4096, batch_size=32):
    """
    将文本编码为向量
    
    Args:
        texts: 单个字符串或字符串列表
        max_length: 最大序列长度
        batch_size: 批处理大小
    
    Returns:
        numpy array of shape (n_texts, 4096)
    """
    if isinstance(texts, str):
        texts = [texts]
    
    eos_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
    texts_with_eos = [text + eos_token for text in texts]
    
    all_embeddings = []
    
    for i in range(0, len(texts_with_eos), batch_size):
        batch_texts = texts_with_eos[i:i+batch_size]
        
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        attention_mask = batch_dict["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1)
        pool_mask = torch.zeros_like(attention_mask)
        for j, length in enumerate(seq_lengths):
            pool_mask[j, length - 1] = 1
        
        with torch.no_grad():
            outputs = model(
                input_ids=batch_dict["input_ids"],
                attention_mask=attention_mask,
                pool_mask=pool_mask
            )
            
            if isinstance(outputs, dict):
                if "sentence_embeddings" in outputs:
                    embeddings = outputs["sentence_embeddings"]
                elif "last_hidden_state" in outputs:
                    embeddings = outputs["last_hidden_state"][:, -1, :]
                else:
                    raise ValueError(f"无法从输出中提取嵌入向量: {outputs.keys()}")
            elif hasattr(outputs, "sentence_embeddings"):
                embeddings = outputs.sentence_embeddings
            elif hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state[:, -1, :]
            else:
                embeddings = outputs
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_cpu = embeddings.float().cpu()
        all_embeddings.append(np.array(embeddings_cpu.tolist()))
    
    return np.vstack(all_embeddings)

# ================== 加载数据 ==================
print("\n" + "=" * 60)
print("Step 2: 加载Gemma回答数据")
print("=" * 60)

data_by_premise = defaultdict(list)  # premise -> list of records
all_records = []

print(f"📁 数据文件: {DATA_PATH}")

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line.strip())
        all_records.append(record)
        premise = record['premise']
        data_by_premise[premise].append(record)

print(f"✅ 加载了 {len(all_records)} 条记录")
print(f"✅ 共有 {len(data_by_premise)} 个不同的 Premise")

# 统计ask-for分布
ask_for_counts = defaultdict(int)
for r in all_records:
    ask_for_counts[r['ask-for']] += 1
print(f"📊 Ask-for 分布: {dict(ask_for_counts)}")

# ================== 向量化 Premise ==================
print("\n" + "=" * 60)
print("Step 3: 向量化所有唯一的 Premise")
print("=" * 60)

unique_premises = list(data_by_premise.keys())
print(f"🔄 正在向量化 {len(unique_premises)} 个唯一 Premise...")

premise_embeddings = {}
batch_size = 32
for i in tqdm(range(0, len(unique_premises), batch_size)):
    batch = unique_premises[i:i+batch_size]
    batch_emb = encode_texts(batch)
    for j, premise in enumerate(batch):
        premise_embeddings[premise] = batch_emb[j]

print(f"✅ Premise 向量化完成！维度: {next(iter(premise_embeddings.values())).shape}")

# ================== 向量化 Gemma_answer 并聚类 ==================
print("\n" + "=" * 60)
print("Step 4: 对每个 Premise 的 Gemma_answer 进行向量化和聚类")
print("=" * 60)

# 存储聚类结果
# Structure: {premise: {
#   'premise_vec': np.array,
#   'ask_for': str,
#   'index': str,
#   'clusters': {cluster_id: {'centroid': np.array, 'count': int, 'answers': list}}
# }}
clustering_results = {}

for premise, records in tqdm(data_by_premise.items(), desc="处理 Premise"):
    # 获取该premise的所有回答
    answers = [r['Gemma_answer'] for r in records]
    ask_for = records[0]['ask-for']
    index = records[0]['index']
    
    # 向量化所有回答
    answer_embeddings = encode_texts(answers)
    answer_embeddings_norm = normalize(answer_embeddings, norm='l2')
    
    # HDBSCAN聚类
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(answer_embeddings_norm)
    
    # 计算每个聚类的centroid和样本数
    unique_clusters = set(cluster_labels)
    clusters_info = {}
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # 跳过噪声点
            continue
        
        # 获取该聚类的所有向量
        mask = cluster_labels == cluster_id
        cluster_vectors = answer_embeddings_norm[mask]
        cluster_answers = [answers[i] for i, m in enumerate(mask) if m]
        
        # 计算centroid
        centroid = np.mean(cluster_vectors, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)  # 归一化
        
        clusters_info[cluster_id] = {
            'centroid': centroid,
            'count': int(np.sum(mask)),
            'answers': cluster_answers[:5]  # 只保存前5个作为示例
        }
    
    # 如果没有有效聚类，将所有点作为一个聚类
    if len(clusters_info) == 0:
        centroid = np.mean(answer_embeddings_norm, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        clusters_info[0] = {
            'centroid': centroid,
            'count': len(answers),
            'answers': answers[:5]
        }
    
    clustering_results[premise] = {
        'premise_vec': premise_embeddings[premise],
        'ask_for': ask_for,
        'index': index,
        'clusters': clusters_info,
        'total_answers': len(answers),
        'n_clusters': len(clusters_info)
    }

print(f"\n✅ 聚类完成！")
print(f"📊 聚类统计:")
total_clusters = sum(r['n_clusters'] for r in clustering_results.values())
print(f"   - 总聚类数: {total_clusters}")
print(f"   - 平均每个Premise的聚类数: {total_clusters / len(clustering_results):.2f}")

# ================== 构造训练数据集 ==================
print("\n" + "=" * 60)
print("Step 5: 构造训练数据集 {A, B, ASKFOR, INDEX}")
print("=" * 60)

# 数据集结构：
# - A: Premise向量 (或 Gemma_answer向量，取决于ASKFOR)
# - B: Gemma_answer向量的centroid (或 Premise向量)
# - ASKFOR: 'cause' 或 'effect'
# - INDEX: 原始数据集中的索引
# 
# 如果 ASKFOR='cause': Premise是果，Gemma_answer是因
# 如果 ASKFOR='effect': Premise是因，Gemma_answer是果

causal_data = []

for premise, result in clustering_results.items():
    ask_for = result['ask_for']
    index = result['index']
    premise_vec = result['premise_vec']
    
    for cluster_id, cluster_info in result['clusters'].items():
        centroid = cluster_info['centroid']
        count = cluster_info['count']
        
        # 根据ask_for确定A和B
        # ask_for='cause' 表示问的是原因，所以Premise是结果(Effect)，Gemma_answer是原因(Cause)
        # ask_for='effect' 表示问的是结果，所以Premise是原因(Cause)，Gemma_answer是结果(Effect)
        
        if ask_for == 'cause':
            # Premise是Effect，Gemma_answer(centroid)是Cause
            A = centroid  # Cause
            B = premise_vec  # Effect
            direction = 'cause'  # 表示 A->B 是 因->果
        else:  # effect
            # Premise是Cause，Gemma_answer(centroid)是Effect
            A = premise_vec  # Cause
            B = centroid  # Effect
            direction = 'effect'  # 表示 A->B 是 因->果
        
        # 根据count复制样本
        for _ in range(count):
            causal_data.append({
                'A': A,
                'B': B,
                'ASKFOR': ask_for,
                'INDEX': index,
                'cluster_id': cluster_id,
                'premise': premise
            })

print(f"✅ 数据集构造完成！")
print(f"📊 数据集统计:")
print(f"   - 总样本数: {len(causal_data)}")
print(f"   - ask-for='cause' 样本数: {sum(1 for d in causal_data if d['ASKFOR'] == 'cause')}")
print(f"   - ask-for='effect' 样本数: {sum(1 for d in causal_data if d['ASKFOR'] == 'effect')}")

# ================== 保存数据 ==================
print("\n" + "=" * 60)
print("Step 6: 保存数据")
print("=" * 60)

# 保存向量化数据（numpy格式）
np.save(os.path.join(OUTPUT_DIR, 'causal_A_vectors.npy'), 
        np.array([d['A'] for d in causal_data]))
np.save(os.path.join(OUTPUT_DIR, 'causal_B_vectors.npy'), 
        np.array([d['B'] for d in causal_data]))

# 保存元数据（JSON格式）
metadata = [{
    'ASKFOR': d['ASKFOR'],
    'INDEX': d['INDEX'],
    'cluster_id': int(d['cluster_id']),
    'premise': d['premise']
} for d in causal_data]

with open(os.path.join(OUTPUT_DIR, 'causal_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# 保存聚类结果摘要
clustering_summary = {}
for premise, result in clustering_results.items():
    clustering_summary[premise] = {
        'ask_for': result['ask_for'],
        'index': result['index'],
        'total_answers': result['total_answers'],
        'n_clusters': result['n_clusters'],
        'clusters': {
            str(k): {
                'count': v['count'],
                'sample_answers': v['answers']
            } for k, v in result['clusters'].items()
        }
    }

with open(os.path.join(OUTPUT_DIR, 'clustering_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(clustering_summary, f, ensure_ascii=False, indent=2)

# 保存Premise向量
premise_vecs = {premise: result['premise_vec'].tolist() 
                for premise, result in clustering_results.items()}
with open(os.path.join(OUTPUT_DIR, 'premise_vectors.json'), 'w', encoding='utf-8') as f:
    json.dump(premise_vecs, f, ensure_ascii=False)

print(f"✅ 数据已保存到: {OUTPUT_DIR}")
print(f"   - causal_A_vectors.npy: A向量（因或果）")
print(f"   - causal_B_vectors.npy: B向量（果或因）")
print(f"   - causal_metadata.json: 元数据")
print(f"   - clustering_summary.json: 聚类摘要")
print(f"   - premise_vectors.json: Premise向量")

print("\n" + "=" * 60)
print("数据处理完成！")
print("=" * 60)
