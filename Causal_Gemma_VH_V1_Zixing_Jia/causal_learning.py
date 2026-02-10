# ===== 因果编码学习（基于NV-Embed向量化数据） =====
# 模仿 Causal_V2.ipynb 的流程进行因果事件编码学习
# 
# 输入：causal_data_processing.py 生成的数据
# - causal_A_vectors.npy: A向量
# - causal_B_vectors.npy: B向量
# - causal_metadata.json: 元数据 (ASKFOR, INDEX)
# nohup python /home/amax/Zixing_Jia/Vector-HASH-tinkering-main/Gemma_anwer_causal/causal_learning.py > training.log 2>&1 &
import numpy as np
from numpy.random import randn, randint
from tqdm import tqdm
import json
import os

# matplotlib 延迟导入（避免 NumPy 版本冲突）
ENABLE_VISUALIZATION = False  # 设为 True 启用可视化（需要兼容的 matplotlib）

# ================== 配置参数 ==================
DATA_DIR = r"/home/amax/Zixing_Jia/Vector-HASH-tinkering-main/Gemma_anwer_causal/Causal_Data"
RESULTS_DIR = r"/home/amax/Zixing_Jia/Vector-HASH-tinkering-main/Gemma_anwer_causal/Causal_Results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ================== 加载自定义模块 ==================
import sys
sys.path.append(r"/home/amax/Zixing_Jia/Vector-HASH-tinkering-main")
from src.assoc_utils_np_2D import gen_gbook_2d, path_integration_Wgg_2d, module_wise_NN_2d
from src.seq_utils import nonlin, sensorymap

# ================== 加载数据 ==================
print("=" * 60)
print("Step 1: 加载因果数据")
print("=" * 60)

A_vectors = np.load(os.path.join(DATA_DIR, 'causal_A_vectors_compressed.npy'))
B_vectors = np.load(os.path.join(DATA_DIR, 'causal_B_vectors_compressed.npy'))

with open(os.path.join(DATA_DIR, 'causal_metadata_compressed.json'), 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print(f"[OK] 加载数据:")
print(f"   - A向量: shape={A_vectors.shape}")
print(f"   - B向量: shape={B_vectors.shape}")
print(f"   - 元数据: {len(metadata)} 条")

# 数据维度
Ns_embed = A_vectors.shape[1]  # 嵌入维度 (4096)
n_total_samples = len(metadata)

print(f"   - 嵌入维度: {Ns_embed}")
print(f"   - 样本总数: {n_total_samples}")

# 统计ask-for分布
askfor_counts = {'cause': 0, 'effect': 0}
for m in metadata:
    askfor_counts[m['ASKFOR']] += 1
print(f"   - ASKFOR分布: {askfor_counts}")

# ================== 基础结构超参数 ==================
print("\n" + "=" * 60)
print("Step 2: 初始化Grid-Place Cell结构")
print("=" * 60)

nruns = 1
Np = 4096  # Place Cell数量
lambdas = [3, 4, 5, 7]  # Grid Cell模块的周期
Ng = np.sum(np.square(lambdas))
Npos = np.prod(lambdas)  # 420

print(f"Grid Size (Npos): {Npos} x {Npos} = {Npos*Npos}")
print(f"Number of Grid Cells (Ng): {Ng}")
print(f"Number of Place Cells (Np): {Np}")

# 生成 gbook (Grid Cell激活模式)
gbook = gen_gbook_2d(lambdas, Ng, Npos)
gbook_flattened = gbook.reshape(Ng, Npos*Npos)

# 生成 pbook (Place Cell激活模式)
rng = np.random.default_rng(42)
Wpg = randn(nruns, Np, Ng)
c = 0.10  # 连接概率
prune = int((1-c)*Np*Ng)
mask = np.ones((Np, Ng))
mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
Wpg = np.multiply(mask, Wpg)
thresh = 2.0
pbook = nonlin(np.einsum('ijk,klm->ijlm', Wpg, gbook), thresh=thresh)
pbook_flattened = pbook.reshape(nruns, Np, Npos*Npos)

print("[OK] Grid-Place Cell结构初始化完成")

# ================== 因果模型超参数 ==================
print("\n" + "=" * 60)
print("Step 3: 初始化因果模型")
print("=" * 60)

# 根据数据特点设置参数
# 获取唯一的INDEX数量作为事件数
unique_indices = list(set(m['INDEX'] for m in metadata))
C = len(unique_indices)  # 因事件数量（唯一Premise数）

k = 3  # Effect capacity parameter
max_effects = 2 * k + 1  # 每个因最多有7个果

# 块布局参数
block_height = 6
block_width = 2 * k + 1

print(f"因事件数量 (C): {C}")
print(f"最大效果数: {max_effects}")

# ================== 辅助函数 ==================

def flat_idx(x, y, Npos=Npos):
    """Convert 2D grid coordinates to 1D flattened index."""
    return int(x * Npos + y)

def get_pvec(x, y, pbook_flattened=pbook_flattened):
    """Retrieve the Place Cell vector for a given coordinate."""
    return pbook_flattened[0, :, flat_idx(x, y)]

def rls_step(W, theta, a, y):
    """
    Recursive Least Squares (RLS) update step.
    用于学习 sensory -> place cell 和 place cell -> sensory 的映射
    """
    a = np.asarray(a).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    
    pred_before = W @ a
    err_before = y - pred_before
    
    denom = 1.0 + (a.T @ theta @ a).item()
    bk = (theta @ a) / denom
    
    theta_new = theta - (theta @ a @ bk.T)
    W_new = W + (err_before @ bk.T)
    
    pred_after = W_new @ a
    err_after = y - pred_after
    
    return W_new, theta_new, float(np.linalg.norm(bk)), float(np.linalg.norm(err_before)), float(np.linalg.norm(err_after))

# ================== RRM 模型 ==================
class RRM_Model:
    """
    Relevance-Rate Matrix Model
    用于学习因果关系中的概率分布
    """
    def __init__(self, max_effects, learning_rate=0.1, decay=0.01, T=0.5):
        self.max_effects = max_effects
        self.eta = learning_rate
        self.lam = decay
        self.T = T
        self.weights = {}
        self.counts = {}
    
    def get_weights(self, cause_id):
        if cause_id not in self.weights:
            self.weights[cause_id] = np.zeros(self.max_effects)
            self.counts[cause_id] = np.zeros(self.max_effects)
        return self.weights[cause_id]
    
    def update(self, cause_id, effect_slot_idx, x_c=1.0, y_o=1.0):
        """Hebbian update"""
        w = self.get_weights(cause_id)
        y_vec = np.zeros(self.max_effects)
        y_vec[effect_slot_idx] = y_o
        
        delta_w = self.eta * x_c * y_vec - self.lam * w
        self.weights[cause_id] += delta_w
        self.weights[cause_id] = np.maximum(self.weights[cause_id], 0)
        self.counts[cause_id][effect_slot_idx] += 1
    
    def sample(self, cause_id, mode="max_prob"):
        """
        Sample an effect slot based on weights.
        
        Args:
            cause_id: 源事件的ID
            mode: 采样模式
                - "direct": 按权重归一化后的概率采样
                - "Boltzmann": 使用Boltzmann分布（softmax）采样
                - "max_prob": 直接选择概率最大的slot（确定性选择）
        
        Returns:
            sampled_slot: 采样到的slot索引
            probs: 概率分布
        """
        w = self.get_weights(cause_id)
        
        if mode == "direct":
            total_w = np.sum(w)
            if total_w == 0:
                probs = np.ones(self.max_effects) / self.max_effects
            else:
                probs = w / total_w
            return np.random.choice(self.max_effects, p=probs), probs
        elif mode == "Boltzmann":
            logits = w / self.T
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            return np.random.choice(self.max_effects, p=probs), probs
        elif mode == "max_prob":
            # 计算概率分布（用于返回）
            total_w = np.sum(w)
            if total_w == 0:
                probs = np.ones(self.max_effects) / self.max_effects
            else:
                probs = w / total_w
            # 直接选择概率最大的slot（确定性）
            max_slot = np.argmax(probs)
            return max_slot, probs
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")

# ================== 数据预处理 ==================
print("\n" + "=" * 60)
print("Step 4: 数据预处理 - 构建事件映射")
print("=" * 60)

# 因果方向定义
CAUSE_TO_EFFECT = 1   # 因到果
EFFECT_TO_CAUSE = -1  # 果到因

# 为每个唯一的(INDEX, ASKFOR, cluster_id)组合创建事件
# 组织数据结构
# index_to_data: {INDEX: {'ask_for': str, 'premise_vec': vec, 'clusters': {cluster_id: {'vec': centroid_vec, 'count': count}}}}
index_to_data = {}

for i, m in enumerate(metadata):
    idx = m['INDEX']
    askfor = m['ASKFOR']
    cluster_id = m['cluster_id']
    
    if idx not in index_to_data:
        index_to_data[idx] = {
            'ask_for': askfor,
            'A_vecs': {},  # cluster_id -> A向量
            'B_vecs': {},  # cluster_id -> B向量
            'counts': {}   # cluster_id -> 出现次数
        }
    
    if cluster_id not in index_to_data[idx]['counts']:
        index_to_data[idx]['A_vecs'][cluster_id] = A_vectors[i]
        index_to_data[idx]['B_vecs'][cluster_id] = B_vectors[i]
        index_to_data[idx]['counts'][cluster_id] = 0
    
    index_to_data[idx]['counts'][cluster_id] += 1

print(f"[OK] 构建了 {len(index_to_data)} 个唯一INDEX的数据映射")

# 分离cause和effect类型的数据
cause_indices = [idx for idx, data in index_to_data.items() if data['ask_for'] == 'cause']
effect_indices = [idx for idx, data in index_to_data.items() if data['ask_for'] == 'effect']

print(f"   - ask_for='cause' 的INDEX数量: {len(cause_indices)}")
print(f"   - ask_for='effect' 的INDEX数量: {len(effect_indices)}")

# ================== Sensory 维度设置 ==================
print("\n" + "=" * 60)
print("Step 5: Sensory 维度设置")
print("=" * 60)

# 直接使用 NV-Embed 的 4096 维输出
Ns = Ns_embed  # 4096 维

# ================== 生成训练流 ==================
print("\n" + "=" * 60)
print("Step 6: 生成训练流")
print("=" * 60)

# 构建训练样本
stream = []

# 为每个INDEX构建训练样本
for idx in index_to_data:
    data = index_to_data[idx]
    ask_for = data['ask_for']
    
    # 根据每个cluster的count复制样本
    for cluster_id, count in data['counts'].items():
        A_vec = data['A_vecs'][cluster_id]
        B_vec = data['B_vecs'][cluster_id]
        
        # causal_data_processing 中的数据结构：
        # - A 总是 Cause，B 总是 Effect
        # 
        # 但位置分配需要根据 ask_for 区分：
        # - ask_for='cause'（一果多因）: 
        #     * Premise 是 Effect (B)，放在 (N, N)
        #     * Gemma_answer 是 Cause (A)，放在 (N-1, N±offset)
        #     * 推理方向: 因→果，从 x-1 到 x
        # - ask_for='effect'（一因多果）: 
        #     * Premise 是 Cause (A)，放在 (N, N)
        #     * Gemma_answer 是 Effect (B)，放在 (N+1, N±offset)
        #     * 推理方向: 因→果，从 x 到 x+1
        
        # 统一使用 CAUSE_TO_EFFECT 方向，但位置分配不同
        direction = CAUSE_TO_EFFECT
        
        for _ in range(count):
            stream.append({
                'direction': direction,
                'A_vec': A_vec,  # Cause向量
                'B_vec': B_vec,  # Effect向量
                'source_idx': idx,
                'cluster_id': cluster_id,
                'ask_for': ask_for
            })

# 打乱样本
rng.shuffle(stream)

print(f"[OK] 生成训练流: {len(stream)} 样本")

# ================== 在线学习循环 ==================
print("\n" + "=" * 60)
print("Step 7: 在线学习")
print("=" * 60)

# 初始化RLS矩阵
# Wps: Sensory -> Place (Np x Ns)
# Wsp: Place -> Sensory (Ns x Np)
Wps = np.zeros((Np, Ns), dtype=float)
Wsp = np.zeros((Ns, Np), dtype=float)
epsilon = 0.05
theta_ps = (1.0 / (epsilon**2)) * np.eye(Ns)
theta_sp = (1.0 / (epsilon**2)) * np.eye(Np)

# 初始化RRM模型
rrm = RRM_Model(max_effects=max_effects, learning_rate=0.05, decay=0.001, T=1.0)

# 位置追踪器 - 使用通用名称
current_x = 0
current_y = 0
premise_locs = {}     # (source_idx, ask_for) -> (x, y)  - Premise 的位置
answer_locs = {}      # (source_idx, ask_for, cluster_id) -> (x, y)  - Answer 的位置
premise_next_slot = {}    # (source_idx, ask_for) -> next_slot
premise_block_origins = {}  # (source_idx, ask_for) -> (block_x, block_y)
premise_vectors = {}  # (source_idx, ask_for) -> vec
answer_vectors = {}   # (source_idx, ask_for, cluster_id) -> vec

# 同时记录因果位置（用于保存和可视化）
cause_locs = {}   # (source_idx, ask_for) or (source_idx, ask_for, cluster_id) -> (x, y)
effect_locs = {}  # (source_idx, ask_for) or (source_idx, ask_for, cluster_id) -> (x, y)
cause_vectors = {}
effect_vectors = {}

def get_premise_key(sample):
    return (sample['source_idx'], sample['ask_for'])

def get_answer_key(sample):
    return (sample['source_idx'], sample['ask_for'], sample['cluster_id'])

# 训练历史
history_err = []

for t, sample in enumerate(tqdm(stream, desc="训练")):
    direction = sample['direction']
    source_idx = sample['source_idx']
    cluster_id = sample['cluster_id']
    A_vec = sample['A_vec']  # Cause 向量
    B_vec = sample['B_vec']  # Effect 向量
    ask_for = sample['ask_for']
    
    premise_key = get_premise_key(sample)
    answer_key = get_answer_key(sample)
    
    # 根据 ask_for 确定 premise 和 answer 的实际向量
    # ask_for='cause'（一果多因）: Premise=Effect(B), Answer=Cause(A)
    # ask_for='effect'（一因多果）: Premise=Cause(A), Answer=Effect(B)
    if ask_for == 'cause':
        # 一果多因: Premise 是 Effect，Answer 是 Cause
        premise_vec = B_vec
        answer_vec = A_vec
        x_offset = -1  # Answer (Cause) 在 Premise (Effect) 的 x-1 方向
    else:  # ask_for == 'effect'
        # 一因多果: Premise 是 Cause，Answer 是 Effect
        premise_vec = A_vec
        answer_vec = B_vec
        x_offset = +1  # Answer (Effect) 在 Premise (Cause) 的 x+1 方向
    
    # 1. 动态分配 Premise 的位置
    if premise_key not in premise_locs:
        if current_x + block_height > Npos:
            current_x = 0
            current_y += block_width
        
        if current_y + block_width > Npos:
            continue
        
        # Premise 的位置（根据 ask_for 调整）
        if ask_for == 'cause':
            # 一果多因: Premise (Effect) 放在 (N+1, N)，预留 x-1 给 Cause
            p_x = current_x + 3  # 空出 x=2 给 Answer (Cause)
        else:
            # 一因多果: Premise (Cause) 放在 (N, N)
            p_x = current_x + 2
        p_y = current_y + k
        
        premise_locs[premise_key] = (p_x, p_y)
        premise_block_origins[premise_key] = (current_x, current_y)
        premise_vectors[premise_key] = premise_vec
        
        # 同时记录因果位置
        if ask_for == 'cause':
            # Premise 是 Effect
            effect_locs[premise_key] = (p_x, p_y)
            effect_vectors[premise_key] = premise_vec
        else:
            # Premise 是 Cause
            cause_locs[premise_key] = (p_x, p_y)
            cause_vectors[premise_key] = premise_vec
        
        current_x += block_height
        premise_next_slot[premise_key] = 0
    
    # 2. 动态分配 Answer 的位置
    if answer_key not in answer_locs:
        slot = premise_next_slot[premise_key]
        if slot < max_effects:
            premise_next_slot[premise_key] += 1
            
            # Answer 的位置（在 Premise 的 x±1 方向）
            p_x, p_y = premise_locs[premise_key]
            
            # 计算 y 方向的 slot 偏移: 0, -1, +1, -2, +2, ...
            if slot == 0:
                offset = 0
            elif slot % 2 != 0:
                offset = -((slot + 1) // 2)
            else:
                offset = slot // 2
            
            a_x = p_x + x_offset  # 根据 ask_for 确定 x 方向
            a_y = p_y + offset
            answer_locs[answer_key] = (a_x, a_y)
            answer_vectors[answer_key] = answer_vec
            
            # 同时记录因果位置
            if ask_for == 'cause':
                # Answer 是 Cause
                cause_locs[answer_key] = (a_x, a_y)
                cause_vectors[answer_key] = answer_vec
            else:
                # Answer 是 Effect
                effect_locs[answer_key] = (a_x, a_y)
                effect_vectors[answer_key] = answer_vec
        else:
            continue
    
    if premise_key not in premise_locs or answer_key not in answer_locs:
        continue
    
    # 3. 获取位置向量
    loc_premise = premise_locs[premise_key]
    loc_answer = answer_locs[answer_key]
    p_premise = get_pvec(*loc_premise)
    p_answer = get_pvec(*loc_answer)
    
    s_premise = premise_vectors[premise_key]
    s_answer = answer_vectors[answer_key]
    
    # 4. RLS更新 - 绑定sensory向量到place cell
    Wps, theta_ps, _, err_before_premise, _ = rls_step(Wps, theta_ps, s_premise, p_premise)
    Wsp, theta_sp, _, _, _ = rls_step(Wsp, theta_sp, p_premise, s_premise)
    Wps, theta_ps, _, err_before_answer, _ = rls_step(Wps, theta_ps, s_answer, p_answer)
    Wsp, theta_sp, _, _, _ = rls_step(Wsp, theta_sp, p_answer, s_answer)
    
    # 5. RRM更新 - 学习因果转移概率
    # 使用 INDEX_ASKFOR 格式作为RRM的key
    rrm_key = f"{source_idx}_{ask_for}"
    
    # 计算slot_idx: answer 在当前 premise 下的 slot 位置
    slot_idx = premise_next_slot[premise_key] - 1
    slot_idx = max(0, min(slot_idx, max_effects - 1))
    
    rrm.update(rrm_key, slot_idx, x_c=1.0, y_o=1.0)
    
    history_err.append((err_before_premise + err_before_answer) / 2)

print(f"\n[OK] 训练完成！")
print(f"   - 学习的 Premise 数量: {len(premise_locs)}")
print(f"   - 学习的 Answer 数量: {len(answer_locs)}")
print(f"   - 学习的 Cause 位置数量: {len(cause_locs)}")
print(f"   - 学习的 Effect 位置数量: {len(effect_locs)}")
print(f"   - RRM条目数: {len(rrm.weights)}")

# ================== 保存训练结果 ==================
print("\n" + "=" * 60)
print("Step 8: 保存训练结果")
print("=" * 60)

# 保存权重矩阵
np.save(os.path.join(RESULTS_DIR, 'Wps.npy'), Wps)
np.save(os.path.join(RESULTS_DIR, 'Wsp.npy'), Wsp)

# 保存 Grid-Place Cell 矩阵 (关键！测试时需要用相同的 Wpg 和 pbook)
np.save(os.path.join(RESULTS_DIR, 'Wpg.npy'), Wpg)  # (nruns, Np, Ng)
np.save(os.path.join(RESULTS_DIR, 'pbook.npy'), pbook_flattened)  # (nruns, Np, Npos^2)

# 计算并保存 Wgp (Grid -> Place 逆映射)
Wgp = gbook_flattened @ np.linalg.pinv(pbook_flattened[0])  # (Ng, Np)
np.save(os.path.join(RESULTS_DIR, 'Wgp.npy'), Wgp)

print(f"[OK] 保存 Wpg: {Wpg.shape}")
print(f"[OK] 保存 pbook: {pbook_flattened.shape}")
print(f"[OK] 保存 Wgp: {Wgp.shape}")

# 保存位置映射 - 同时保存 premise/answer 和 cause/effect 格式
# cause_locs 和 effect_locs 的 key 可能是 tuple 或 tuple with 3 elements
def format_loc_key(k):
    if len(k) == 2:
        return f"{k[0]}_{k[1]}"
    else:
        return f"{k[0]}_{k[1]}_{k[2]}"

positions = {
    # 新格式: premise/answer (用于推理)
    'premise_locs': {format_loc_key(k): list(v) for k, v in premise_locs.items()},
    'answer_locs': {format_loc_key(k): list(v) for k, v in answer_locs.items()},
    # 兼容格式: cause/effect (用于语义理解)
    'cause_locs': {format_loc_key(k): list(v) for k, v in cause_locs.items()},
    'effect_locs': {format_loc_key(k): list(v) for k, v in effect_locs.items()},
}
with open(os.path.join(RESULTS_DIR, 'positions.json'), 'w', encoding='utf-8') as f:
    json.dump(positions, f, ensure_ascii=False, indent=2)

print(f"[OK] 保存位置映射: {len(premise_locs)} premises, {len(answer_locs)} answers")

# 保存RRM权重
# RRM key格式: "premise_type_source_idx", e.g., "Cause_0", "Effect_1"
rrm_data = {
    'weights': {k: v.tolist() for k, v in rrm.weights.items()},
    'counts': {k: v.tolist() for k, v in rrm.counts.items()},
    'config': {
        'max_effects': max_effects,
        'learning_rate': 0.05,
        'decay': 0.001,
        'T': 1.0
    }
}
with open(os.path.join(RESULTS_DIR, 'rrm_model.json'), 'w', encoding='utf-8') as f:
    json.dump(rrm_data, f, ensure_ascii=False, indent=2)

print(f"[OK] 结果已保存到: {RESULTS_DIR}")

# ================== 可视化 ==================
print("\n" + "=" * 60)
print("Step 9: 可视化")
print("=" * 60)

if ENABLE_VISUALIZATION:
    try:
        import matplotlib.pyplot as plt
        
        # 绘制训练误差曲线
        plt.figure(figsize=(10, 4))
        plt.plot(history_err)
        plt.xlabel('Training Step')
        plt.ylabel('RLS Error')
        plt.title('Training Error over Time')
        plt.savefig(os.path.join(RESULTS_DIR, 'training_error.png'), dpi=150)
        plt.close()

        # 绘制空间分布
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制Cause位置
        cause_x = [loc[0] for loc in cause_locs.values()]
        cause_y = [loc[1] for loc in cause_locs.values()]
        ax.scatter(cause_x, cause_y, c='red', s=80, marker='s', label='Cause', alpha=0.7)

        # 绘制Effect位置
        effect_x = [loc[0] for loc in effect_locs.values()]
        effect_y = [loc[1] for loc in effect_locs.values()]
        ax.scatter(effect_x, effect_y, c='blue', s=40, marker='o', label='Effect', alpha=0.5)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Spatial Distribution of Causal Events')
        ax.legend()
        ax.set_xlim(0, min(30, Npos))
        ax.set_ylim(0, min(20, Npos))
        plt.savefig(os.path.join(RESULTS_DIR, 'spatial_distribution.png'), dpi=150)
        plt.close()

        print(f"[OK] 可视化图像已保存")
    except Exception as e:
        print(f"[WARN] 可视化失败 (matplotlib兼容性问题): {e}")
else:
    print("[SKIP] 跳过可视化 (ENABLE_VISUALIZATION=False)")


