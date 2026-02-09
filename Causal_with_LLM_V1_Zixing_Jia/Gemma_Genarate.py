"""
因果推理测试脚本
从 train.jsonl 中随机抽取样本，使用 Gemma-2-9B-IT 模型进行因果推理回答
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
import os
from tqdm import tqdm
from datetime import datetime

# =================配置区域=================
# 模型路径配置
CACHE_DIR = "D:/huggingface_cache/hub2/hub"
MODEL_NAME = "google/gemma-2-9b-it"

# 数据路径配置
INPUT_FILE = "D:/Jiazixing/Vector-HASH-tinkering-main/Causal_Reasoning/train.jsonl"
OUTPUT_DIR = "D:/Jiazixing/Vector-HASH-tinkering-main/Gemma_anwer_causal"

# 已回答文件路径（排除这些 index，新回答追加到此文件）
EXISTING_ANSWERS_FILE = "D:/Jiazixing/Vector-HASH-tinkering-main/Gemma_anwer_causal/T_1.25_Gemma_Answer/gemma_causal_answers_100.jsonl"

# 采样和生成配置
N_SAMPLES = 700          # 随机抽取的样本数量
K_ANSWERS = 100            # 每条样本回答的次数

# 生成参数（参考 generate_pairs_2.py）
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.25
TOP_P = 0.95
DO_SAMPLE = True
# ========================================


def setup_model():
    """加载本地 Gemma 模型"""
    print(f"正在加载模型: {MODEL_NAME} ...")
    
    # 查找本地模型路径
    model_base = os.path.join(CACHE_DIR, "models--google--gemma-2-9b-it", "snapshots")
    
    if not os.path.exists(model_base):
        raise FileNotFoundError(f"模型目录不存在: {model_base}")
    
    snapshots = os.listdir(model_base)
    if not snapshots:
        raise FileNotFoundError("模型尚未下载，请先下载模型")
    
    local_model_path = os.path.join(model_base, snapshots[0])
    print(f"📁 本地模型路径: {local_model_path}")
    
    # 检查文件是否存在
    safetensor_files = [f for f in os.listdir(local_model_path) if f.endswith('.safetensors')]
    print(f"📦 找到 {len(safetensor_files)} 个 safetensor 文件")
    
    # 加载 tokenizer
    print("\n⏳ 正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True
    )
    print("✅ Tokenizer 加载成功！")
    
    # 加载模型
    print("\n⏳ 正在加载模型（约需 1-2 分钟）...")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"✅ 模型加载成功！")
    print(f"💻 模型设备: {model.device}")
    print(f"📊 模型参数量: {model.num_parameters() / 1e9:.2f}B")
    
    return tokenizer, model


def load_existing_indices(existing_file):
    """加载已回答过的 index 列表"""
    existing_indices = set()
    if os.path.exists(existing_file):
        print(f"\n📂 加载已回答文件: {existing_file}")
        with open(existing_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    existing_indices.add(record['index'])
        print(f"📊 已回答的 unique index 数: {len(existing_indices)}")
    else:
        print(f"⚠️ 已回答文件不存在: {existing_file}，将不排除任何样本")
    return existing_indices


def load_data(file_path, n_samples, exclude_indices=None):
    """从 jsonl 文件加载数据并随机抽样，同时返回完整数据集用于 few-shot"""
    print(f"\n📂 正在加载数据: {file_path}")
    
    all_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    
    print(f"📊 总样本数: {len(all_data)}")
    
    # 排除已回答的 index
    if exclude_indices:
        available_data = [item for item in all_data if item['index'] not in exclude_indices]
        print(f"🚫 排除已回答的 {len(exclude_indices)} 个 index")
        print(f"📊 可用于抽样的样本数: {len(available_data)}")
    else:
        available_data = all_data
    
    # 随机抽样
    if n_samples > len(available_data):
        print(f"⚠️ 请求样本数 ({n_samples}) 大于可用样本数 ({len(available_data)})，将使用全部可用数据")
        n_samples = len(available_data)
    
    sampled_data = random.sample(available_data, n_samples)
    print(f"🎲 随机抽取样本数: {len(sampled_data)}")
    
    return sampled_data, all_data


def get_few_shot_examples(all_data, current_index, ask_for, n_examples=3):
    """
    从数据集中获取 few-shot 示例
    
    Args:
        all_data: 完整数据集
        current_index: 当前样本的 index（避免使用）
        ask_for: "effect" 或 "cause"
        n_examples: 示例数量
    
    Returns:
        示例列表，每个元素是 (premise, answer) 元组
    """
    # 筛选相同类型（effect/cause）且不是当前样本的数据
    candidates = [
        item for item in all_data 
        if item["ask-for"] == ask_for and item["index"] != current_index
    ]
    
    # 随机选择 n_examples 个示例
    if len(candidates) < n_examples:
        selected = candidates
    else:
        selected = random.sample(candidates, n_examples)
    
    examples = []
    for item in selected:
        premise = item["premise"]
        # 根据 label 选择正确的 hypothesis 作为答案
        if item["label"] == 0:
            answer = item["hypothesis1"]
        else:
            answer = item["hypothesis2"]
        examples.append((premise, answer))
    
    return examples

def construct_prompt(premise, ask_for, all_data=None, current_index=None, n_examples=3):
    """
    Construct a prompt based on premise and ask_for, including few-shot examples.

    Args:
        premise: the premise / event description (string)
        ask_for: either "effect" or "cause"
        all_data: the full dataset (used to sample few-shot examples)
        current_index: index of the current sample (to avoid using the same sample as an example)
        n_examples: number of few-shot examples to include

    Returns:
        A prompt string ready to feed into the model.
    """
    # Gather few-shot examples
    examples_text = ""
    if all_data is not None and current_index is not None:
        examples = get_few_shot_examples(all_data, current_index, ask_for, n_examples)
        
        if ask_for == "effect":
            examples_text = "Here are some examples:\n\n"
            for i, (ex_premise, ex_answer) in enumerate(examples, 1):
                examples_text += f"Example {i}:\nPremise: {ex_premise}\nResult: {ex_answer}\n\n"
        else:  # cause
            examples_text = "Here are some examples:\n\n"
            for i, (ex_premise, ex_answer) in enumerate(examples, 1):
                examples_text += f"Example {i}:\nResult: {ex_premise}\nCause: {ex_answer}\n\n"
    
    if ask_for == "effect":
        prompt = f"""You are a causal inference expert. Given a premise, infer a possible outcome or effect.

{examples_text}Now please answer:

Premise: {premise}

Please answer in one short sentence: what result might this premise lead to?

Requirements:
1. Provide only one most likely result.
2. Keep the answer concise, no more than 40 words.
3. Give the answer directly, do not explain the reasoning process.
4. All answers must be in English.

Result:"""
    else:  # ask_for == "cause"
        prompt = f"""You are a causal inference expert. Given an outcome, infer a possible cause.

{examples_text}Now please answer:

Result: {premise}

Please answer in one short sentence: what cause might have led to this result?

Requirements:
1. Provide only one most likely cause.
2. Keep the answer concise, no more than 30 words.
3. Give the answer directly, do not explain the reasoning process.

Cause:"""
    
    return prompt



def generate_answer(tokenizer, model, prompt):
    """
    使用模型生成回答
    
    Args:
        tokenizer: 分词器
        model: 语言模型
        prompt: 输入 prompt
    
    Returns:
        生成的回答文本
    """
    # 使用 chat template
    chat = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 只提取新生成的部分
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    # 清理回答（去除多余的空白和换行）
    answer = generated_text.strip()
    
    # 如果回答过长，截取第一句
    if len(answer) > 200:
        # 尝试在句号处截断
        for sep in ['。', '.', '！', '!', '？', '?', '\n']:
            if sep in answer:
                answer = answer.split(sep)[0] + sep
                break
    
    return answer


def run_inference(tokenizer, model, sampled_data, all_data, k_answers, output_file, n_examples=3):
    """
    对抽样数据进行推理
    
    Args:
        tokenizer: 分词器
        model: 语言模型
        sampled_data: 抽样的数据列表
        all_data: 完整数据集（用于 few-shot 示例）
        k_answers: 每条样本回答的次数
        output_file: 输出文件路径（追加模式）
        n_examples: few-shot 示例数量
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\n🚀 开始推理...")
    print(f"📝 输出文件（追加模式）: {output_file}")
    print(f"📊 新样本数: {len(sampled_data)}, 每条回答 {k_answers} 次")
    print(f"📈 本次生成次数: {len(sampled_data) * k_answers}")
    print(f"📚 每个 prompt 包含 {n_examples} 个 few-shot 示例")
    
    results = []
    
    # 使用 tqdm 显示进度
    with tqdm(total=len(sampled_data) * k_answers, desc="生成回答") as pbar:
        for item in sampled_data:
            index = item["index"]
            premise = item["premise"]
            ask_for = item["ask-for"]
            
            # 构建 prompt（包含 few-shot 示例）
            prompt = construct_prompt(premise, ask_for, all_data, index, n_examples)
            
            # 生成 k 次回答
            for k in range(k_answers):
                answer = generate_answer(tokenizer, model, prompt)
                
                # 构建结果记录
                record = {
                    "index": index,
                    "premise": premise,
                    "ask-for": ask_for,
                    "answer_round": k + 1,
                    "Gemma_answer": answer
                }
                
                results.append(record)
                
                # 实时写入文件
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                pbar.update(1)
    
    print(f"\n✅ 推理完成！")
    print(f"📁 结果已保存到: {output_file}")
    print(f"📊 共生成 {len(results)} 条回答")
    
    return results, output_file


def print_sample_results(results, n=5):
    """打印部分结果示例"""
    print(f"\n📋 结果示例（前 {n} 条）:")
    print("=" * 80)
    
    for i, record in enumerate(results[:n]):
        print(f"\n[{i+1}] Index: {record['index']}")
        print(f"    Premise: {record['premise']}")
        print(f"    Ask-for: {record['ask-for']}")
        print(f"    Round: {record['answer_round']}")
        print(f"    Gemma Answer: {record['Gemma_answer']}")
        print("-" * 80)


def main():
    """主函数"""
    print("=" * 60)
    print("🔬 因果推理测试 - Gemma-2-9B-IT")
    print("=" * 60)
    
    # 设置随机种子（可选，用于复现）
    random.seed(42)
    
    # 1. 加载模型
    tokenizer, model = setup_model()
    
    # 2. 加载已回答的 index（用于排除）
    existing_indices = load_existing_indices(EXISTING_ANSWERS_FILE)
    
    # 3. 加载并抽样数据（排除已回答的，同时获取完整数据集用于 few-shot）
    sampled_data, all_data = load_data(INPUT_FILE, N_SAMPLES, exclude_indices=existing_indices)
    
    # 4. 进行推理（追加到已有文件）
    results, output_file = run_inference(
        tokenizer, 
        model, 
        sampled_data,
        all_data,
        K_ANSWERS, 
        EXISTING_ANSWERS_FILE,  # 直接追加到已有文件
        n_examples=3  # few-shot 示例数量
    )
    
    # 4. 打印示例结果
    print_sample_results(results)
    
    # 5. 统计信息
    print("\n📊 统计信息:")
    effect_count = sum(1 for r in results if r["ask-for"] == "effect")
    cause_count = sum(1 for r in results if r["ask-for"] == "cause")
    print(f"   - Effect 类型回答: {effect_count}")
    print(f"   - Cause 类型回答: {cause_count}")
    
    print("\n🎉 任务完成！")


if __name__ == "__main__":
    main()
