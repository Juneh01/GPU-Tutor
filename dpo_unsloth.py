"""
GPU课程模型 DPO 对齐训练 - Unsloth版本

支持两种模式：
1. LoRA模式（默认，省显存）
2. 全参数模式（效果可能更好，需要更多显存）

使用方法：
python dpo_unsloth.py --prepare          # 准备数据
python dpo_unsloth.py --train            # LoRA模式训练
python dpo_unsloth.py --train --full     # 全参数模式训练
"""

import os
import json
import torch
import random
from datasets import Dataset

# ==================== 配置 ====================
class Config:
    # 模型路径（使用SFT微调后的模型）
    BASE_MODEL = "./Qwen2.5-0.5B-largefinetuned-merged"
    OUTPUT_DIR = "./Qwen2.5-0.5B-gpu-dpo"
    
    # 训练配置
    EPOCHS = 1
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 5e-6        # DPO学习率要小一些
    MAX_SEQ_LENGTH = 512
    
    # DPO配置
    BETA = 0.1                  # DPO温度参数，越小越激进
    
    # LoRA配置（仅LoRA模式使用）
    LORA_R = 16
    LORA_ALPHA = 32
    
    # 数据路径
    DPO_DATA_FILE = "./finetune_data/dpo_data.json"
    
    # 数据源
    SFT_DATA_FILES = [
        "./data/processed_dataset_converted.json",
        "./finetune_data/train_sft_bonus_gt.json",
    ]

config = Config()

# ==================== 数据准备 ====================
def create_dpo_dataset():
    """
    创建DPO训练数据
    
    策略：用原始好答案作为chosen，构造差答案作为rejected
    """
    print("="*60)
    print("准备DPO数据")
    print("="*60)
    
    all_qa = []
    
    # 加载SFT数据
    for filepath in config.SFT_DATA_FILES:
        if not os.path.exists(filepath):
            print(f"  跳过不存在的文件: {filepath}")
            continue
        
        print(f"  加载: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            q = item.get("instruction") or item.get("question", "")
            a = item.get("output") or item.get("answer", "")
            if q and a and len(a) > 20:  # 过滤太短的答案
                all_qa.append((q.strip(), a.strip()))
    
    print(f"\n  加载了 {len(all_qa)} 个QA对")
    
    # 构造DPO数据
    dpo_data = []
    
    for question, good_answer in all_qa:
        # 构造差的回答（多种策略）
        bad_answer = create_bad_answer(question, good_answer)
        
        dpo_data.append({
            "prompt": question,
            "chosen": good_answer,
            "rejected": bad_answer,
        })
    
    # 打乱数据
    random.shuffle(dpo_data)
    
    # 保存
    os.makedirs(os.path.dirname(config.DPO_DATA_FILE), exist_ok=True)
    with open(config.DPO_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ DPO数据保存到: {config.DPO_DATA_FILE}")
    print(f"  样本数: {len(dpo_data)}")
    
    # 显示示例
    print("\n示例:")
    sample = dpo_data[0]
    print(f"  Prompt: {sample['prompt'][:60]}...")
    print(f"  Chosen: {sample['chosen'][:60]}...")
    print(f"  Rejected: {sample['rejected'][:60]}...")
    
    return dpo_data

def create_bad_answer(question: str, good_answer: str) -> str:
    """构造差的回答"""
    strategies = []
    
    # 策略1: 太短/敷衍
    strategies.append("这个问题比较复杂，需要具体情况具体分析。")
    strategies.append("不太清楚，建议查阅相关文档。")
    strategies.append("这涉及到很多方面。")
    
    # 策略2: 截断（信息不完整）
    if len(good_answer) > 30:
        strategies.append(good_answer[:25] + "...")
    
    # 策略3: 重复（低质量）
    if len(good_answer) > 40:
        first_part = good_answer[:20]
        strategies.append(first_part + first_part + first_part)
    
    # 策略4: 跑题
    strategies.append("GPU是一种图形处理器，主要用于图形渲染。")
    strategies.append("编程是一门需要不断学习的技术。")
    
    # 策略5: 格式混乱
    if len(good_answer) > 50:
        words = good_answer[:50].split()
        random.shuffle(words)
        strategies.append(' '.join(words))
    
    return random.choice(strategies)

# ==================== 格式化函数 ====================
def format_dpo_sample(example, tokenizer):
    """格式化DPO样本为对话格式"""
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    
    # 构建prompt部分
    prompt_text = f"""<|im_start|>system
GPU编程专家。简洁准确回答。<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    
    # chosen和rejected只需要回答部分
    chosen_text = chosen + "<|im_end|>"
    rejected_text = rejected + "<|im_end|>"
    
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }

# ==================== DPO训练 ====================
def train_dpo(use_lora: bool = True):
    """
    执行DPO训练
    
    Args:
        use_lora: True使用LoRA，False使用全参数微调
    """
    from unsloth import FastLanguageModel
    from trl import DPOTrainer, DPOConfig
    
    print("="*60)
    print(f"DPO对齐训练 ({'LoRA模式' if use_lora else '全参数模式'})")
    print("="*60)
    
    # 加载模型
    print("\n加载模型...")
    
    if use_lora:
        # LoRA模式
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.BASE_MODEL,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,  # 4bit加载省显存
        )
        
        # 添加LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print(f"  LoRA R: {config.LORA_R}, Alpha: {config.LORA_ALPHA}")
    else:
        # 全参数模式
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.BASE_MODEL,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=False,  # 全参数不用4bit
            full_finetuning=True,
        )
        print("  全参数微调模式")
    
    # 设置tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 加载DPO数据
    print("\n加载DPO数据...")
    if not os.path.exists(config.DPO_DATA_FILE):
        print("  数据不存在，先创建...")
        create_dpo_dataset()
    
    with open(config.DPO_DATA_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 格式化数据
    formatted_data = [format_dpo_sample(item, tokenizer) for item in raw_data]
    dataset = Dataset.from_list(formatted_data)
    
    print(f"  训练样本: {len(dataset)}")
    
    # DPO配置
    print("\n配置DPO训练...")
    
    dpo_config = DPOConfig(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        beta=config.BETA,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit" if use_lora else "adamw_torch",
        seed=42,
        report_to="none",
        max_length=config.MAX_SEQ_LENGTH,
        max_prompt_length=256,
    )
    
    # 创建DPO训练器
    print("\n开始DPO训练...")
    print(f"  Beta: {config.BETA}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Batch Size: {config.BATCH_SIZE} x {config.GRADIENT_ACCUMULATION}")
    
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth会自动处理
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # 训练
    dpo_trainer.train()
    
    # 保存
    print("\n保存模型...")
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"✓ 模型保存到: {config.OUTPUT_DIR}")
    
    # 合并并保存（仅LoRA模式需要）
    if use_lora:
        print("\n合并LoRA权重...")
        merged_dir = config.OUTPUT_DIR + "-merged"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"✓ 合并模型保存到: {merged_dir}")
    
    print("\n" + "="*60)
    print("✓ DPO训练完成！")
    print("="*60)
    
    if use_lora:
        print(f"\n使用合并后的模型进行推理:")
        print(f"  MODEL_PATH = '{config.OUTPUT_DIR}-merged'")
    else:
        print(f"\n使用训练后的模型进行推理:")
        print(f"  MODEL_PATH = '{config.OUTPUT_DIR}'")

# ==================== 测试模型 ====================
def test_model():
    """测试DPO训练后的模型"""
    from unsloth import FastLanguageModel
    
    model_path = config.OUTPUT_DIR + "-merged"
    if not os.path.exists(model_path):
        model_path = config.OUTPUT_DIR
    
    print(f"加载模型: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    # 测试问题
    test_questions = [
        "什么是CUDA？",
        "共享内存如何优化GPU程序？",
        "什么是Warp divergence？",
        "如何避免bank conflict？",
    ]
    
    print("\n" + "="*60)
    print("模型测试")
    print("="*60)
    
    for q in test_questions:
        prompt = f"""<|im_start|>system
GPU编程专家。简洁准确回答。<|im_end|>
<|im_start|>user
{q}<|im_end|>
<|im_start|>assistant
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"\nQ: {q}")
        print(f"A: {response[:200]}...")

# ==================== Main ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unsloth DPO训练")
    parser.add_argument("--prepare", action="store_true", help="准备DPO数据")
    parser.add_argument("--train", action="store_true", help="执行DPO训练")
    parser.add_argument("--full", action="store_true", help="使用全参数微调（默认LoRA）")
    parser.add_argument("--test", action="store_true", help="测试模型")
    args = parser.parse_args()
    
    if args.prepare:
        create_dpo_dataset()
    
    if args.train:
        use_lora = not args.full
        train_dpo(use_lora=use_lora)
    
    if args.test:
        test_model()
    
    if not any([args.prepare, args.train, args.test]):
        print("Unsloth DPO对齐训练")
        print("="*50)
        print("\n使用方法:")
        print("  python dpo_unsloth.py --prepare        # 准备DPO数据")
        print("  python dpo_unsloth.py --train          # LoRA模式训练")
        print("  python dpo_unsloth.py --train --full   # 全参数训练")
        print("  python dpo_unsloth.py --test           # 测试模型")
        print("\n推荐流程:")
        print("  1. python dpo_unsloth.py --prepare")
        print("  2. python dpo_unsloth.py --train")
        print("  3. python dpo_unsloth.py --test")
