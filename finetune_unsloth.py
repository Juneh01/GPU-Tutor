"""
GPU课程模型微调脚本 - Unsloth框架

目标：让模型内化GPU知识，减少对RAG的依赖，提升推理速度

使用方法：
1. 安装依赖：pip install unsloth
2. 准备数据：将所有训练数据放到 ./finetune_data/ 目录
3. 运行：python finetune_unsloth.py
"""

import os
import json
import torch
from datasets import Dataset

# ==================== 配置 ====================
class Config:
    # 模型配置
    BASE_MODEL = "./local-model/Qwen2.5-0.5B"  # 或本地路径 "./local-model/Qwen3-0.6B" /home/howard/Workspace/IEEEAICAS/code/Qwen2.5-0.5B
    OUTPUT_DIR = "./Qwen2.5-0.5B-largefinetuned"
    
    # LoRA配置
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05
    
    # 训练配置
    EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 1024
    
    # 数据路径
    DATA_FILES = [
        "./data/processed_dataset_converted.json",
        "./finetune_data/train_sft_bonus_gt.json",
        "./finetune_data/train_dataset_basebonus.json",
        "./finetune_data/doubao_generated.json",  # 豆包生成的数据
        #"./finetune_data/rag_qna_pairs.json",     # RAG中的QA对
    ]

config = Config()

# ==================== 1. 加载数据 ====================
def load_all_data():
    """加载并合并所有训练数据"""
    all_data = []
    
    for filepath in config.DATA_FILES:
        if not os.path.exists(filepath):
            print(f"⚠ 文件不存在，跳过: {filepath}")
            continue
        
        print(f"加载: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # 支持多种数据格式
            if "instruction" in item and "output" in item:
                q = item["instruction"]
                a = item["output"]
            elif "question" in item and "answer" in item:
                q = item["question"]
                a = item["answer"]
            elif "input" in item and "output" in item:
                q = item["input"]
                a = item["output"]
            elif "prompt" in item and "response" in item:
                q = item["prompt"]
                a = item["response"]
            else:
                continue
            
            # 过滤空数据
            if not q or not a:
                continue
            
            all_data.append({
                "instruction": q.strip(),
                "output": a.strip()
            })
    
    print(f"\n总数据量: {len(all_data)}")
    return all_data

def format_prompt(example):
    """格式化为对话格式"""
    instruction = example["instruction"]
    output = example["output"]
    
    # Qwen3对话格式
    text = f"""<|im_start|>system
你是GPU编程专家，请简洁准确地回答问题。<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    
    return {"text": text}

# ==================== 2. 加载模型 ====================
def load_model():
    """使用Unsloth加载模型"""
    from unsloth import FastLanguageModel
    
    print("\n" + "="*60)
    print("加载模型...")
    print("="*60)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,  # 自动检测
        load_in_4bit=True,  # 4bit加载，节省显存
    )
    
    # 添加LoRA适配器
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print("✓ 模型加载完成")
    return model, tokenizer

# ==================== 3. 训练 ====================
def train():
    """执行训练"""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    # 加载数据
    print("\n" + "="*60)
    print("准备数据...")
    print("="*60)
    
    raw_data = load_all_data()
    if not raw_data:
        print("❌ 没有有效数据！")
        return
    
    # 转换为Dataset
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    print(f"训练样本数: {len(dataset)}")
    print(f"样本示例:\n{dataset[0]['text'][:500]}...")
    
    # 加载模型
    model, tokenizer = load_model()
    
    # 训练参数
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    print("\n" + "="*60)
    print("保存模型...")
    print("="*60)
    
    # 保存LoRA权重
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"✓ LoRA权重保存到: {config.OUTPUT_DIR}")
    
    # 合并并保存完整模型
    merged_dir = config.OUTPUT_DIR + "-merged"
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"✓ 合并模型保存到: {merged_dir}")
    
    print("\n" + "="*60)
    print("✓ 训练完成！")
    print("="*60)

# ==================== 4. 从RAG提取QA对 ====================
def extract_qa_from_rag():
    """从RAG索引中提取QA对用于微调"""
    import pickle
    
    rag_dirs = [
        "./rag_indexes/rag_qna",
        "./rag_indexes/rag_cuda",
    ]
    
    qa_pairs = []
    
    for rag_dir in rag_dirs:
        docs_path = os.path.join(rag_dir, "documents.pkl")
        if not os.path.exists(docs_path):
            continue
        
        with open(docs_path, 'rb') as f:
            documents = pickle.load(f)
        
        for doc in documents:
            text = doc.get('text', str(doc)) if isinstance(doc, dict) else str(doc)
            
            # 尝试解析QA格式
            if "问题：" in text and "答案：" in text:
                parts = text.split("答案：", 1)
                if len(parts) == 2:
                    q = parts[0].replace("问题：", "").strip()
                    a = parts[1].strip()
                    if q and a:
                        qa_pairs.append({"question": q, "answer": a})
            elif "Q:" in text and "A:" in text:
                parts = text.split("A:", 1)
                if len(parts) == 2:
                    q = parts[0].replace("Q:", "").strip()
                    a = parts[1].strip()
                    if q and a:
                        qa_pairs.append({"question": q, "answer": a})
    
    # 保存
    os.makedirs("./finetune_data", exist_ok=True)
    output_path = "./finetune_data/rag_qna_pairs.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 从RAG提取了 {len(qa_pairs)} 个QA对，保存到 {output_path}")
    return qa_pairs

# ==================== 5. 量化导出 ====================
def quantize_model():
    """量化微调后的模型"""
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.transformers import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    merged_dir = config.OUTPUT_DIR + "-merged"
    output_dir = config.OUTPUT_DIR + "-w4a16"
    
    print(f"加载模型: {merged_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
    
    # 准备校准数据
    calibration_data = [
        "什么是CUDA？",
        "共享内存如何优化？",
        "什么是Warp？",
        "如何避免bank conflict？",
        "GPU的内存层次结构是什么？",
    ]
    
    # GPTQ量化
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )
    
    oneshot(
        model=model,
        dataset=calibration_data,
        recipe=recipe,
        output_dir=output_dir,
        tokenizer=tokenizer,
    )
    
    print(f"✓ 量化模型保存到: {output_dir}")

# ==================== Main ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-rag", action="store_true", help="从RAG提取QA对")
    parser.add_argument("--train", action="store_true", help="执行训练")
    parser.add_argument("--quantize", action="store_true", help="量化模型")
    parser.add_argument("--all", action="store_true", help="执行所有步骤")
    args = parser.parse_args()
    
    if args.extract_rag or args.all:
        extract_qa_from_rag()
    
    if args.train or args.all:
        train()
    
    if args.quantize or args.all:
        quantize_model()
    
    if not any([args.extract_rag, args.train, args.quantize, args.all]):
        print("使用方法:")
        print("  python finetune_unsloth.py --extract-rag  # 从RAG提取QA对")
        print("  python finetune_unsloth.py --train        # 执行训练")
        print("  python finetune_unsloth.py --quantize     # 量化模型")
        print("  python finetune_unsloth.py --all          # 执行所有步骤")
