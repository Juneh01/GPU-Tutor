"""
本地评测脚本
计算 ROUGE-L 分数
"""
import json
import argparse
import jieba
from rouge_chinese import Rouge

PREDICTIONS_PATH = "./predictions.json"


def load_predictions(path):
    """加载预测结果"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def tokenize_chinese(text):
    """中文分词"""
    return " ".join(jieba.cut(text))


def calculate_rouge_l(prediction, ground_truth):
    """计算单个样本的 ROUGE-L 分数"""
    if not prediction or not ground_truth:
        return 0.0
    
    # 分词
    pred_tokens = tokenize_chinese(prediction)
    gt_tokens = tokenize_chinese(ground_truth)
    
    if not pred_tokens.strip() or not gt_tokens.strip():
        return 0.0
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(pred_tokens, gt_tokens)
        return scores[0]["rouge-l"]["f"]
    except Exception as e:
        return 0.0


def evaluate(predictions_data):
    """评测所有样本"""
    scores = []
    
    for item in predictions_data:
        pred = item.get("prediction", "")
        gt = item.get("ground_truth", "")
        
        score = calculate_rouge_l(pred, gt)
        scores.append(score)
    
    return scores


def print_results(scores, threshold=0.35):
    """打印评测结果"""
    avg_score = sum(scores) / len(scores) if scores else 0
    passed = avg_score >= threshold
    
    print(f"\n{'='*60}")
    print(f"              本地评测结果（jieba分词版）")
    print(f"{'='*60}")
    print(f"评测样本数: {len(scores)}")
    print(f"平均准确率: {avg_score:.4f} (ROUGE-L F-measure)")
    print(f"及格阈值:   {threshold}")
    print(f"评测结果:   {'✅ 及格' if passed else '❌ 不及格'}")
    print(f"{'='*60}")
    
    # 分数分布
    bins = [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.01)]
    print("分数分布:")
    for low, high in bins:
        count = sum(1 for s in scores if low <= s < high)
        pct = count / len(scores) * 100
        bar = "█" * int(pct / 3)
        label = f"{low:.2f}-{high:.2f}" if high <= 1.0 else f"{low:.2f}-1.00"
        print(f"  {label}: {count:3d} ({pct:5.1f}%) {bar}")
    print(f"{'-'*60}\n")
    
    return avg_score, passed


def show_examples(predictions_data, scores, num_examples=5):
    """显示示例"""
    # 显示低分样本
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
    
    print(f"\n{'='*60}")
    print(f"低分样本示例（前{num_examples}个）")
    print(f"{'='*60}")
    
    for i in sorted_indices[:num_examples]:
        item = predictions_data[i]
        print(f"\n[样本 {i+1}] 分数: {scores[i]:.4f}")
        print(f"问题: {item['instruction'][:100]}...")
        print(f"预测: {item['prediction'][:100]}..." if item['prediction'] else "预测: (空)")
        print(f"答案: {item['ground_truth'][:100]}...")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地评测")
    parser.add_argument("-p", "--predictions", type=str, default=PREDICTIONS_PATH, help="预测结果路径")
    parser.add_argument("-t", "--threshold", type=float, default=0.35, help="及格阈值")
    parser.add_argument("--examples", type=int, default=5, help="显示示例数量")
    args = parser.parse_args()
    
    # 加载预测
    print(f"加载预测结果: {args.predictions}")
    predictions_data = load_predictions(args.predictions)
    print(f"共 {len(predictions_data)} 条数据")
    
    # 评测
    print("开始评测...")
    scores = evaluate(predictions_data)
    
    # 打印结果
    avg_score, passed = print_results(scores, args.threshold)
    
    # 显示示例
    show_examples(predictions_data, scores, args.examples)
    
    # 保存详细结果
    results = {
        "avg_score": avg_score,
        "passed": passed,
        "threshold": args.threshold,
        "num_samples": len(scores),
        "scores": scores,
    }
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果已保存到: evaluation_results.json")
