"""
生成预测结果（Batch支持版）
用于本地评测
"""
import json
import requests
import argparse
from tqdm import tqdm

API_URL = "http://localhost:8000/predict"
TEST_DATA_PATH = "./data/test_ground_truth.json"
OUTPUT_PATH = "./predictions.json"


def load_test_data(path):
    """加载测试数据"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def generate_predictions_batch(data, batch_size=8):
    """批量生成预测"""
    predictions = []
    questions = [item.get("instruction", item.get("input", "")) for item in data]
    
    # 分批处理
    num_batches = (len(questions) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="生成预测"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(questions))
        batch_questions = questions[batch_start:batch_end]
        
        try:
            response = requests.post(
                API_URL,
                json={"prompt": batch_questions},
                timeout=120
            )
            
            if response.status_code == 200:
                responses = response.json().get("response", [])
                if isinstance(responses, list):
                    predictions.extend(responses)
                else:
                    predictions.append(responses)
            else:
                print(f"❌ Batch {i+1} 请求失败: {response.status_code}")
                predictions.extend([""] * len(batch_questions))
                
        except Exception as e:
            print(f"❌ Batch {i+1} 异常: {e}")
            predictions.extend([""] * len(batch_questions))
    
    return predictions


def generate_predictions_single(data):
    """单个生成预测（备用）"""
    predictions = []
    
    for item in tqdm(data, desc="生成预测"):
        question = item.get("instruction", item.get("input", ""))
        
        try:
            response = requests.post(
                API_URL,
                json={"prompt": question},
                timeout=60
            )
            
            if response.status_code == 200:
                pred = response.json().get("response", "")
                predictions.append(pred)
            else:
                predictions.append("")
                
        except Exception as e:
            print(f"❌ 异常: {e}")
            predictions.append("")
    
    return predictions


def save_predictions(data, predictions, output_path):
    """保存预测结果"""
    results = []
    for item, pred in zip(data, predictions):
        results.append({
            "instruction": item.get("instruction", item.get("input", "")),
            "ground_truth": item.get("output", ""),
            "prediction": pred,
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 预测结果已保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成预测结果")
    parser.add_argument("-d", "--data", type=str, default=TEST_DATA_PATH, help="测试数据路径")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_PATH, help="输出路径")
    parser.add_argument("-b", "--batch", type=int, default=5000, help="Batch大小")
    parser.add_argument("-u", "--url", type=str, default=API_URL, help="API地址")
    parser.add_argument("--single", action="store_true", help="使用单请求模式")
    args = parser.parse_args()
    
    API_URL = args.url
    
    # 加载数据
    print(f"加载测试数据: {args.data}")
    data = load_test_data(args.data)
    print(f"共 {len(data)} 条数据")
    
    # 生成预测
    if args.single:
        predictions = generate_predictions_single(data)
    else:
        predictions = generate_predictions_batch(data, args.batch)
    
    # 保存结果
    save_predictions(data, predictions, args.output)
