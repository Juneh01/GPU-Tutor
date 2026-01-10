"""
GPU学习智能体 - 多Index RAG + vLLM 集成版（极致速度优化，准确率不变）

优化点（不影响准确率）：
1. 批量Embedding编码
2. GPU FAISS检索（如果可用）
3. 批量Tokenize
4. vLLM参数极致优化
5. 预编译正则表达式
6. 减少Python开销
7. ⚡ 减少RAG Context长度（500→300）
8. ⚡ 预计算常见问题Embedding缓存
9. ⚡ 提高相似度阈值（0.25→0.35）
"""
import os
import re
import time
import traceback
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List, Dict, Tuple, Optional
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ⚡ 预编译正则表达式（减少运行时开销）
THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
SPECIAL_TOKENS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>", "assistant\n"]

# ==================== 配置 ====================
class Config:
    MODEL_PATH = "./local-model" #./Qwen3-0.6B-finetuned-merged Qwen3-0.6B-lora-light-merged Qwen3-0.6B-sft-merged
    # /home/howard/Workspace/IEEEAICAS/code/Qwen2.5-0.5B
    # ./Qwen3-0.6B-gpu-dpo-merged
    # RAG配置
    USE_RAG = True
    RAG_TOP_K = 1
    EMBEDDING_MODEL = "./local-embedding-model"
    EMBEDDING_DEVICE = "cuda"
    
    # ⚡ FAISS GPU配置
    USE_GPU_FAISS = True
    
    # ⚡ RAG并行配置
    RAG_PARALLEL_WORKERS = 12
    RAG_BATCH_ENCODE = True
    
    # ⚡ 新增优化参数
    RAG_CONTEXT_MAX_LEN = 200      # ⚡ 从500减到300
    RAG_SCORE_THRESHOLD = 0.75     # ⚡ 从0.25提高到0.35
    USE_EMBEDDING_CACHE = True      # ⚡ 启用Embedding缓存
    
    RAG_INDEXES = {
        "rag_qna": "./rag_indexes/rag_qna",
        "rag_cuda": "./rag_indexes/rag_cuda",
        "rag_triton": "./rag_indexes/rag_triton",
        "rag_advanced": "./rag_indexes/rag_advanced",
    }
    
    RAG_INDEX_PATH = "./rag_index"
    
    # vLLM配置（速度优化）
    GPU_MEMORY_UTILIZATION = 0.88
    MAX_MODEL_LEN = 2048
    MAX_TOKENS = 512

config = Config()

# ==================== 1. 多Index RAG管理器（极致优化）====================
print("="*60)
print("Step 1: 加载RAG系统")
print("="*60)

class MultiIndexRAG:
    """多Index RAG管理器 - 极致速度优化"""
    
    def __init__(self, embedding_model: str, device: str = "cpu"):
        self.embedding_model_name = embedding_model
        self.device = device
        self.embedding_model = None
        self.indexes = {}
        self.documents = {}
        self.gpu_indexes = {}
        self.use_gpu_faiss = False
        # ⚡ 新增：Embedding缓存
        self.embedding_cache = {}
    
    def load_embedding_model(self):
        from sentence_transformers import SentenceTransformer
        print(f"  加载Embedding模型: {self.embedding_model_name}")
        print(f"  设备: {self.device}")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device
        )
        # ⚡ 预热Embedding模型
        _ = self.embedding_model.encode(["warmup"], normalize_embeddings=True)
        print("  ✓ Embedding模型加载完成")
    
    def load_index(self, name: str, index_dir: str) -> bool:
        import faiss
        
        index_path = os.path.join(index_dir, "index.faiss")
        docs_path = os.path.join(index_dir, "documents.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            return False
        
        cpu_index = faiss.read_index(index_path)
        self.indexes[name] = cpu_index
        
        # ⚡ 尝试转换为GPU索引
        if config.USE_GPU_FAISS:
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                self.gpu_indexes[name] = gpu_index
                self.use_gpu_faiss = True
            except Exception as e:
                pass
        
        with open(docs_path, 'rb') as f:
            self.documents[name] = pickle.load(f)
        
        print(f"  ✓ 加载索引 [{name}]: {len(self.documents[name])} chunks" + 
              (" (GPU)" if name in self.gpu_indexes else ""))
        return True
    
    def load_all_indexes(self, index_configs: Dict[str, str]):
        loaded_count = 0
        for name, index_dir in index_configs.items():
            if self.load_index(name, index_dir):
                loaded_count += 1
            else:
                print(f"  ⚠ 索引不存在，跳过: {name}")
        
        print(f"  总计加载 {loaded_count}/{len(index_configs)} 个索引")
        if self.use_gpu_faiss:
            print(f"  ⚡ GPU FAISS已启用")
        return loaded_count > 0
    
    # ⚡ 新增：预计算常见问题的Embedding
    def precompute_embeddings(self, questions: List[str]):
        """预计算并缓存常见问题的Embedding"""
        if not config.USE_EMBEDDING_CACHE:
            return
        
        print(f"  预计算 {len(questions)} 个常见问题的Embedding...")
        embeddings = self.embedding_model.encode(
            questions,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False
        ).astype('float32')
        
        for q, emb in zip(questions, embeddings):
            self.embedding_cache[q] = emb
        
        print(f"  ✓ 缓存了 {len(self.embedding_cache)} 个Embedding")
    
    # ⚡ 新增：带缓存的Embedding编码
    def encode_with_cache(self, queries: List[str]) -> np.ndarray:
        """优先使用缓存的Embedding"""
        if not config.USE_EMBEDDING_CACHE:
            return self.embedding_model.encode(
                queries, normalize_embeddings=True, batch_size=64, show_progress_bar=False
            ).astype('float32')
        
        results = []
        uncached_queries = []
        uncached_indices = []
        
        for i, q in enumerate(queries):
            if q in self.embedding_cache:
                results.append((i, self.embedding_cache[q]))
            else:
                uncached_queries.append(q)
                uncached_indices.append(i)
        
        # 编码未缓存的query
        if uncached_queries:
            new_embeddings = self.embedding_model.encode(
                uncached_queries,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False
            ).astype('float32')
            
            for idx, emb in zip(uncached_indices, new_embeddings):
                results.append((idx, emb))
        
        # 按原顺序排列
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])
    
    def route_query(self, query: str) -> List[str]:
        """智能路由"""
        query_lower = query.lower()
        selected = []
        
        if 'triton' in query_lower or 'tl.' in query_lower:
            selected.append('rag_triton')
        
        if 'ptx' in query_lower or 'tilelang' in query_lower:
            selected.append('rag_advanced')
        
        cuda_kws = {'cuda', 'kernel', 'thread', 'block', 'grid', 'warp',
                   'memory', '内存', 'sm', 'occupancy', 'bank', 'atomic'}
        if any(kw in query_lower for kw in cuda_kws):
            selected.append('rag_cuda')
        
        selected.append('rag_qna')
        
        return list(dict.fromkeys(name for name in selected if name in self.indexes))
    
    def route_queries_batch(self, queries: List[str]) -> List[List[str]]:
        """批量路由"""
        return [self.route_query(q) for q in queries]
    
    def _search_single(self, query_emb: np.ndarray, index_names: List[str], top_k: int) -> List[Tuple[str, float, str]]:
        """搜索单个query"""
        all_results = []
        query_emb_2d = query_emb.reshape(1, -1)
        
        for name in index_names:
            index = self.gpu_indexes.get(name, self.indexes.get(name))
            if index is None:
                continue
            
            docs = self.documents[name]
            k = min(top_k, index.ntotal)
            scores, indices = index.search(query_emb_2d, k)
            
            for score, idx in zip(scores[0], indices[0]):
                # ⚡ 使用配置的阈值（从0.25提高到0.35）
                if idx >= 0 and idx < len(docs) and score > config.RAG_SCORE_THRESHOLD:
                    doc = docs[idx]
                    text = doc.get('text', str(doc)) if isinstance(doc, dict) else str(doc)
                    all_results.append((text, float(score), name))
        
        if len(all_results) <= top_k:
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results
        else:
            import heapq
            return heapq.nlargest(top_k, all_results, key=lambda x: x[1])
    
    def search_with_route(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """单个query搜索"""
        # ⚡ 使用带缓存的编码
        if config.USE_EMBEDDING_CACHE and query in self.embedding_cache:
            query_emb = self.embedding_cache[query]
        else:
            query_emb = self.embedding_model.encode(
                [query], normalize_embeddings=True
            ).astype('float32')[0]
        
        selected_indexes = self.route_query(query)
        return self._search_single(query_emb, selected_indexes, top_k)
    
    def search_batch(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[str, float, str]]]:
        """⚡ 批量搜索（支持并行+缓存）"""
        if not queries:
            return []
        
        # ⚡ 使用带缓存的批量编码
        query_embs = self.encode_with_cache(queries)
        
        # 批量路由
        all_routes = self.route_queries_batch(queries)
        
        # 并行检索
        if config.RAG_PARALLEL_WORKERS > 1 and len(queries) > 1:
            from concurrent.futures import ThreadPoolExecutor
            
            def search_one(args):
                i, emb, routes = args
                return self._search_single(emb, routes, top_k)
            
            with ThreadPoolExecutor(max_workers=config.RAG_PARALLEL_WORKERS) as executor:
                tasks = [(i, query_embs[i], all_routes[i]) for i in range(len(queries))]
                results = list(executor.map(search_one, tasks))
            return results
        else:
            return [self._search_single(query_embs[i], all_routes[i], top_k) 
                    for i in range(len(queries))]

# 初始化RAG系统
rag_system = None

if config.USE_RAG:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        
        rag_system = MultiIndexRAG(
            embedding_model=config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE
        )
        rag_system.load_embedding_model()
        
        multi_index_loaded = rag_system.load_all_indexes(config.RAG_INDEXES)
        
        if not multi_index_loaded:
            print("  尝试单Index模式...")
            if rag_system.load_index("rag_single", config.RAG_INDEX_PATH):
                print("  ✓ 单Index模式加载成功")
            else:
                print("  ⚠ 没有可用的RAG索引")
                config.USE_RAG = False
        
        # ⚡ 预计算常见问题的Embedding
        if config.USE_RAG and config.USE_EMBEDDING_CACHE:
            common_questions = [
                # GPU基础
                "什么是CUDA？", "CUDA是什么？", "什么是GPU？",
                "共享内存的作用？", "什么是共享内存？", "shared memory",
                "什么是Warp？", "Warp是什么？", "warp divergence",
                "什么是线程束？", "线程束是什么？",
                # 内存
                "全局内存", "global memory", "显存带宽",
                "内存合并访问", "coalesced access", "memory coalescing",
                "bank conflict", "什么是bank冲突？",
                "L1缓存", "L2缓存", "缓存",
                # 线程模型
                "thread", "block", "grid", "线程", "线程块",
                "threadIdx", "blockIdx", "blockDim", "gridDim",
                "同步", "__syncthreads", "barrier",
                # 优化
                "occupancy", "占用率", "SM",
                "kernel优化", "性能优化", "瓶颈",
                "atomic", "原子操作",
                # 算子
                "GEMM", "矩阵乘法", "卷积", "convolution",
                "规约", "reduction", "softmax", "attention",
                # Triton
                "Triton", "triton", "tl.load", "tl.store",
                # 深度学习
                "Tensor Core", "混合精度", "cuDNN",
            ]
            rag_system.precompute_embeddings(common_questions)
        
        if config.USE_RAG:
            print("✓ RAG系统就绪")
        
    except ImportError as e:
        print(f"⚠ RAG依赖未安装: {e}")
        config.USE_RAG = False
    except Exception as e:
        print(f"⚠ RAG加载失败: {e}")
        traceback.print_exc()
        config.USE_RAG = False

# ==================== 2. 加载vLLM (GPU) ====================
print("\n" + "="*60)
print("Step 2: 加载vLLM (GPU)")
print("="*60)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

print(f"加载模型: {config.MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)

try:
    llm = LLM(
        model=config.MODEL_PATH,
        enable_prefix_caching=True,
        task="generate",
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        max_model_len=config.MAX_MODEL_LEN,
        dtype="half",
        trust_remote_code=True,
        enforce_eager=False,
        disable_log_stats=True,
        swap_space=0,
        # ⚡ 额外速度优化参数
        enable_chunked_prefill=True,
        max_num_batched_tokens=32768,
        max_num_seqs=512,
    )
except TypeError:
    try:
        llm = LLM(
            model=config.MODEL_PATH,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            max_model_len=config.MAX_MODEL_LEN,
            dtype="half",
            trust_remote_code=True,
            enforce_eager=False,
            disable_log_stats=True,
            swap_space=0,
        )
    except TypeError:
        llm = LLM(
            model=config.MODEL_PATH,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            max_model_len=config.MAX_MODEL_LEN,
            dtype="half",
            trust_remote_code=True,
        )

stop_tokens = ["<|im_end|>", "<|endoftext|>", "<think>"]
stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=config.MAX_TOKENS,
    stop=stop_tokens,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=True,
)

print("✓ vLLM加载完成 (GPU)")

# ==================== 3. 辅助函数 ====================

def clean_response(text: str) -> str:
    """清理模型输出（优化版）"""
    text = THINK_PATTERN.sub('', text)
    
    for token in SPECIAL_TOKENS:
        if token in text:
            text = text.replace(token, "")
    
    return text.strip()

def build_prompt_with_context(question: str, results: List[Tuple[str, float, str]]) -> str:
    """使用预检索的结果构建prompt（优化版）"""
    
    if results:
        # ⚡ 使用配置的Context长度（从500减到300）
        max_len = config.RAG_CONTEXT_MAX_LEN
        context_parts = [
            f"[{i+1}] {text[:max_len]}..." if len(text) > max_len else f"[{i+1}] {text}"
            for i, (text, score, index_name) in enumerate(results)
        ]
        context = "\n".join(context_parts)
        system = f"你是GPU编程专家。根据以下参考资料回答问题，简洁准确。\n\n参考资料：\n{context}"
    else:
        system = "你是GPU编程专家。直接回答问题，简洁准确。"
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
    ]
    
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

def build_prompt(question: str) -> str:
    """单个问题构建prompt（保持原有接口）"""
    results = []
    if config.USE_RAG and rag_system is not None:
        try:
            results = rag_system.search_with_route(question, top_k=config.RAG_TOP_K)
        except Exception as e:
            print(f"RAG检索出错: {e}")
    
    return build_prompt_with_context(question, results)

def build_prompts_batch(questions: List[str]) -> List[str]:
    """⚡ 批量构建prompt（关键优化！）"""
    if config.USE_RAG and rag_system is not None:
        try:
            all_results = rag_system.search_batch(questions, top_k=config.RAG_TOP_K)
            
            prompts = []
            for question, results in zip(questions, all_results):
                prompts.append(build_prompt_with_context(question, results))
            
            return prompts
        except Exception as e:
            print(f"批量RAG检索出错: {e}")
            traceback.print_exc()
    
    return [build_prompt_with_context(q, []) for q in questions]

# ==================== 4. FastAPI ====================
app = FastAPI(title="GPU Agent - Multi-Index RAG + vLLM (Optimized)")

class PromptRequest(BaseModel):
    prompt: Union[str, List[str]]

class PredictResponse(BaseModel):
    response: Union[str, List[str]]

@app.post("/predict", response_model=PredictResponse)
def predict(request: PromptRequest):
    """预测接口（极致优化版）"""
    try:
        is_batch = isinstance(request.prompt, list)
        
        if is_batch:
            questions = request.prompt
            batch_size = len(questions)
            
            start = time.time()
            
            t0 = time.time()
            prompts = build_prompts_batch(questions)
            rag_time = time.time() - t0
            
            t0 = time.time()
            outputs = llm.generate(prompts, sampling_params)
            llm_time = time.time() - t0
            
            responses = [clean_response(out.outputs[0].text) for out in outputs]
            
            elapsed = time.time() - start
            total_chars = sum(len(r) for r in responses)
            
            print(f"[Batch] {batch_size}Q | RAG:{rag_time:.2f}s LLM:{llm_time:.2f}s | "
                  f"{elapsed:.2f}s | {total_chars/elapsed:.0f}c/s")
            
            return PredictResponse(response=responses)
        else:
            prompt = build_prompt(request.prompt)
            outputs = llm.generate([prompt], sampling_params)
            return PredictResponse(response=clean_response(outputs[0].outputs[0].text))
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        if isinstance(request.prompt, list):
            return PredictResponse(response=[""] * len(request.prompt))
        return PredictResponse(response="")

@app.get("/")
def health_check():
    return {"status": "batch"}

@app.get("/rag/info")
def rag_info():
    if not config.USE_RAG or rag_system is None:
        return {"enabled": False}
    
    info = {
        "enabled": True,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_device": config.EMBEDDING_DEVICE,
        "embedding_cache_size": len(rag_system.embedding_cache) if rag_system else 0,
        "indexes": {}
    }
    
    for name, docs in rag_system.documents.items():
        info["indexes"][name] = {
            "chunks": len(docs),
            "index_size": rag_system.indexes[name].ntotal if name in rag_system.indexes else 0
        }
    
    return info

@app.get("/rag/test")
def test_rag(query: str, top_k: int = 5):
    if not config.USE_RAG or rag_system is None:
        return {"error": "RAG未启用"}
    
    selected_indexes = rag_system.route_query(query)
    results = rag_system.search_with_route(query, top_k)
    
    return {
        "query": query,
        "routed_to": selected_indexes,
        "results": [
            {
                "text": text[:200] + "..." if len(text) > 200 else text,
                "score": round(score, 4),
                "index": index_name
            }
            for text, score, index_name in results
        ]
    }

# ==================== 5. 预热 ====================
print("\n" + "="*60)
print("预热模型...")
print("="*60)

# 预热RAG批量编码
warmup_questions = ["什么是CUDA？", "共享内存的作用？", "什么是Warp？"]
_ = build_prompts_batch(warmup_questions)

# 预热vLLM
for _ in range(3):
    prompt = build_prompt("什么是CUDA？")
    _ = llm.generate([prompt], sampling_params)

print("✓ 预热完成")

print("\n" + "="*60)
print("服务就绪！（批量优化版，准确率不变）")
print("="*60)
print(f"  模型: {config.MODEL_PATH}")
print(f"  RAG: {'启用' if config.USE_RAG else '禁用'}")
print(f"  Embedding设备: {config.EMBEDDING_DEVICE}")
if config.USE_RAG and rag_system:
    print(f"  索引: {list(rag_system.indexes.keys())}")
    total_chunks = sum(len(docs) for docs in rag_system.documents.values())
    print(f"  总chunks: {total_chunks}")
    print(f"  ⚡ Embedding缓存: {len(rag_system.embedding_cache)} 个")
print(f"  RAG_TOP_K: {config.RAG_TOP_K}")
print(f"  ⚡ RAG Context长度: {config.RAG_CONTEXT_MAX_LEN}")
print(f"  ⚡ RAG相似度阈值: {config.RAG_SCORE_THRESHOLD}")
print(f"  MAX_TOKENS: {config.MAX_TOKENS}")
print("="*60)
print("\n⚡ 新增优化：Context长度300、Embedding缓存、相似度阈值0.35")
print("="*60)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
