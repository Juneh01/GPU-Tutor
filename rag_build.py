"""
RAG第二阶段：专业技术文档知识库构建指南

架构设计：
┌─────────────────────────────────────────────────────────────┐
│                      RAG Multi-Index 架构                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  用户问题 ──→ Router（路由决策）                             │
│                    │                                         │
│         ┌─────────┼─────────┐                               │
│         ↓         ↓         ↓                               │
│    rag_qna    rag_cuda   rag_triton                         │
│    (Q&A库)   (CUDA文档)  (Triton库)                         │
│         │         │         │                               │
│         └─────────┴─────────┘                               │
│                    ↓                                         │
│              合并检索结果 ──→ LLM生成答案                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
"""

import os
import json
import re
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ==================== 第一部分：推荐收集的资料清单 ====================

RECOMMENDED_RESOURCES = """
═══════════════════════════════════════════════════════════════════
                    RAG第二阶段 - 推荐资料清单
═══════════════════════════════════════════════════════════════════

【优先级说明】
🔴 必选 - 直接影响评测准确率
🟡 推荐 - 提升回答深度和广度  
🟢 可选 - 进阶内容

═══════════════════════════════════════════════════════════════════
一、CUDA官方文档 🔴必选
═══════════════════════════════════════════════════════════════════

1. CUDA C++ Programming Guide（最重要！）
   下载：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
   PDF：https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
   
   重点章节：
   ├── Chapter 2: Programming Model（Grid/Block/Thread）
   ├── Chapter 4: Hardware Implementation（SM/Warp）
   ├── Chapter 5: Memory Hierarchy（内存层次）
   ├── Chapter 6: Performance Guidelines（性能优化）
   └── Appendix: PTX ISA（PTX指令集）

2. CUDA C++ Best Practices Guide 🔴必选
   PDF：https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf
   
   重点章节：
   ├── Memory Optimizations
   ├── Execution Configuration Optimizations
   └── Instruction Optimization

3. PTX ISA Reference 🟡推荐
   PDF：https://docs.nvidia.com/cuda/pdf/ptx_isa_8.5.pdf
   用于：PTX指令级问题

═══════════════════════════════════════════════════════════════════
二、Triton官方资料 🔴必选（难题占16%）
═══════════════════════════════════════════════════════════════════

1. Triton官方教程（GitHub）🔴必选
   地址：https://github.com/triton-lang/triton/tree/main/python/tutorials
   
   必收教程：
   ├── 01-vector-add.py        # 基础
   ├── 02-fused-softmax.py     # 融合算子
   ├── 03-matrix-multiplication.py  # 矩阵乘法（重要！）
   ├── 04-low-memory-dropout.py
   ├── 05-layer-norm.py
   └── 06-fused-attention.py   # FlashAttention相关

2. Triton官方文档 🔴必选
   地址：https://triton-lang.org/main/index.html
   
   重点：
   ├── Programming Guide
   ├── Language Reference（tl.load, tl.store等）
   └── Autotuning

3. Triton论文（MLSYS 2019）🟡推荐
   标题："Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"
   用于：理解Triton vs CUDA的设计理念

═══════════════════════════════════════════════════════════════════
三、TileLang资料 🟡推荐
═══════════════════════════════════════════════════════════════════

1. TileLang官方仓库
   地址：https://github.com/tile-ai/tilelang（如果公开）
   
2. 相关论文/文档
   - 搜索"TileLang GPU"相关论文

═══════════════════════════════════════════════════════════════════
四、补充资料 🟢可选
═══════════════════════════════════════════════════════════════════

1. PMPP教材（如果有PDF）
   - "Programming Massively Parallel Processors"
   - 加分题的直接来源

2. FlashAttention论文
   - 理解高效Attention实现

3. NVIDIA技术博客精选
   - https://developer.nvidia.com/blog/
   - 搜索：CUDA optimization, Triton

4. cuBLAS/cuDNN文档
   - 了解库函数优化原理

═══════════════════════════════════════════════════════════════════
五、资料收集优先级排序
═══════════════════════════════════════════════════════════════════

第一批（立即收集）：
1. CUDA C++ Programming Guide PDF
2. CUDA Best Practices Guide PDF
3. Triton官方教程（01-06）

第二批（有时间再收集）：
4. PTX ISA Reference
5. Triton完整文档
6. TileLang资料

第三批（可选）：
7. PMPP教材
8. FlashAttention论文
9. 技术博客

═══════════════════════════════════════════════════════════════════
"""

# ==================== 第二部分：Index配置 ====================

@dataclass
class IndexConfig:
    """单个Index的配置"""
    name: str                          # Index名称
    description: str                   # 描述
    source_dir: str                    # 源文件目录
    index_dir: str                     # 索引保存目录
    chunk_size: int = 800              # chunk大小（字符）
    chunk_overlap: int = 100           # chunk重叠
    file_types: List[str] = field(default_factory=lambda: [".pdf", ".md", ".txt"])
    metadata_template: Dict = field(default_factory=dict)

# 多Index配置
MULTI_INDEX_CONFIG = {
    "rag_qna": IndexConfig(
        name="rag_qna",
        description="Q&A问答库（训练数据）",
        source_dir="./data/qna",
        index_dir="./rag_indexes/qna",
        chunk_size=500,
        metadata_template={"type": "qna", "level": "basic"}
    ),
    "rag_cuda": IndexConfig(
        name="rag_cuda",
        description="CUDA官方文档",
        source_dir="./data/cuda_docs",
        index_dir="./rag_indexes/cuda",
        chunk_size=800,
        metadata_template={"type": "official", "lang": "cuda"}
    ),
    "rag_triton": IndexConfig(
        name="rag_triton",
        description="Triton官方教程和文档",
        source_dir="./data/triton_docs",
        index_dir="./rag_indexes/triton",
        chunk_size=800,
        metadata_template={"type": "tutorial", "lang": "triton"}
    ),
    "rag_advanced": IndexConfig(
        name="rag_advanced",
        description="高级主题（PTX/TileLang等）",
        source_dir="./data/advanced_docs",
        index_dir="./rag_indexes/advanced",
        chunk_size=600,
        metadata_template={"type": "advanced", "lang": "mixed"}
    ),
}

# ==================== 第三部分：文档处理器 ====================

class DocumentProcessor:
    """文档处理器：支持PDF、Markdown、TXT"""
    
    def __init__(self, config: IndexConfig):
        self.config = config
    
    def load_pdf(self, pdf_path: str) -> List[Dict]:
        """加载PDF文件"""
        try:
            from pypdf import PdfReader
        except ImportError:
            print("需要安装pypdf: pip install pypdf")
            return []
        
        reader = PdfReader(pdf_path)
        chunks = []
        current_text = ""
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            
            # 清理文本
            text = self._clean_text(text)
            current_text += text + "\n"
            
            # 按chunk_size切分
            while len(current_text) >= self.config.chunk_size:
                # 找到合适的切分点（句子边界）
                split_point = self._find_split_point(current_text, self.config.chunk_size)
                chunk_text = current_text[:split_point].strip()
                
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "type": "pdf"
                    })
                
                current_text = current_text[split_point - self.config.chunk_overlap:]
        
        # 处理剩余文本
        if current_text.strip():
            chunks.append({
                "text": current_text.strip(),
                "source": os.path.basename(pdf_path),
                "page": len(reader.pages),
                "type": "pdf"
            })
        
        return chunks
    
    def load_markdown(self, md_path: str) -> List[Dict]:
        """加载Markdown文件，按章节切分"""
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = []
        
        # 按标题切分（# ## ###）
        sections = re.split(r'\n(?=#{1,3}\s)', content)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # 提取标题
            title_match = re.match(r'^(#{1,3})\s+(.+?)(?:\n|$)', section)
            title = title_match.group(2) if title_match else "Untitled"
            
            # 如果section太长，进一步切分
            if len(section) > self.config.chunk_size:
                sub_chunks = self._split_long_text(section)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        "text": sub_chunk,
                        "source": os.path.basename(md_path),
                        "section": title,
                        "part": i + 1,
                        "type": "markdown"
                    })
            else:
                chunks.append({
                    "text": section,
                    "source": os.path.basename(md_path),
                    "section": title,
                    "type": "markdown"
                })
        
        return chunks
    
    def load_txt(self, txt_path: str) -> List[Dict]:
        """加载TXT文件"""
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = []
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) < self.config.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": os.path.basename(txt_path),
                        "type": "txt"
                    })
                current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "source": os.path.basename(txt_path),
                "type": "txt"
            })
        
        return chunks
    
    def load_triton_tutorial(self, py_path: str) -> List[Dict]:
        """加载Triton教程（Python文件，包含代码和注释）"""
        with open(py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = []
        
        # 提取docstring和关键代码块
        # Triton教程通常有详细的注释
        
        # 1. 提取模块docstring
        docstring_match = re.match(r'^"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            chunks.append({
                "text": docstring_match.group(1).strip(),
                "source": os.path.basename(py_path),
                "section": "Overview",
                "type": "triton_tutorial"
            })
        
        # 2. 提取带@triton.jit装饰器的kernel函数
        kernel_pattern = r'(@triton\.jit\s*\ndef\s+\w+.*?)(?=\n@triton\.jit|\ndef\s+\w+(?!.*triton)|$)'
        kernels = re.findall(kernel_pattern, content, re.DOTALL)
        
        for kernel in kernels:
            # 提取函数名
            func_name_match = re.search(r'def\s+(\w+)', kernel)
            func_name = func_name_match.group(1) if func_name_match else "kernel"
            
            chunks.append({
                "text": kernel.strip(),
                "source": os.path.basename(py_path),
                "section": f"Kernel: {func_name}",
                "type": "triton_kernel",
                "has_code": True
            })
        
        # 3. 提取重要注释块
        comment_blocks = re.findall(r'# %%.*?\n(.*?)(?=\n# %%|$)', content, re.DOTALL)
        for block in comment_blocks:
            if len(block.strip()) > 100:  # 只保留有意义的块
                chunks.append({
                    "text": block.strip(),
                    "source": os.path.basename(py_path),
                    "type": "triton_explanation"
                })
        
        return chunks
    
    def load_json_qna(self, json_path: str) -> List[Dict]:
        """加载JSON格式的Q&A数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = []
        for item in data:
            # 支持多种JSON格式
            question = item.get("instruction", item.get("question", item.get("input", "")))
            answer = item.get("output", item.get("answer", item.get("response", "")))
            
            if question and answer:
                # 格式化为【问题】【答案】格式
                chunk_text = f"【问题】{question.strip()}\n【答案】{answer.strip()}"
                chunks.append({
                    "text": chunk_text,
                    "source": os.path.basename(json_path),
                    "type": "qna",
                    "question": question.strip(),
                    "answer": answer.strip()
                })
        
        return chunks
    
    def load_python_code(self, py_path: str) -> List[Dict]:
        """加载Python代码文件（包括Triton教程）"""
        with open(py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = []
        filename = os.path.basename(py_path)
        
        # 1. 提取模块级docstring
        docstring_match = re.match(r'^"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            doc_text = docstring_match.group(1).strip()
            if len(doc_text) > 50:  # 有实质内容
                chunks.append({
                    "text": f"【{filename}】概述：\n{doc_text}",
                    "source": filename,
                    "section": "Overview",
                    "type": "python_doc"
                })
        
        # 2. 提取带装饰器的函数（@triton.jit, @torch.jit等）
        decorated_func_pattern = r'(@\w+(?:\.\w+)*(?:\([^)]*\))?\s*\n)+def\s+(\w+)\s*\([^)]*\).*?(?=\n@|\ndef\s|\nclass\s|$)'
        decorated_funcs = re.findall(decorated_func_pattern, content, re.DOTALL)
        
        # 更简单的方式：直接匹配@triton.jit到下一个顶级定义
        triton_kernels = re.findall(
            r'(@triton\.(?:jit|autotune)[^\n]*\n(?:@[^\n]*\n)*def\s+\w+.*?)(?=\n@triton|\ndef\s+\w+\(|$)',
            content, re.DOTALL
        )
        
        for kernel in triton_kernels:
            func_name_match = re.search(r'def\s+(\w+)', kernel)
            func_name = func_name_match.group(1) if func_name_match else "kernel"
            
            chunks.append({
                "text": f"【Triton Kernel: {func_name}】\n{kernel.strip()}",
                "source": filename,
                "section": f"Kernel: {func_name}",
                "type": "triton_kernel",
                "has_code": True
            })
        
        # 3. 如果没有找到Triton kernel，尝试提取普通函数和注释
        if not triton_kernels:
            # 按段落切分（代码块+注释块）
            blocks = re.split(r'\n\n+', content)
            current_chunk = ""
            
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                
                if len(current_chunk) + len(block) < self.config.chunk_size:
                    current_chunk += "\n\n" + block if current_chunk else block
                else:
                    if current_chunk and len(current_chunk) > 100:
                        chunks.append({
                            "text": current_chunk,
                            "source": filename,
                            "type": "python_code"
                        })
                    current_chunk = block
            
            if current_chunk and len(current_chunk) > 100:
                chunks.append({
                    "text": current_chunk,
                    "source": filename,
                    "type": "python_code"
                })
        
        # 4. 提取重要的注释块（# %% 标记的cell）
        cells = re.split(r'\n# %%[^\n]*\n', content)
        for cell in cells:
            # 只保留有大量注释的cell
            comment_lines = [l for l in cell.split('\n') if l.strip().startswith('#')]
            if len(comment_lines) > 5:
                comment_text = '\n'.join(comment_lines)
                if len(comment_text) > 100:
                    chunks.append({
                        "text": f"【{filename}】注释说明：\n{comment_text}",
                        "source": filename,
                        "type": "python_comment"
                    })
        
        return chunks
    
    def process_directory(self, source_dir: str = None) -> List[Dict]:
        """处理整个目录"""
        source_dir = source_dir or self.config.source_dir
        all_chunks = []
        
        if not os.path.exists(source_dir):
            print(f"目录不存在: {source_dir}")
            return []
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                print(f"处理: {file_path}")
                
                chunks = []
                
                if ext == '.pdf':
                    chunks = self.load_pdf(file_path)
                elif ext == '.md':
                    chunks = self.load_markdown(file_path)
                elif ext == '.txt':
                    chunks = self.load_txt(file_path)
                elif ext == '.json':
                    chunks = self.load_json_qna(file_path)
                elif ext == '.py':
                    chunks = self.load_python_code(file_path)
                else:
                    print(f"  -> 跳过（不支持的格式）")
                    continue
                
                if not chunks:
                    print(f"  -> 0 chunks（无有效内容）")
                    continue
                
                # 添加metadata
                for chunk in chunks:
                    chunk.update(self.config.metadata_template)
                
                all_chunks.extend(chunks)
                print(f"  -> {len(chunks)} chunks")
        
        return all_chunks
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()
    
    def _find_split_point(self, text: str, target_length: int) -> int:
        """找到合适的切分点（句子边界）"""
        # 优先在句子边界切分
        for sep in ['。', '.\n', '. ', '\n\n', '\n', ' ']:
            pos = text.rfind(sep, 0, target_length + 50)
            if pos > target_length * 0.5:
                return pos + len(sep)
        return target_length
    
    def _split_long_text(self, text: str) -> List[str]:
        """切分长文本"""
        chunks = []
        while len(text) > self.config.chunk_size:
            split_point = self._find_split_point(text, self.config.chunk_size)
            chunks.append(text[:split_point].strip())
            text = text[split_point - self.config.chunk_overlap:]
        if text.strip():
            chunks.append(text.strip())
        return chunks

# ==================== 第四部分：Index构建器 ====================

class IndexBuilder:
    """Index构建器"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        self.embedding_model_name = embedding_model
        self.device = device
        self.embedding_model = None
    
    def load_embedding_model(self):
        """加载Embedding模型"""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"加载Embedding模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name, 
                device=self.device
            )
            print("✓ Embedding模型加载完成")
    
    def build_index(self, chunks: List[Dict], index_dir: str):
        """构建FAISS索引"""
        import faiss
        import numpy as np
        
        self.load_embedding_model()
        
        os.makedirs(index_dir, exist_ok=True)
        
        # 提取文本
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"生成向量... ({len(texts)} chunks)")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # 构建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        # 保存索引
        faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
        
        # 保存文档（包含metadata）
        with open(os.path.join(index_dir, "documents.pkl"), 'wb') as f:
            pickle.dump(chunks, f)
        
        # 保存配置
        config = {
            "embedding_model": self.embedding_model_name,
            "num_chunks": len(chunks),
            "dimension": dimension
        }
        with open(os.path.join(index_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ 索引保存到: {index_dir}")
        print(f"  - chunks: {len(chunks)}")
        print(f"  - dimension: {dimension}")

# ==================== 第五部分：多Index管理器 ====================

class MultiIndexManager:
    """多Index管理器"""
    
    def __init__(self, index_dirs: Dict[str, str], embedding_model: str = "./local-embedding-model"):
        """
        Args:
            index_dirs: {"index_name": "index_dir_path", ...}
            embedding_model: Embedding模型路径
        """
        self.index_dirs = index_dirs
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.indexes = {}
        self.documents = {}
    
    def load_all(self):
        """加载所有索引"""
        import faiss
        from sentence_transformers import SentenceTransformer
        
        # 加载Embedding模型
        print(f"加载Embedding模型: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
        
        # 加载各个索引
        for name, index_dir in self.index_dirs.items():
            if not os.path.exists(index_dir):
                print(f"⚠ 索引不存在，跳过: {name}")
                continue
            
            print(f"加载索引: {name}")
            self.indexes[name] = faiss.read_index(os.path.join(index_dir, "index.faiss"))
            with open(os.path.join(index_dir, "documents.pkl"), 'rb') as f:
                self.documents[name] = pickle.load(f)
            print(f"  ✓ {len(self.documents[name])} chunks")
    
    def search(self, query: str, index_names: List[str] = None, top_k: int = 5) -> List[Tuple[Dict, float, str]]:
        """
        搜索多个索引
        
        Returns:
            List of (chunk_dict, score, index_name)
        """
        if index_names is None:
            index_names = list(self.indexes.keys())
        
        # 编码query
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        
        all_results = []
        
        for name in index_names:
            if name not in self.indexes:
                continue
            
            index = self.indexes[name]
            docs = self.documents[name]
            
            scores, indices = index.search(query_emb.astype('float32'), top_k)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(docs) and score > 0.3:
                    all_results.append((docs[idx], float(score), name))
        
        # 按分数排序
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return all_results[:top_k]
    
    def route_and_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float, str]]:
        """
        智能路由：根据query内容选择合适的索引
        """
        query_lower = query.lower()
        
        # 路由规则
        selected_indexes = []
        
        # Triton相关
        if any(kw in query_lower for kw in ['triton', 'tl.', '@triton', 'autotune']):
            selected_indexes.append('rag_triton')
        
        # PTX/TileLang相关
        if any(kw in query_lower for kw in ['ptx', 'tilelang', 'asm', '汇编']):
            selected_indexes.append('rag_advanced')
        
        # CUDA基础问题
        if any(kw in query_lower for kw in ['cuda', 'kernel', 'thread', 'block', 'grid', 'warp', 
                                             'shared memory', 'global memory', '共享内存', '全局内存',
                                             'sm', 'occupancy', 'bank conflict']):
            selected_indexes.append('rag_cuda')
        
        # Q&A库始终作为兜底
        selected_indexes.append('rag_qna')
        
        # 去重
        selected_indexes = list(dict.fromkeys(selected_indexes))
        
        return self.search(query, selected_indexes, top_k)

# ==================== 第六部分：构建脚本 ====================

def build_single_index(
    source_dir: str,
    index_dir: str,
    index_name: str,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    chunk_size: int = 800
):
    """构建单个索引的便捷函数"""
    
    config = IndexConfig(
        name=index_name,
        description=f"Index for {index_name}",
        source_dir=source_dir,
        index_dir=index_dir,
        chunk_size=chunk_size
    )
    
    # 处理文档
    processor = DocumentProcessor(config)
    chunks = processor.process_directory()
    
    if not chunks:
        print(f"⚠ 没有找到可处理的文档: {source_dir}")
        return
    
    print(f"\n总计 {len(chunks)} chunks")
    
    # 构建索引
    builder = IndexBuilder(embedding_model)
    builder.build_index(chunks, index_dir)

def build_all_indexes(base_dir: str = "./data", embedding_model: str = "BAAI/bge-small-zh-v1.5"):
    """构建所有索引"""
    
    print("="*60)
    print("构建多Index RAG知识库")
    print("="*60)
    
    # 目录结构
    """
    data/
    ├── qna/                    # Q&A数据（已有）
    │   └── train_data.json
    ├── cuda_docs/              # CUDA文档
    │   ├── cuda_programming_guide.pdf
    │   └── cuda_best_practices.pdf
    ├── triton_docs/            # Triton文档
    │   ├── 01-vector-add.py
    │   ├── 03-matrix-multiplication.py
    │   └── triton_guide.md
    └── advanced_docs/          # 高级文档
        ├── ptx_isa.pdf
        └── tilelang.md
    """
    
    indexes_to_build = [
        ("qna", "rag_qna", 500),
        ("cuda_docs", "rag_cuda", 800),
        ("triton_docs", "rag_triton", 800),
        ("advanced_docs", "rag_advanced", 600),
    ]
    
    for subdir, index_name, chunk_size in indexes_to_build:
        source_dir = os.path.join(base_dir, subdir)
        index_dir = f"./rag_indexes/{index_name}"
        
        if os.path.exists(source_dir):
            print(f"\n{'='*60}")
            print(f"构建索引: {index_name}")
            print(f"源目录: {source_dir}")
            print(f"{'='*60}")
            
            build_single_index(source_dir, index_dir, index_name, embedding_model, chunk_size)
        else:
            print(f"\n⚠ 跳过 {index_name}，目录不存在: {source_dir}")

# ==================== 主函数 ====================

def main():
    """主函数：打印指南并提供构建选项"""
    
    print(RECOMMENDED_RESOURCES)
    
    print("\n" + "="*60)
    print("使用说明")
    print("="*60)
    
    print("""
1. 创建目录结构：
   mkdir -p data/qna data/cuda_docs data/triton_docs data/advanced_docs

2. 放置文档：
   - data/cuda_docs/: 放入CUDA PDF文档
   - data/triton_docs/: 放入Triton教程(.py)和文档(.md)
   - data/advanced_docs/: 放入PTX、TileLang等文档

3. 运行构建：
   python rag_stage2_guide.py --build

4. 使用多Index：
   from rag_stage2_guide import MultiIndexManager
   
   manager = MultiIndexManager({
       "rag_qna": "./rag_indexes/rag_qna",
       "rag_cuda": "./rag_indexes/rag_cuda",
       "rag_triton": "./rag_indexes/rag_triton",
   })
   manager.load_all()
   results = manager.route_and_search("什么是shared memory?")
""")

if __name__ == "__main__":
    import sys
    
    if "--build" in sys.argv:
        build_all_indexes()
    elif "--help" in sys.argv:
        print(RECOMMENDED_RESOURCES)
    else:
        main()