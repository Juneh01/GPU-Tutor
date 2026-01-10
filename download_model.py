# download_models.py
# 使用 ModelScope SDK 下载模型（国内镜像，速度快）

from modelscope.hub.api import HubApi
from modelscope import snapshot_download

# 登录 ModelScope
api = HubApi()
api.login('ms-cbf282e9-504f-4d81-bb76-fea53d1318a3')

# ==================== 1. 下载LLM模型 ====================
llm_model_id = 'junharthur/GPU-Course-Qwen3-4B-V1-MS'
llm_local_path = './local-model'

print(f"[1/2] 开始下载LLM模型 {llm_model_id}...")
llm_model_dir = snapshot_download(llm_model_id, local_dir=llm_local_path)
print(f"✅ LLM模型下载完成！路径: {llm_model_dir}")

# ==================== 2. 下载Embedding模型 ====================
# BGE-small-zh-v1.5 在 ModelScope 上的ID
embedding_model_id = 'BAAI/bge-small-zh-v1.5'
embedding_local_path = './local-embedding-model'

print(f"\n[2/2] 开始下载Embedding模型 {embedding_model_id}...")
embedding_model_dir = snapshot_download(embedding_model_id, local_dir=embedding_local_path)
print(f"✅ Embedding模型下载完成！路径: {embedding_model_dir}")

# ==================== 完成 ====================
print("\n" + "="*60)
print("所有模型下载完成！")
print("="*60)
print(f"LLM模型路径:       {llm_local_path}")
print(f"Embedding模型路径: {embedding_local_path}")
print("\n使用方式:")
print('  # LLM')
print(f'  MODEL_PATH = "{llm_local_path}"')
print('  # Embedding')
print(f'  EMBEDDING_MODEL = "{embedding_local_path}"')