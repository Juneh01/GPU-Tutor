# 使用一个轻量的Python基础镜像

# for GPU
FROM vllm/vllm-openai@sha256:6766ce0c459e24b76f3e9ba14ffc0442131ef4248c904efdcbf0d89e38be01fe
# FROM vllm/vllm-openai:latest

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装依赖
# 这一步单独做可以利用Docker的层缓存机制，如果requirements.txt不变，则不会重新安装
COPY requirements.txt .
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py . 
RUN python3 download_model.py
#
# 复制项目中的所有其他文件
COPY . .

# 声明容器对外暴露的端口
EXPOSE 8000

# 重置 ENTRYPOINT，否则会被 vllm 命令拦截
ENTRYPOINT []

# 容器启动时运行的命令
# 使用uvicorn启动FastAPI服务，并监听所有网络接口的8000端口
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]