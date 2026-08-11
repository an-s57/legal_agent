# === AI法律助手Dockerfile ===

#1.基础镜像
FROM python:3.13-slim
#2.工作目录
WORKDIR /app
#3.先独立安装CPU版torch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 后续 pip 装依赖走清华镜像（torch 那行显式指定官方 CPU 源，不受影响）
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

#4.拷贝依赖清单
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#5.拷贝项目代码
COPY main.py .
COPY agent/ ./agent/
COPY tools/ ./tools/
COPY rag/ ./rag/
COPY memory/ ./memory/
COPY build_vectorstore.py .
COPY llm_client.py .
#6.拷贝前端构建产物
COPY frontend/dist ./frontend/dist
#7.环境变量bge ranker放在挂载目录，不放在镜像
ENV HF_HOME=/models
#8.声明容器监听端口
EXPOSE 8000
#9.启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
