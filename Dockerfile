# === AI��������Dockerfile ===

#1.��������
FROM python:3.13-slim
#2.����Ŀ¼
WORKDIR /app
#3.�ȶ�����װCPU��torch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# ���� pip װ�������廪����torch ������ʽָ���ٷ� CPU Դ������Ӱ�죩
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

#4.���������嵥
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#5.������Ŀ����
COPY main.py .
COPY config.py .
COPY logger.py .
COPY agent/ ./agent/
COPY tools/ ./tools/
COPY rag/ ./rag/
COPY memory/ ./memory/
COPY build_vectorstore.py .
COPY llm_client.py .
#6.����ǰ�˹�������
COPY frontend/dist ./frontend/dist
#7.��������bge ranker���ڹ���Ŀ¼�������ھ���
ENV HF_HOME=/models
#8.�������������˿�
EXPOSE 8000
#9.��������
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
