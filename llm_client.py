"""共享 LLM 客户端配置 — agent 和 memory 模块共用同一个实例。

避免在多个文件中重复创建 ChatOpenAI 和重复调用 load_dotenv()。
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key=os.getenv("GLM_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)
