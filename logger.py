"""统一日志配置 — 全项目共用，print → logging。

设计要点：
- 输出到 **stderr**：终端照常显示，但 stdout 被腾空——
  MCP server 用 stdout 传输协议消息（JSON-RPC），print 日志混进 stdout
  会污染协议流导致客户端解析失败；stderr 客户端不看，日志照打。
- 日志级别由环境变量 LOG_LEVEL 控制（DEBUG/INFO/WARNING/ERROR），默认 INFO。
- 每个模块通过 get_logger("legal_agent.<模块名>") 获取独立 logger。
- setup_logging() 幂等：无论被调用几次只配置一次，且不重复加 handler。
"""
import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: str | None = None) -> None:
    """配置根日志：stderr handler + 统一格式。可重复调用（幂等）。"""
    global _configured
    if _configured:
        return
    _configured = True

    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # 避免重复添加 stderr handler（uvicorn 等第三方库可能已配置过）
    if not any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
        for h in root.handlers
    ):
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """获取项目 logger；首次调用时自动完成默认配置。"""
    setup_logging()
    return logging.getLogger(name)
