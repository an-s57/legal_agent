"""运营问答审计日志 —— JSONL 追加写。

为什么是文件不是库：查库的 query_user 是只读账号（安全设计反过来约束了架构，
审计没有权限写进 ops_demo），写本地文件最简单，且 data/ 目录不进 git。
每行一条 JSON：时间 / 令牌指纹 / 问题 / SQL / 放行与否 / 哪层拦的 / 行数 / 耗时。

fail-open：审计写失败绝不影响问答主流程，只记日志。
"""
import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path

from config import DATA_DIR
from logger import get_logger

logger = get_logger("legal_agent.ops_qa")

AUDIT_PATH = Path(DATA_DIR) / "ops_audit.jsonl"
_lock = threading.Lock()


def fingerprint(token: str) -> str:
    """令牌指纹：审计不落原始令牌，存 sha256 前 8 位，足以区分"哪个使用者"。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8] if token else "-"


def record(token: str, **fields) -> None:
    """追加一条审计记录；写失败只打日志（fail-open），绝不影响问答。"""
    row = {"ts": datetime.now().isoformat(timespec="seconds"), "token": fingerprint(token), **fields}
    try:
        with _lock:
            AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with AUDIT_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[AUDIT] 审计写入失败（{type(e).__name__}: {e}），不影响主流程")


def recent(n: int = 20) -> list[dict]:
    """读最近 n 条（从文件尾倒序）；文件不存在或个别行损坏时尽量多给。"""
    try:
        if not AUDIT_PATH.exists():
            return []
        entries: list[dict] = []
        for line in reversed(AUDIT_PATH.read_text(encoding="utf-8").splitlines()):
            if len(entries) >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 半截行（进程写一半被杀）跳过
        return entries
    except Exception as e:
        logger.warning(f"[AUDIT] 审计读取失败（{type(e).__name__}: {e}）")
        return []
