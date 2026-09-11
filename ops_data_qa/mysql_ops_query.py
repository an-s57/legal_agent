"""运营数据问答 —— text-to-SQL 查询网关（只读 + 白名单 + 错误重试 + 人话答案）。

流程：词面意图拦截 → LLM 意图门 → 生成 SQL → sqlglot 语法树校验 → 只读账号执行 → 结果人话化

安全纵深（每层各管一段、互为兜底；哪层拦的记在 result["blocked_by"]）：
  1. 词面意图拦截（validator.check_intent，fail-closed）：明显的删除/清空词直接拒，零成本
  2. LLM 意图门（intent_gate.check_intent_llm，fail-open）：抓词面抓不住的拐弯说法
     （"把差评处理掉"），GLM-4.7 判定；本层故障时退回词面结论
  3. sqlglot 语法树校验（validator.validate_sql）：单条 SELECT、表白名单、
     库名前缀校验、危险函数黑名单 + 自动补 LIMIT
  4. 数据库层：query_user 只读账号（只能 SELECT，实测 1142 拒绝 DELETE）——
     就算上层全失守，账号也没权限写

用法：
    python ops_data_qa/mysql_ops_query.py "上个月哪种案件类型咨询最多？"
    python ops_data_qa/mysql_ops_query.py --explain "8月有多少个会话？"   # 只打印 SQL 不执行
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    OPS_DB_HOST,
    OPS_DB_NAME,
    OPS_DB_PASSWORD,
    OPS_DB_PORT,
    OPS_DB_USER,
    OPS_QA_MAX_RETRY,
    OPS_QA_MAX_ROWS,
)
from llm_client import llm  # 复用项目已有的 LLM 封装
from ops_data_qa.intent_gate import check_intent_llm
from ops_data_qa.validator import check_intent, enforce_limit, validate_sql

# ── 配置：连接参数全部走 config（.env 注入），密码绝不写进代码 ──
DB_CONFIG = {
    "host": OPS_DB_HOST,
    "port": OPS_DB_PORT,
    "user": OPS_DB_USER,       # 只读账号
    "password": OPS_DB_PASSWORD,
    "database": OPS_DB_NAME,
    "charset": "utf8mb4",
}

ALLOWED_TABLES = {"sessions", "messages", "answer_ratings"}

# 表结构说明（喂给 LLM，表/字段少 → 减少歧义）
SCHEMA_DESC = """
表 sessions（咨询会话）:
  id            会话ID
  session_key   会话编号
  case_type     案件类型（劳动纠纷/消费维权/合同纠纷/交通事故/房屋租赁/婚姻家庭/刑事咨询），NULL 表示纯知识问答
  created_at    会话开始时间 (DATETIME)
  updated_at    会话结束时间
表 messages（会话消息）:
  id, session_id(关联 sessions.id), sender('user'=用户/'agent'=助手), content, created_at
表 answer_ratings（回答评价）:
  id, session_id(关联 sessions.id), rating(1=点赞, -1=点踩), comment, created_at
"""

SQL_PROMPT = """你是 MySQL 查询专家。请根据表结构，把用户的自然语言问题转换成一条 SQL 查询。

规则：
1. 只输出一条 SELECT 语句，不要输出任何解释、不要用 markdown 代码块
2. 只允许查询这些表：sessions, messages, answer_ratings
3. 用到的列必须真实存在于表结构中
4. 相对时间请换算成具体日期（当前时间：{now}）
5. 需要统计时用 COUNT/SUM/AVG/GROUP BY，结果加有意义的别名
6. **严格按字面回答问题，不要自行添加过滤条件**——除非用户问题里明确说了筛选条件。
   例如"一共有多少个会话"就是 COUNT(*) 全部，不要加 WHERE case_type IS NOT NULL。
7. 不要臆测用户的隐含意图，不要"帮忙优化"问题

表结构：
{schema}

用户问题：{question}

SQL："""


# ── 第 1 步：自然语言 → SQL ────────────────────────────
def nl_to_sql(question: str, error_hint: str = "") -> str:
    """调 LLM 把问题转成 SQL；error_hint 非空时把上次报错喂回去让它改。"""
    # 运行时注入真实当前时间（原来写死"当前时间视为 2026-09-15"，日期一过评测全错）
    prompt = SQL_PROMPT.format(
        schema=SCHEMA_DESC,
        question=question,
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    if error_hint:
        prompt += f"\n\n上一次生成的 SQL 执行报错了，请修正。报错信息：{error_hint}"

    resp = llm.invoke(prompt)
    sql = (resp.content or "").strip()
    # 清理：去掉可能的 ```sql 代码块标记
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    # 只取第一条语句（防 LLM 输出多条）
    sql = sql.split(";")[0].strip() + ";"
    return sql


# ── 第 2 步：安全校验（意图拦截 validate 语法树校验、LIMIT 兜底）──────
# 实现全部在 ops_data_qa/validator.py，可离线单测（见 tests/test_ops_sql_validator.py）

# ── 第 3 步：用只读账号执行 ────────────────────────────
def execute_sql(sql: str) -> tuple[list, list]:
    """执行 SQL，返回 (列名, 行)。异常向上抛给重试逻辑。"""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
        return cols, rows
    finally:
        conn.close()


# ── 第 4 步：结果人话化 ───────────────────────────────
def rows_to_text(question: str, sql: str, cols: list, rows: list) -> str:
    """把查询结果翻译成自然语言（再调一次 LLM；失败则退化为原始表格）。"""
    if not rows:
        return "查询结果为空。"
    table = " | ".join(str(c) for c in cols) + "\n"
    for r in rows[:50]:
        table += " | ".join(str(v) for v in r) + "\n"
    prompt = (
        "你在为运营人员解读数据库查询结果。用一句简洁的中文回答用户问题，"
        "直接给结论和关键数字，不要复述 SQL。\n"
        f"用户问题：{question}\nSQL：{sql}\n查询结果：\n{table}"
    )
    try:
        return (llm.invoke(prompt).content or "").strip()
    except Exception:
        return table.strip()


# ── 主流程 ────────────────────────────────────────────
def ask(question: str, explain_only: bool = False, answer_text: bool = True) -> dict:
    """完整链路：意图拦截 → 问题转 SQL → 校验 → 执行（失败重试）→ 人话答案。

    answer_text=False 时跳过"人话翻译"这一次 LLM 调用（评测批量跑时省一半成本）。
    """
    result = {"question": question, "sql": None, "allowed": False,
              "rows": [], "answer": None, "error": None, "retries": 0,
              "blocked_by": None}   # word=词面拦截 / llm=意图门拦截

    # ── 第一道门：词面意图拦截（免费、瞬时；fail-closed，不生成 SQL）──
    destructive, reason = check_intent(question)
    if destructive:
        result["error"] = f"拒绝执行：{reason}。本工具是只读查询网关，不支持删除/修改数据。"
        result["blocked_by"] = "word"
        return result

    # ── 第二道门：LLM 意图门（抓词面抓不住的拐弯说法；fail-open）──
    llm_intent = check_intent_llm(question)
    if llm_intent["destructive"] is True:
        why = llm_intent["reason"] or "意图门判定为数据变更请求"
        result["error"] = f"拒绝执行：{why}。本工具是只读查询网关，不支持删除/修改数据。"
        result["blocked_by"] = "llm"
        return result

    error_hint = ""
    for attempt in range(OPS_QA_MAX_RETRY + 1):
        try:
            sql = nl_to_sql(question, error_hint)
        except Exception as e:
            result["error"] = f"生成 SQL 失败: {e}"
            return result

        result["sql"] = sql
        # ── 第二道门：sqlglot 语法树安全校验（语句类型/越权表/危险函数）──
        ok, reason2 = validate_sql(sql, ALLOWED_TABLES, OPS_DB_NAME)
        if not ok:
            result["error"] = f"安全校验未通过: {reason2}"
            return result
        result["allowed"] = True

        if explain_only:
            return result

        safe_sql = enforce_limit(sql, OPS_QA_MAX_ROWS)
        try:
            cols, rows = execute_sql(safe_sql)
            result["rows"] = rows
            if answer_text:
                result["answer"] = rows_to_text(question, safe_sql, cols, rows)
            return result
        except Exception as e:
            result["retries"] = attempt + 1
            error_hint = str(e)
            if attempt >= OPS_QA_MAX_RETRY:
                result["error"] = f"执行失败（已重试 {OPS_QA_MAX_RETRY} 次）: {error_hint}"

    return result


def main():
    parser = argparse.ArgumentParser(description="运营数据问答（text-to-SQL）")
    parser.add_argument("question", help="自然语言问题")
    parser.add_argument("--explain", action="store_true", help="只生成 SQL 不执行")
    args = parser.parse_args()

    r = ask(args.question, explain_only=args.explain)
    print(f"[问题] {r['question']}")
    print(f"[SQL] {r['sql']}")
    print(f"[校验] {'通过' if r['allowed'] else '拦截'}")
    if r["error"]:
        print(f"[错误] {r['error']}")
    if r["rows"]:
        print(f"[行数] {len(r['rows'])}")
    if r["answer"]:
        print(f"[回答] {r['answer']}")


if __name__ == "__main__":
    main()
