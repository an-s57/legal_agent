"""SQL 安全校验 —— 语法树级白名单（可离线单测的纯函数，零外部服务依赖）。

为什么用 sqlglot（语法树）而不是正则/关键词黑名单：
- 正则按"字符串"找表名：逗号多表（FROM a, b）、反引号、子查询里的表都可能漏；
- 关键词黑名单既会漏报也会误杀（列名叫 replace 的正常查询都被拦）；
- sqlglot 把 SQL 解析成语法树，程序在"结构"层面数表、验语句类型，
  花招藏不住、正常语句不误杀。

三层防线分工（纵深防御，每层各管一段、互为兜底）：
  ① check_intent   意图门卫：破坏性问题根本不生成 SQL（fail-closed，零 LLM 成本）
  ② validate_sql   安检仪：不管 LLM 吐出什么，只有结构安全的查询能过本层
  ③ 只读账号       数据库层的物理底线：就算上层全失守，账号也没权限写
"""
import re

import sqlglot
from sqlglot import exp

# 危险函数黑名单：读服务器文件 / 故意延时拖库 / 全局锁
_DENY_FUNCTIONS = {"LOAD_FILE", "SLEEP", "BENCHMARK", "GET_LOCK", "RELEASE_LOCK"}

# ── ① 意图门卫：破坏性意图拦截（在生成 SQL 之前，词面判断即可，fail-closed）──
# 背景：只做 SQL 层校验不够——用户说"删除所有记录"时，LLM 会"善意地"生成一条
# SELECT 来绕过（数据没丢，但破坏性意图没被拒绝）。所以先拦意图。
STRONG_DESTRUCTIVE = re.compile(
    r"(删除|删掉|删了|清空|清除|移除|干掉|drop|delete|truncate)", re.IGNORECASE)
# 弱写操作词：若问题里同时出现"分析类词"（多少/几个/哪些/统计/排名…）才放行，
# 例如"最近更新的会话有哪些"应放行，"把所有评价更新成点赞"应拦截。
SOFT_WRITE = re.compile(
    r"(修改|更改|改成|改掉|更新|覆盖|写入|插入|insert|update|alter|modify|replace)",
    re.IGNORECASE)
ANALYTIC_WORDS = re.compile(
    r"(多少|几个|几条|几种|哪些|哪个|哪类|最多|最少|平均|统计|分布|比例|排名|趋势|列出|查询|看看)")


def check_intent(question: str) -> tuple[bool, str]:
    """判断问题是否为破坏性意图（写操作）。返回 (是否破坏性, 原因)。"""
    if STRONG_DESTRUCTIVE.search(question):
        return True, "检测到删除/清空类操作"
    if SOFT_WRITE.search(question) and not ANALYTIC_WORDS.search(question):
        return True, "检测到修改/更新类操作，且不是统计分析问题"
    return False, "ok"


# ── ② 安检仪：sqlglot 语法树校验 ──

def validate_sql(sql: str, allowed_tables: set[str], allowed_db: str = "") -> tuple[bool, str]:
    """校验 SQL 是否为"安全的单条查询"。返回 (是否通过, 原因)。

    规则：
    - 必须能解析、且只有一条语句（多语句注入在语句计数处被拦）；
    - 顶层必须是 SELECT/UNION —— DELETE/UPDATE/DROP/TRUNCATE/CREATE 等一票否决；
    - 语法树里出现的每一张表（含子查询、逗号连接、UNION 分支）都必须在白名单内，
      带库名前缀时库名必须匹配；
    - CTE（WITH t AS ...）的别名不是真表，放行，但 CTE 定义体内的真表仍要过白名单；
    - 危险函数（读文件/延时/全局锁）一律拦截。
    """
    try:
        statements = [s for s in sqlglot.parse(sql, read="mysql") if s is not None]
    except sqlglot.errors.ParseError as e:
        return False, f"SQL 解析失败（可能不是合法 SQL）: {str(e)[:100]}"

    if not statements:
        return False, "空语句"
    if len(statements) > 1:
        return False, f"禁止多条语句（检测到 {len(statements)} 条）"
    tree = statements[0]

    # 顶层语句类型一票否决：不是查询就不放行
    if not isinstance(tree, (exp.Select, exp.Union)):
        return False, f"只允许 SELECT 查询（实际是 {type(tree).__name__}）"

    # 危险函数黑名单
    for fn in tree.find_all(exp.Func):
        fname = (getattr(fn, "name", "") or "").upper()
        if fname in _DENY_FUNCTIONS:
            return False, f"检测到危险函数 {fname}"

    # 表白名单：语法树里的每一张表都要过关
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    seen_tables = 0
    for node in tree.find_all(exp.Table):
        seen_tables += 1
        db = (node.db or "").lower()
        table = node.name.lower()
        if db and allowed_db and db != allowed_db.lower():
            return False, f"访问了白名单外的库: {db}.{node.name}"
        if table not in allowed_tables and table not in cte_names:
            return False, f"访问了白名单外的表: {node.name}"
    if seen_tables == 0:
        return False, "未识别到查询的表"

    return True, "ok"


# ── ③ LIMIT 兜底：没写 LIMIT 的查询自动补行数上限，防止一次拉全表 ──

def enforce_limit(sql: str, max_rows: int = 200) -> str:
    """没写 LIMIT 时补一个大上限（防拖垮数据库，不是限制正确答案）。"""
    if re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        return sql
    return sql.rstrip(";").rstrip() + f" LIMIT {max_rows};"
