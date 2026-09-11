# MySQL 索引实验手册

用 100 万行真实数据，亲眼看见"索引为什么快"。每个实验都写了**应该看到什么**。

前置：先跑 `python mysql_lab/make_big_table.py` 造好数据（约 1~3 分钟）。

查数据用只读账号（query_user），建索引用 root（建索引是 DDL，只读账号干不了）。

进入 MySQL 命令行：

```bash
docker exec -it legal-mysql mysql -uquery_user -pquery123 ops_demo
```

---

## 实验 1：没有索引时，查询有多慢

```sql
SELECT COUNT(*) FROM sessions_big WHERE case_type = '刑事咨询';
```

注意命令回车后最后一行：`1 row in set (0.XX sec)`——记下这个秒数（预计 0.2~0.5s）。

再看数据库打算怎么执行这条查询：

```sql
EXPLAIN SELECT * FROM sessions_big WHERE case_type = '刑事咨询';
```

看结果表里的两列：

| 列 | 你应该看到 | 意思 |
|---|---|---|
| type | **ALL** | 全表扫描：一百万行挨个看 |
| rows | 约 1000000 | 预计要检查的行数 |
| key | NULL | 没用上任何索引 |

**记住这组数字，下面马上对比。**

## 实验 2：建索引，再看一遍

换 root 进去（root 密码见你 .env 或 docker 启动时的设置）：

```bash
docker exec -it legal-mysql mysql -uroot -p你的root密码 ops_demo
```

```sql
CREATE INDEX idx_case_type ON sessions_big(case_type);
```

建完回到 query_user（或者就用 root 也行），把实验 1 的两条再跑一遍：

```sql
SELECT COUNT(*) FROM sessions_big WHERE case_type = '刑事咨询';
EXPLAIN SELECT * FROM sessions_big WHERE case_type = '刑事咨询';
```

这次应该看到：

- 耗时：掉到 0.0X 秒（快几倍到几十倍）；
- type 变成 **ref**（走索引定位）；
- rows 变成一万多（刑事咨询只占总数的约 1.5%）；
- key 显示 idx_case_type。

**前后数字各截一张图，这就是周记素材。**

## 实验 3：唯一值查询，更极端的对比

```sql
EXPLAIN SELECT * FROM sessions_big WHERE session_key = 's_big_00050123';
```

没有索引时 type=ALL；给 session_key 建索引后再看：

```sql
-- 用 root
CREATE UNIQUE INDEX idx_session_key ON sessions_big(session_key);
```

type 会变成 **const**（主键或唯一索引的等值查找，最高级）——一百万行里直接跳到那一行。

## 实验 4：联合索引 + 最左前缀

用 root 建一个"两列合在一起"的索引：

```sql
CREATE INDEX idx_type_time ON sessions_big(case_type, created_at);
```

然后对比三条查询：

```sql
-- ① 两列都用 → 走索引
EXPLAIN SELECT * FROM sessions_big
WHERE case_type = '劳动纠纷' AND created_at >= '2025-06-01';

-- ② 只用第二列 → 走不了（最左前缀：联合索引必须从最左列开始用）
EXPLAIN SELECT * FROM sessions_big WHERE created_at >= '2025-06-01';

-- ③ 只用第一列 → 能走（但只用到第一列的长度）
EXPLAIN SELECT * FROM sessions_big WHERE case_type = '劳动纠纷';
```

看 ② 的 type 是不是还是 ALL 或 range 但 key 是别的——**这就是"最左前缀规则"的现场**。

## 实验 5：三种让索引失效的写法

```sql
-- ① 对列用函数 → 索引失效
EXPLAIN SELECT * FROM sessions_big WHERE DATE(created_at) = '2025-06-01';
-- 对比：改写成范围查询就能走索引
EXPLAIN SELECT * FROM sessions_big
WHERE created_at >= '2025-06-01' AND created_at < '2025-06-02';

-- ② LIKE 以 % 开头 → 索引失效
EXPLAIN SELECT * FROM sessions_big WHERE session_key LIKE '%0501%';
-- 对比：前缀匹配能走
EXPLAIN SELECT * FROM sessions_big WHERE session_key LIKE 's_big_0005%';

-- ③ 隐式类型转换：session_key 是字符串，却拿数字去比 → 索引失效
EXPLAIN SELECT * FROM sessions_big WHERE session_key = 50123;
```

每条失效的，type 都会退回 ALL 或 key 变 NULL。这三条是面试最常问的"索引失效场景"。

## 附：EXPLAIN 小抄

- **type 从好到差**：const > eq_ref > ref > range > index > ALL。看到 ALL 就要想"该建索引了吗、写得对吗"；
- **key**：实际用上的索引名，NULL 就是没用上；
- **rows**：预计要检查多少行，越少越好；
- **Extra**：`Using index` 是好事（覆盖索引，连回表都省了）。

## 收尾

实验做完想清理：

```sql
-- root
DROP TABLE sessions_big;
```

留着也没关系：它不在白名单里，Agent 和评测都碰不到它。
