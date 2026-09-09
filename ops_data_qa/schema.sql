-- ops_data_qa 运营分析 demo 库：建表脚本
-- 用法：docker exec -i legal-mysql mysql -uroot -pdevpw123 < ops_data_qa/schema.sql
-- 若重复执行想重置：先手动执行 DROP DATABASE ops_demo;

CREATE DATABASE IF NOT EXISTS ops_demo CHARACTER SET utf8mb4;
USE ops_demo;

-- 咨询会话：一次完整对话 = 一条 session
CREATE TABLE IF NOT EXISTS sessions (
  id INT PRIMARY KEY AUTO_INCREMENT,
  session_key VARCHAR(64) NOT NULL,   -- 对应 LexAgent 的 session_id
  case_type VARCHAR(30) DEFAULT NULL, -- 案件类型：劳动纠纷/消费维权/合同纠纷/交通事故；NULL=纯知识问答
  created_at DATETIME NOT NULL,
  updated_at DATETIME
);

-- 消息：一次会话里有多条 user/agent 消息
CREATE TABLE IF NOT EXISTS messages (
  id INT PRIMARY KEY AUTO_INCREMENT,
  session_id INT NOT NULL,
  sender VARCHAR(10) NOT NULL,        -- user / agent
  content TEXT,
  created_at DATETIME NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- 回答评价：用户对 Agent 某次回答点 赞/踩
CREATE TABLE IF NOT EXISTS answer_ratings (
  id INT PRIMARY KEY AUTO_INCREMENT,
  session_id INT NOT NULL,
  rating TINYINT NOT NULL,            -- 1 = 赞，-1 = 踩
  comment VARCHAR(200),
  created_at DATETIME NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
