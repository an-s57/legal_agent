import { useCallback, useState } from 'react'
import { motion } from 'framer-motion'

interface OpsResult {
  question: string
  sql: string | null
  allowed: boolean
  columns?: string[]
  rows: (string | number | null)[][]
  answer: string | null
  error: string | null
  retries?: number
  blocked_by?: string | null
  elapsed_ms?: number
  trace?: string
}

interface HistoryItem {
  question: string
  kind: 'ok' | 'blocked' | 'failed'
  summary: string
}

const TOKEN_KEY = 'ops_qa_token'
const MAX_TABLE_ROWS = 20

/**
 * 运营数据问答页 —— 与聊天页共用同一套视觉语言：
 * glass 卡片 / #374151 主按钮 / markdown-body 表格与代码块 / typing-dot 加载动画。
 */
export default function OpsQa() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY) || '')
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<OpsResult | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])

  const handleTokenChange = (v: string) => {
    setToken(v)
    localStorage.setItem(TOKEN_KEY, v)
  }

  const ask = useCallback(async () => {
    const q = question.trim()
    if (!q || loading) return
    const t = token.trim()
    if (!t) {
      setNotice('请先填入访问令牌（后端 .env 里 OPS_QA_TOKEN 的值）')
      return
    }
    setLoading(true)
    setNotice(null)
    try {
      const res = await fetch('/ops/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-OPS-TOKEN': t },
        body: JSON.stringify({ message: q }),
      })
      if (res.status === 429) {
        const body = (await res.json().catch(() => null)) as { detail?: string } | null
        setNotice(`⏳ ${body?.detail || '问得太快了，稍后再试'}`)
        return
      }
      if (res.status === 401) {
        setNotice('🔑 令牌不对：请检查访问令牌是否与后端 OPS_QA_TOKEN 一致')
        return
      }
      if (res.status === 503) {
        setNotice('⚙️ 运营问答未启用：后端未配置 OPS_QA_TOKEN')
        return
      }
      if (!res.ok) {
        setNotice(`请求失败（HTTP ${res.status}）`)
        return
      }
      const data = await res.json() as OpsResult
      setResult(data)
      const kind: HistoryItem['kind'] = data.error
        ? (data.allowed === false ? 'blocked' : 'failed')
        : 'ok'
      const summary = data.error
        ? data.error
        : (data.answer || `已生成 SQL（${(data.rows || []).length} 行）`).slice(0, 80)
      setHistory(prev => [{ question: q, kind, summary }, ...prev].slice(0, 20))
    } catch {
      setNotice('网络请求失败，请确认服务器已启动')
    } finally {
      setLoading(false)
    }
  }, [question, token, loading])

  const isBlocked = !!result && !!result.error && result.allowed === false
  const isFailed = !!result && !!result.error && result.allowed !== false
  const rows = result?.rows || []
  const columns = result?.columns || []

  return (
    <div className="flex-1 overflow-y-auto" style={{ padding: '16px 72px' }}>
      <motion.div
        className="flex flex-col gap-3"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
      >
        {/* 访问令牌 */}
        <section className="glass rounded-2xl px-3.5 py-3">
          <p className="text-[11px] font-medium text-gray-400 mb-1.5">访问令牌 · X-OPS-TOKEN</p>
          <div className="input-glow rounded-2xl flex items-center px-3 py-1.5 transition-all duration-300 ease-in-out">
            <input
              type="password"
              value={token}
              onChange={e => handleTokenChange(e.target.value)}
              placeholder="后端 .env 里 OPS_QA_TOKEN 的值"
              className="flex-1 bg-transparent outline-none text-[13px] text-ink placeholder:text-gray-300"
            />
          </div>
          <p className="text-[10px] text-gray-300 mt-1.5">
            只存在浏览器本地，用于调用后台只读查询接口
          </p>
        </section>

        {/* 提问 —— 与聊天输入框同款容器 */}
        <section className="flex-shrink-0">
          <div className="input-glow glass rounded-3xl flex items-center gap-2 px-3.5 py-2 transition-all duration-300 ease-in-out">
            <input
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') void ask() }}
              placeholder="例如：8月有多少个会话？哪种案件类型最多？"
              className="flex-1 bg-transparent outline-none text-[15px] text-ink placeholder:text-gray-300"
              disabled={loading}
            />
            <button
              onClick={() => void ask()}
              disabled={loading || !question.trim()}
              className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center
                         text-white transition-all duration-300 ease-in-out
                         hover:scale-105 hover:shadow-lg
                         disabled:bg-gray-200 disabled:scale-100 disabled:shadow-none"
              style={{ background: loading || !question.trim() ? undefined : '#374151' }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                <path d="M12 4L12 20M12 4L6 10M12 4L18 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
          <p className="text-center text-[10px] text-gray-300 mt-1.5">
            只读查询 · 四道门安全链路 · 同一令牌每 60 秒最多 10 次
          </p>
        </section>

        {notice && (
          <div className="glass rounded-2xl px-3.5 py-2.5 flex items-start gap-2">
            <span className="text-[13px]">{notice.startsWith('⏳') ? '⏳' : '⚠️'}</span>
            <span className="text-[13px] text-gray-600 leading-relaxed">{notice.replace(/^[^A-Za-z\u4e00-\u9fff]*/, '')}</span>
          </div>
        )}

        {/* 查询中 —— 复用聊天页 typing-dot 动画 */}
        {loading && (
          <div className="glass rounded-2xl rounded-bl-md px-3.5 py-2.5 flex items-center gap-1 w-fit">
            <span className="typing-dot" />
            <span className="typing-dot" />
            <span className="typing-dot" />
          </div>
        )}

        {/* 结果卡片 */}
        {result && !loading && (
          <motion.section
            key={result.trace}
            className="glass rounded-2xl px-3.5 py-3"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
          >
            {(isBlocked || isFailed) && (
              <div className="flex items-center gap-1.5 mb-1.5">
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-50 text-blue-500">
                  {isBlocked ? '已拦截 · 未执行查询' : '执行失败'}
                </span>
                {result.blocked_by && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-50 text-blue-500">
                    拦截层：{result.blocked_by}
                  </span>
                )}
              </div>
            )}

            <p className="text-[15px] leading-relaxed text-ink whitespace-pre-wrap">
              {isBlocked || isFailed ? result.error : (result.answer || '（无回答）')}
            </p>

            {result.sql && (
              <div className="markdown-body mt-2">
                <pre><code>{result.sql}</code></pre>
              </div>
            )}

            {!isBlocked && !isFailed && rows.length > 0 && (
              <div className="markdown-body mt-2 overflow-x-auto">
                <table>
                  <thead>
                    <tr>{columns.map((c, i) => <th key={i}>{c}</th>)}</tr>
                  </thead>
                  <tbody>
                    {rows.slice(0, MAX_TABLE_ROWS).map((row, i) => (
                      <tr key={i}>
                        {row.map((cell, j) => <td key={j}>{cell === null ? '' : String(cell)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {rows.length > MAX_TABLE_ROWS && (
                  <p className="text-[11px] text-gray-400 mt-1">仅展示前 {MAX_TABLE_ROWS} 行，共 {rows.length} 行</p>
                )}
              </div>
            )}

            {!isBlocked && !isFailed && rows.length === 0 && (
              <p className="text-[13px] text-gray-400 mt-2">查询结果为空。</p>
            )}

            <div className="mt-2 pt-2 border-t border-gray-100 flex gap-1.5 flex-wrap">
              {result.elapsed_ms != null && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-50 text-blue-500">
                  耗时 {result.elapsed_ms}ms
                </span>
              )}
              {!!result.retries && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-50 text-blue-500">
                  重试 {result.retries} 次
                </span>
              )}
              {result.trace && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-50 text-blue-500">
                  trace {result.trace}
                </span>
              )}
            </div>
          </motion.section>
        )}

        {/* 本次会话记录 */}
        {history.length > 0 && (
          <section className="glass rounded-2xl px-3.5 py-3">
            <p className="text-[11px] font-medium text-gray-400 mb-1.5">本次会话记录</p>
            <div className="space-y-1">
              {history.map((h, i) => (
                <div key={i} className="text-[12px] text-gray-600 truncate">
                  <span className="mr-1.5">{h.kind === 'ok' ? '✓' : h.kind === 'blocked' ? '🚫' : '✗'}</span>
                  {h.question}
                  <span className="text-gray-300 ml-1.5">{h.summary}</span>
                </div>
              ))}
            </div>
          </section>
        )}
      </motion.div>
    </div>
  )
}
