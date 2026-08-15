import { useState, useCallback, useEffect, useRef } from 'react'
import Background from './components/Background'
import Sidebar from './components/Sidebar'
import ChatArea, { type Message } from './components/ChatArea'
import InputBox from './components/InputBox'

const WELCOME = '你好，我是 AI 法律助手。请描述你遇到的法律问题，我会先了解案情再为你查找相关法条。'
const ACTIVE_SESSION_STORAGE_KEY = 'lexagent_active_session_id'

function createSessionId(): string {
  return `session-${crypto.randomUUID()}`
}

/**
 * 仅处理不会影响案情判断的纯闲聊，避免“你好”也进入 Planner + Agent 链路。
 * 法律问题、带具体事实的问题仍然必须发送到后端。
 */
function getQuickReply(message: string): string | null {
  const normalized = message
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[!！?？。．.]/g, '')

  if (['你好', '您好', '嗨', '哈喽', 'hello', 'hi', '在吗', '在不在'].includes(normalized)) {
    return '你好！我是 AI 法律助手。你可以直接描述遇到的法律问题，我会先了解案情，再帮你查找相关法条。'
  }

  if (['你是谁', '你能做什么', '你可以做什么', '你能干什么', '有什么功能'].includes(normalized)) {
    return '我可以协助梳理法律咨询中的关键信息、检索本地法律资料，并在需要时联网查询较新的法律信息。你可以直接说说发生了什么。'
  }

  if (['谢谢', '谢谢你', '感谢', 'thanks', 'thank you'].includes(normalized)) {
    return '不客气。如果你愿意，可以继续补充案情的时间、经过、损失和希望解决的问题。'
  }

  return null
}

interface SessionInfo {
  id: string
  messages: Message[]
  caseSummary: Record<string, string>
  preview: string
  time: number
}

interface PersistedTurn {
  human: string
  ai: string
}

interface PersistedSessionResponse {
  history: PersistedTurn[]
  case_summary: Record<string, string>
}

export default function App() {
  const [sessionId, setSessionId] = useState(
    () => localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY) || createSessionId(),
  )
  const [messages, setMessages] = useState<Message[]>([
    { id: crypto.randomUUID(), role: 'bot', content: WELCOME },
  ])
  const [isTyping, setIsTyping] = useState(false)
  const [caseSummary, setCaseSummary] = useState<Record<string, string>>({})
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  // 防止刷新恢复请求在用户已经开始新操作后，迟到并覆盖当前界面。
  const restoreVersionRef = useRef(0)

  // 记住当前会话 ID；刷新后由后端 SQLite 恢复这一个会话的历史。
  useEffect(() => {
    localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, sessionId)
  }, [sessionId])

  useEffect(() => {
    let cancelled = false
    const restoreVersion = ++restoreVersionRef.current

    const restoreActiveSession = async () => {
      try {
        const res = await fetch(`/legal/session/${encodeURIComponent(sessionId)}`)
        if (!res.ok) return

        const saved = await res.json() as PersistedSessionResponse
        if (
          cancelled
          || restoreVersion !== restoreVersionRef.current
          || !Array.isArray(saved.history)
          || saved.history.length === 0
        ) return

        const restoredMessages: Message[] = [
          { id: crypto.randomUUID(), role: 'bot', content: WELCOME },
          ...saved.history.flatMap(turn => [
            { id: crypto.randomUUID(), role: 'user' as const, content: turn.human },
            { id: crypto.randomUUID(), role: 'bot' as const, content: turn.ai },
          ]),
        ]
        const restoredSummary = saved.case_summary || {}
        const firstUserMessage = saved.history[0]?.human || '历史会话'

        setMessages(restoredMessages)
        setCaseSummary(restoredSummary)
        setSessions(prev => [
          {
            id: sessionId,
            messages: restoredMessages,
            caseSummary: restoredSummary,
            preview: firstUserMessage.slice(0, 30),
            time: Date.now(),
          },
          ...prev.filter(session => session.id !== sessionId),
        ])
      } catch {
        // 后端暂不可用时保留新会话的欢迎语，避免页面初始化失败。
      }
    }

    void restoreActiveSession()
    return () => { cancelled = true }
  }, [sessionId])

  // 保存当前会话到 sessions 列表
  const saveCurrentSession = useCallback((sid: string, msgs: Message[], summary: Record<string, string>) => {
    if (msgs.length <= 1) return // 只有欢迎语，不保存
    const firstUserMsg = msgs.find(m => m.role === 'user')
    const preview = firstUserMsg ? firstUserMsg.content.slice(0, 30) : '新会话'
    setSessions(prev => {
      const existing = prev.find(s => s.id === sid)
      if (existing) {
        return prev.map(s => s.id === sid
          ? { ...s, messages: msgs, caseSummary: summary, preview, time: Date.now() }
          : s
        )
      }
      return [{ id: sid, messages: msgs, caseSummary: summary, preview, time: Date.now() }, ...prev]
    })
  }, [])

  const handleNewSession = useCallback(() => {
    restoreVersionRef.current += 1
    // 保存当前会话
    saveCurrentSession(sessionId, messages, caseSummary)
    const newId = createSessionId()
    setSessionId(newId)
    setMessages([{ id: crypto.randomUUID(), role: 'bot', content: WELCOME }])
    setCaseSummary({})
  }, [sessionId, messages, caseSummary, saveCurrentSession])

  const handleSelectSession = useCallback((id: string) => {
    restoreVersionRef.current += 1
    // 保存当前会话
    saveCurrentSession(sessionId, messages, caseSummary)
    // 切换到选中的会话
    const target = sessions.find(s => s.id === id)
    if (target) {
      setSessionId(target.id)
      setMessages(target.messages)
      setCaseSummary(target.caseSummary)
    }
  }, [sessionId, messages, caseSummary, saveCurrentSession, sessions])

  const handleSend = useCallback(async (msg: string) => {
    restoreVersionRef.current += 1
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: msg }

    // 纯问候/闲聊由前端立即响应，不请求后端，也不写入服务端案情摘要。
    const quickReply = getQuickReply(msg)
    if (quickReply) {
      const botMsg: Message = { id: crypto.randomUUID(), role: 'bot', content: quickReply }
      const newMessages: Message[] = [...messages, userMsg, botMsg]
      setMessages(newMessages)
      saveCurrentSession(sessionId, newMessages, caseSummary)
      return
    }

    setMessages(prev => [...prev, userMsg])
    setIsTyping(true)

    const botId = crypto.randomUUID()
    let botContent = ''
    let toolsUsed: string[] = []
    let botAdded = false
    let finalSummary = caseSummary

    try {
      const res = await fetch('/legal/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: msg }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buf += decoder.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop() || ''

        for (const part of parts) {
          for (const line of part.split('\n')) {
            if (!line.startsWith('data: ')) continue
            try {
              const event = JSON.parse(line.slice(6))

              if (event.type === 'token') {
                botContent += event.text
                if (!botAdded) {
                  setMessages(prev => [...prev, { id: botId, role: 'bot', content: botContent }])
                  botAdded = true
                } else {
                  setMessages(prev => {
                    const next = [...prev]
                    next[next.length - 1] = { ...next[next.length - 1], content: botContent }
                    return next
                  })
                }
              } else if (event.type === 'planner_question') {
                botContent = event.text
                setMessages(prev => [...prev, { id: botId, role: 'bot', content: event.text }])
                botAdded = true
              } else if (event.type === 'tool_start') {
                toolsUsed.push(event.name)
              } else if (event.type === 'done') {
                if (event.tools_used) toolsUsed = event.tools_used
                // 服务端随后可能还会推送 case_summary 事件（摘要更新），
                // 先结束输入状态，避免转圈动画等待摘要的 LLM 调用。
                setIsTyping(false)
              } else if (event.type === 'case_summary') {
                finalSummary = event.data || {}
                setCaseSummary(finalSummary)
              } else if (event.type === 'error') {
                // 服务端流式处理中途出错：展示错误信息，避免留下空白气泡
                botContent = event.message || '请求处理失败'
                if (!botAdded) {
                  setMessages(prev => [...prev, { id: botId, role: 'bot', content: botContent }])
                  botAdded = true
                } else {
                  setMessages(prev => {
                    const next = [...prev]
                    next[next.length - 1] = { ...next[next.length - 1], content: botContent }
                    return next
                  })
                }
              }
            } catch { /* skip malformed events */ }
          }
        }
      }

      // Stream complete — attach tools to final message and save session
      const botMsg: Message = { id: botId, role: 'bot', content: botContent, tools: toolsUsed }
      const newMessages: Message[] = [...messages, userMsg, botMsg]

      setMessages(prev => {
        const next = [...prev]
        if (next.length > 0 && next[next.length - 1].role === 'bot') {
          next[next.length - 1] = botMsg
        }
        return next
      })

      saveCurrentSession(sessionId, newMessages, finalSummary)

    } catch (err) {
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'bot',
        content: '请求失败，请确认服务器已启动。',
      }])
    } finally {
      setIsTyping(false)
    }
  }, [sessionId, messages, caseSummary, saveCurrentSession])

  return (
    <div className="relative h-screen flex">
      <Background />
      <Sidebar
        sessionId={sessionId}
        onNewSession={handleNewSession}
        caseSummary={caseSummary}
        sessionList={sessions.map(s => ({ id: s.id, preview: s.preview, time: s.time }))}
        onSelectSession={handleSelectSession}
      />
      <main className="flex-1 flex flex-col z-10">
        <ChatArea messages={messages} isTyping={isTyping} />
        <InputBox onSend={handleSend} disabled={isTyping} />
      </main>
    </div>
  )
}
