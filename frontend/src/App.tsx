import { useState, useCallback, useEffect } from 'react'
import Background from './components/Background'
import Sidebar from './components/Sidebar'
import ChatArea, { type Message } from './components/ChatArea'
import InputBox from './components/InputBox'

const WELCOME = '你好，我是 AI 法律助手。请描述你遇到的法律问题，我会先了解案情再为你查找相关法条。'

interface SessionInfo {
  id: string
  messages: Message[]
  caseSummary: Record<string, string>
  preview: string
  time: number
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => 'session-' + Date.now())
  const [messages, setMessages] = useState<Message[]>([
    { role: 'bot', content: WELCOME },
  ])
  const [isTyping, setIsTyping] = useState(false)
  const [caseSummary, setCaseSummary] = useState<Record<string, string>>({})
  const [sessions, setSessions] = useState<SessionInfo[]>([])

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
    // 保存当前会话
    saveCurrentSession(sessionId, messages, caseSummary)
    const newId = 'session-' + Date.now()
    setSessionId(newId)
    setMessages([{ role: 'bot', content: WELCOME }])
    setCaseSummary({})
  }, [sessionId, messages, caseSummary, saveCurrentSession])

  const handleSelectSession = useCallback((id: string) => {
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
    setMessages(prev => [...prev, { role: 'user', content: msg }])
    setIsTyping(true)

    try {
      const res = await fetch('/legal/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: msg }),
      })
      const data = await res.json()

      let answerText = ''
      if (Array.isArray(data.answer)) {
        answerText = data.answer.map((s: string, i: number) => `${i + 1}. ${s}`).join('\n')
      } else {
        answerText = data.answer || '抱歉，无法生成回答'
      }

      const newMessages = [...messages, { role: 'user' as const, content: msg }, { role: 'bot' as const, content: answerText, tools: data.tools_used || [] }]
      setMessages(prev => [...prev, {
        role: 'bot',
        content: answerText,
        tools: data.tools_used || [],
      }])

      if (data.case_summary && typeof data.case_summary === 'object') {
        setCaseSummary(data.case_summary)
        saveCurrentSession(sessionId, newMessages, data.case_summary)
      } else {
        saveCurrentSession(sessionId, newMessages, caseSummary)
      }
    } catch (err) {
      setMessages(prev => [...prev, {
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
