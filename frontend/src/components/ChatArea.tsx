import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export interface Message {
  role: 'user' | 'bot'
  content: string
  tools?: string[]
}

interface ChatAreaProps {
  messages: Message[]
  isTyping: boolean
}

export default function ChatArea({ messages, isTyping }: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  const handleCopy = (text: string, idx: number) => {
    navigator.clipboard.writeText(text)
    setCopiedIdx(idx)
    setTimeout(() => setCopiedIdx(null), 2000)
  }

  return (
    <div className="flex-1 overflow-y-auto flex flex-col gap-2" style={{ padding: '16px 72px' }}>
      <AnimatePresence mode="popLayout">
        {messages.map((msg, idx) => (
          <motion.div
            key={idx}
            layout
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className="group relative max-w-[70%]">
              <div
                className={`px-3.5 py-2 text-[11px] leading-relaxed
                  ${msg.role === 'user'
                    ? 'rounded-2xl rounded-br-md text-white'
                    : 'glass rounded-2xl rounded-bl-md text-ink'
                  }`
                }
                style={msg.role === 'user' ? { background: '#374151' } : {}}
              >
                {msg.role === 'bot' && idx === 0 && (
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-[9px] font-medium text-gray-400">LexAgent</span>
                  </div>
                )}
                <div className="whitespace-pre-wrap">{msg.content}</div>
                {msg.tools && msg.tools.length > 0 && (
                  <div className="mt-1.5 pt-1.5 border-t border-gray-100 flex gap-1.5 flex-wrap">
                    {msg.tools.map((tool) => (
                      <span
                        key={tool}
                        className="text-[8px] px-1.5 py-0.5 rounded-full bg-blue-50 text-blue-500"
                      >
                        {tool}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              {/* 复制按钮 */}
              {msg.role === 'bot' && idx !== 0 && (
                <button
                  onClick={() => handleCopy(msg.content, idx)}
                  className="absolute -bottom-4 left-1 text-[8px] text-gray-300
                             opacity-0 group-hover:opacity-100 transition-opacity duration-200
                             hover:text-gray-500"
                >
                  {copiedIdx === idx ? '已复制' : '复制'}
                </button>
              )}
            </div>
          </motion.div>
        ))}
      </AnimatePresence>

      {/* Typing indicator */}
      <AnimatePresence>
        {isTyping && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex justify-start"
          >
            <div className="glass rounded-2xl rounded-bl-md px-3.5 py-2.5 flex items-center gap-1">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div ref={bottomRef} />
    </div>
  )
}
