import { useState, useRef, useEffect } from 'react'

interface InputBoxProps {
  onSend: (msg: string) => void
  disabled: boolean
}

export default function InputBox({ onSend, disabled }: InputBoxProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const ta = textareaRef.current
    if (ta) {
      ta.style.height = 'auto'
      ta.style.height = Math.min(ta.scrollHeight, 80) + 'px'
    }
  }, [value])

  const handleSend = () => {
    const msg = value.trim()
    if (!msg || disabled) return
    onSend(msg)
    setValue('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div style={{ padding: '0 72px 16px' }} className="flex-shrink-0">
      <div
        className="input-glow glass rounded-3xl flex items-end gap-2 px-3.5 py-2
                   transition-all duration-300 ease-in-out"
        style={{ boxShadow: '0 2px 20px rgba(0,0,0,0.04)' }}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="描述你遇到的法律问题..."
          rows={1}
          disabled={disabled}
          className="flex-1 bg-transparent resize-none outline-none text-[15px] text-ink
                     placeholder:text-gray-300 disabled:opacity-50"
          style={{ maxHeight: '80px' }}
        />
        <button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center
                     text-white transition-all duration-300 ease-in-out
                     hover:scale-105 hover:shadow-lg
                     disabled:bg-gray-200 disabled:scale-100 disabled:shadow-none"
          style={{ background: disabled || !value.trim() ? undefined : '#374151' }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
            <path
              d="M12 4L12 20M12 4L6 10M12 4L18 10"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
      <p className="text-center text-[10px] text-gray-300 mt-1.5">
        Agent 会先了解案情再检索法条 · 按 Enter 发送
      </p>
    </div>
  )
}
