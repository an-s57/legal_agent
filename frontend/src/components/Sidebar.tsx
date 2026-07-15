import { motion } from 'framer-motion'
import Logo from './Logo'

interface SidebarProps {
  sessionId: string
  onNewSession: () => void
  caseSummary: Record<string, string>
  sessionList: { id: string; preview: string; time: number }[]
  onSelectSession: (id: string) => void
}

const summaryLabels: Record<string, string> = {
  case_type: '案件类型',
  event_description: '事件描述',
  event_time: '发生时间',
  damages: '损失/后果',
  user_claim: '用户诉求',
}

export default function Sidebar({ sessionId, onNewSession, caseSummary, sessionList, onSelectSession }: SidebarProps) {
  const hasSummary = Object.values(caseSummary).some(v => v && v.trim())

  return (
    <aside
      className="glass-sidebar flex flex-col z-10"
      style={{ width: '30%', maxWidth: '360px', padding: '20px 24px', height: '100vh', overflow: 'hidden' }}
    >
      {/* Logo + 名称 */}
      <div className="mb-5 flex-shrink-0 flex items-center gap-2">
        <Logo size={22} />
        <div>
          <h1 className="text-sm font-semibold text-ink leading-tight">LexAgent</h1>
          <p className="text-[12px] text-gray-400 leading-tight">AI 法律助手</p>
        </div>
      </div>

      {/* 新建会话 */}
      <button
        onClick={onNewSession}
        className="w-full py-1.5 px-3 rounded-3xl text-white text-[12px] font-medium
                   transition-all duration-300 ease-in-out hover:scale-[1.02] hover:shadow-lg flex-shrink-0"
        style={{ background: '#374151' }}
      >
        + 新建会话
      </button>

      {/* 会话列表 */}
      <div className="mt-5 flex-1 overflow-y-auto min-h-0">
        <p className="text-[11px] font-medium text-gray-400 mb-1.5 px-1">历史会话</p>
        <div className="space-y-0.5">
          {sessionList.length === 0 ? (
            <p className="text-[11px] text-gray-300 px-1">暂无历史会话</p>
          ) : (
            sessionList.map((s) => (
              <button
                key={s.id}
                onClick={() => onSelectSession(s.id)}
                className={`w-full text-left px-2.5 py-1.5 rounded-xl text-[12px] transition-all duration-200
                  ${s.id === sessionId
                    ? 'bg-gray-100 text-ink'
                    : 'text-gray-500 hover:bg-gray-50'
                  }`
                }
              >
                <p className="truncate">{s.preview}</p>
                <p className="text-[12px] text-gray-300 mt-0.5">
                  {new Date(s.time).toLocaleDateString('zh-CN')}
                </p>
              </button>
            ))
          )}
        </div>
      </div>

      {/* 案情摘要 */}
      {hasSummary && (
        <div className="mt-3 pt-3 border-t border-gray-100 flex-shrink-0 max-h-[35vh] overflow-y-auto">
          <p className="text-[11px] font-medium text-gray-400 mb-1.5">案情摘要</p>
          <div className="space-y-1.5">
            {Object.entries(summaryLabels).map(([key, label]) => {
              const value = caseSummary[key]
              if (!value || !value.trim()) return null
              return (
                <motion.div
                  key={key}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                  className="text-[11px]"
                >
                  <span className="text-gray-400">{label}</span>
                  <p className="text-gray-600 mt-0.5 leading-relaxed">{value}</p>
                </motion.div>
              )
            })}
          </div>
        </div>
      )}

      {/* 风险提示 */}
      <div className="mt-3 pt-3 border-t border-gray-100 flex-shrink-0">
        <p className="text-[12px] text-gray-300 leading-relaxed">
          本工具仅提供法律信息参考，不构成正式法律意见。具体情况请咨询专业律师。
        </p>
      </div>
    </aside>
  )
}
