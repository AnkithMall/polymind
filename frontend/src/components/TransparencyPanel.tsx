import { useState } from 'react'
import { ChevronDown, ChevronUp, Zap, Clock, AlertTriangle } from 'lucide-react'
import { Subtask, SubtaskResult } from '../types'

const DOMAIN_COLORS: Record<string, string> = {
  code:          'bg-blue-500/20 text-blue-300 border-blue-500/30',
  math:          'bg-purple-500/20 text-purple-300 border-purple-500/30',
  creative:      'bg-pink-500/20 text-pink-300 border-pink-500/30',
  research:      'bg-green-500/20 text-green-300 border-green-500/30',
  summarization: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  translation:   'bg-orange-500/20 text-orange-300 border-orange-500/30',
  qa:            'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
  general:       'bg-gray-500/20 text-gray-300 border-gray-500/30',
}

function domainBadge(domain: string) {
  const cls = DOMAIN_COLORS[domain] ?? DOMAIN_COLORS.general
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full border ${cls}`}>
      {domain}
    </span>
  )
}

interface Props {
  subtasks: Subtask[]
  results: SubtaskResult[]
}

export function TransparencyPanel({ subtasks, results }: Props) {
  const [open, setOpen] = useState(false)

  const totalLatency = results.reduce((s, r) => s + r.latency_ms, 0)
  const totalTokens  = results.reduce((s, r) => s + r.input_tokens + r.output_tokens, 0)
  const hasFallback  = results.some((r) => r.fallback_used)
  const hasError     = results.some((r) => r.error && r.model_used === 'none')

  return (
    <div className="mt-3 border border-white/10 rounded-xl overflow-hidden text-sm">
      {/* Header row */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-3 px-4 py-2.5 bg-white/5 hover:bg-white/8 transition-colors text-left"
      >
        <Zap size={13} className="text-brand-500 shrink-0" />
        <span className="text-gray-400 text-xs">
          {results.length} specialist{results.length !== 1 ? 's' : ''}
          {' · '}
          <span className="text-gray-300">{(totalLatency / 1000).toFixed(1)}s</span>
          {totalTokens > 0 && (
            <> · <span className="text-gray-300">{totalTokens.toLocaleString()} tokens</span></>
          )}
          {hasFallback && <> · <span className="text-amber-400">fallback used</span></>}
          {hasError    && <> · <span className="text-red-400">error</span></>}
        </span>
        <span className="ml-auto text-gray-500">
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </span>
      </button>

      {/* Detail rows */}
      {open && (
        <div className="divide-y divide-white/5">
          {results.map((r) => {
            const plan = subtasks.find((s) => s.id === r.subtask_id)
            return (
              <div key={r.subtask_id} className="px-4 py-3 bg-white/[0.02]">
                <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                  {domainBadge(r.domain)}
                  <span className="text-gray-300 text-xs font-mono">{r.model_used}</span>
                  <span className="text-gray-500 text-xs">via {r.provider}</span>
                  {r.fallback_used && (
                    <span className="text-xs text-amber-400 flex items-center gap-1">
                      <AlertTriangle size={11} /> fallback
                    </span>
                  )}
                  <span className="ml-auto text-gray-500 text-xs flex items-center gap-1">
                    <Clock size={11} /> {r.latency_ms}ms
                  </span>
                </div>
                <p className="text-gray-400 text-xs leading-relaxed line-clamp-2">
                  {plan?.description ?? r.description}
                </p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
