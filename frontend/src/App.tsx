import { useState, useRef, useEffect } from 'react'
import { Send, Trash2, Brain, Zap, GitBranch } from 'lucide-react'
import { useChatStore } from './store/chat'
import { MessageBubble } from './components/MessageBubble'
import { Subtask, SubtaskResult } from './types'

const API = ''  // empty = same origin (proxied by Vite)

export default function App() {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const {
    messages, isLoading, executionMode,
    addMessage, appendToken, setSubtasks, setResults, setStreaming,
    setLoading, setExecutionMode, clearMessages,
  } = useChatStore()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    const prompt = input.trim()
    if (!prompt || isLoading) return

    setInput('')
    setLoading(true)

    // Add user message
    const userId = crypto.randomUUID()
    addMessage({ id: userId, role: 'user', content: prompt })

    // Add placeholder assistant message
    const assistantId = crypto.randomUUID()
    addMessage({ id: assistantId, role: 'assistant', content: '', isStreaming: true })

    try {
      const res = await fetch(`${API}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: prompt }],
          stream: true,
          execution_mode: executionMode,
        }),
      })

      if (!res.body) throw new Error('No response body')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (raw === '[DONE]') continue

          try {
            const event = JSON.parse(raw)

            if (event.event === 'plan') {
              setSubtasks(assistantId, event.subtasks as Subtask[])
            } else if (event.event === 'subtask_results') {
              setResults(assistantId, event.results as SubtaskResult[])
            } else if (event.event === 'token') {
              appendToken(assistantId, event.content)
            } else if (event.event === 'done') {
              setStreaming(assistantId, false)
            } else if (event.event === 'error') {
              appendToken(assistantId, `\n\n⚠️ Error: ${event.message}`)
              setStreaming(assistantId, false)
            }
          } catch {
            // ignore parse errors on individual lines
          }
        }
      }
    } catch (err) {
      appendToken(assistantId, `⚠️ Connection error: ${err}`)
      setStreaming(assistantId, false)
    } finally {
      setStreaming(assistantId, false)
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white">

      {/* Header */}
      <header className="flex items-center gap-3 px-5 py-3.5 border-b border-white/10 bg-gray-950/80 backdrop-blur-sm shrink-0">
        <Brain size={20} className="text-brand-500" />
        <span className="font-semibold text-white tracking-tight">PolyMind</span>
        <span className="text-xs text-gray-500 hidden sm:block">multi-specialist LLM orchestrator</span>

        <div className="ml-auto flex items-center gap-2">
          {/* Execution mode toggle */}
          <div className="flex items-center gap-1 bg-white/5 rounded-lg p-1">
            <button
              onClick={() => setExecutionMode('sequential')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                executionMode === 'sequential'
                  ? 'bg-brand-700 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Zap size={12} />
              Sequential
            </button>
            <button
              onClick={() => setExecutionMode('parallel')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                executionMode === 'parallel'
                  ? 'bg-brand-700 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <GitBranch size={12} />
              Parallel
            </button>
          </div>

          <button
            onClick={clearMessages}
            title="Clear chat"
            className="p-2 rounded-lg text-gray-500 hover:text-white hover:bg-white/5 transition-colors"
          >
            <Trash2 size={15} />
          </button>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-20 space-y-3">
              <Brain size={40} className="text-brand-700 mx-auto" />
              <p className="text-gray-400 text-lg font-medium">What would you like to know?</p>
              <p className="text-gray-600 text-sm">
                PolyMind will decompose your prompt and route each part to the best specialist model.
              </p>
              <div className="flex flex-wrap gap-2 justify-center mt-6">
                {[
                  'Explain recursion and write a Python example',
                  'Solve: integrate x² + 3x from 0 to 5',
                  'Write a short story about a robot learning to paint',
                ].map((example) => (
                  <button
                    key={example}
                    onClick={() => setInput(example)}
                    className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white text-xs transition-colors text-left"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="shrink-0 border-t border-white/10 px-4 py-4 bg-gray-950">
        <div className="max-w-3xl mx-auto flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything… (Shift+Enter for new line)"
            rows={1}
            className="flex-1 resize-none bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-brand-500/50 focus:ring-1 focus:ring-brand-500/20 transition-colors max-h-40"
            style={{ minHeight: '48px' }}
            onInput={(e) => {
              const el = e.currentTarget
              el.style.height = 'auto'
              el.style.height = `${Math.min(el.scrollHeight, 160)}px`
            }}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="shrink-0 w-11 h-11 rounded-xl bg-brand-700 hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
          >
            <Send size={16} className="text-white" />
          </button>
        </div>
        <p className="max-w-3xl mx-auto mt-2 text-xs text-gray-600 text-center">
          Mode: <span className="text-gray-500">{executionMode}</span>
          {executionMode === 'sequential' && ' · safe for single-GPU laptops'}
          {executionMode === 'parallel' && ' · runs subtasks concurrently'}
        </p>
      </div>
    </div>
  )
}
