import { Message } from '../types'
import { TransparencyPanel } from './TransparencyPanel'
import { User, Brain } from 'lucide-react'

interface Props {
  message: Message
}

// Minimal markdown renderer: bold, inline code, code blocks, line breaks
function renderMarkdown(text: string) {
  const lines = text.split('\n')
  const elements: React.ReactNode[] = []
  let inCodeBlock = false
  let codeLines: string[] = []
  let lang = ''

  lines.forEach((line, i) => {
    if (line.startsWith('```')) {
      if (!inCodeBlock) {
        inCodeBlock = true
        lang = line.slice(3).trim()
        codeLines = []
      } else {
        inCodeBlock = false
        elements.push(
          <pre key={i} className="my-3 rounded-lg bg-black/40 border border-white/10 overflow-x-auto">
            {lang && (
              <div className="px-4 py-1.5 text-xs text-gray-500 border-b border-white/10 font-mono">
                {lang}
              </div>
            )}
            <code className="block px-4 py-3 text-sm text-gray-200 font-mono whitespace-pre">
              {codeLines.join('\n')}
            </code>
          </pre>
        )
        codeLines = []
        lang = ''
      }
      return
    }

    if (inCodeBlock) {
      codeLines.push(line)
      return
    }

    if (line.startsWith('### ')) {
      elements.push(<h3 key={i} className="text-base font-semibold text-white mt-4 mb-1">{line.slice(4)}</h3>)
    } else if (line.startsWith('## ')) {
      elements.push(<h2 key={i} className="text-lg font-semibold text-white mt-5 mb-2">{line.slice(3)}</h2>)
    } else if (line.startsWith('# ')) {
      elements.push(<h1 key={i} className="text-xl font-bold text-white mt-5 mb-2">{line.slice(2)}</h1>)
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      elements.push(
        <li key={i} className="ml-4 list-disc text-gray-200 leading-relaxed">
          {inlineFormat(line.slice(2))}
        </li>
      )
    } else if (line.trim() === '') {
      elements.push(<div key={i} className="h-2" />)
    } else {
      elements.push(<p key={i} className="text-gray-200 leading-relaxed">{inlineFormat(line)}</p>)
    }
  })

  return elements
}

function inlineFormat(text: string): React.ReactNode {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={i} className="px-1.5 py-0.5 rounded bg-black/40 text-blue-300 text-sm font-mono">{part.slice(1, -1)}</code>
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} className="text-white font-semibold">{part.slice(2, -2)}</strong>
    }
    return part
  })
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex gap-3 justify-end">
        <div className="max-w-[80%] bg-brand-700 rounded-2xl rounded-tr-sm px-4 py-3 text-white text-sm leading-relaxed">
          {message.content}
        </div>
        <div className="shrink-0 w-8 h-8 rounded-full bg-brand-700 flex items-center justify-center mt-1">
          <User size={14} className="text-white" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-3">
      <div className="shrink-0 w-8 h-8 rounded-full bg-brand-900 border border-brand-700/50 flex items-center justify-center mt-1">
        <Brain size={14} className="text-brand-500" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-sm">
          {message.isStreaming && message.content === '' ? (
            <div className="flex items-center gap-1.5 text-gray-500">
              <span className="w-1.5 h-1.5 bg-brand-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1.5 h-1.5 bg-brand-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1.5 h-1.5 bg-brand-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          ) : (
            <div className="space-y-0.5">{renderMarkdown(message.content)}</div>
          )}
        </div>

        {message.results && message.subtasks && message.results.length > 0 && (
          <TransparencyPanel subtasks={message.subtasks} results={message.results} />
        )}
      </div>
    </div>
  )
}
