export interface Subtask {
  id: string
  domain: string
  description: string
  depends_on: string[]
}

export interface SubtaskResult {
  subtask_id: string
  domain: string
  description: string
  output: string
  model_used: string
  provider: string
  latency_ms: number
  input_tokens: number
  output_tokens: number
  fallback_used: boolean
  error: string | null
}

export type MessageRole = 'user' | 'assistant'

export interface Message {
  id: string
  role: MessageRole
  content: string
  subtasks?: Subtask[]
  results?: SubtaskResult[]
  isStreaming?: boolean
}

export type ExecutionMode = 'sequential' | 'parallel'
