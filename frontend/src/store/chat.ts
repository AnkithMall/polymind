import { create } from 'zustand'
import { Message, Subtask, SubtaskResult, ExecutionMode } from '../types'

interface ChatStore {
  messages: Message[]
  isLoading: boolean
  executionMode: ExecutionMode
  addMessage: (msg: Message) => void
  appendToken: (id: string, token: string) => void
  setSubtasks: (id: string, subtasks: Subtask[]) => void
  setResults: (id: string, results: SubtaskResult[]) => void
  setStreaming: (id: string, streaming: boolean) => void
  setExecutionMode: (mode: ExecutionMode) => void
  setLoading: (v: boolean) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  isLoading: false,
  executionMode: 'sequential',

  addMessage: (msg) =>
    set((s) => ({ messages: [...s.messages, msg] })),

  appendToken: (id, token) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + token } : m
      ),
    })),

  setSubtasks: (id, subtasks) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, subtasks } : m
      ),
    })),

  setResults: (id, results) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, results } : m
      ),
    })),

  setStreaming: (id, isStreaming) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, isStreaming } : m
      ),
    })),

  setExecutionMode: (executionMode) => set({ executionMode }),
  setLoading: (isLoading) => set({ isLoading }),
  clearMessages: () => set({ messages: [] }),
}))
