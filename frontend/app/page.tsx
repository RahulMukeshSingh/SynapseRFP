// frontend/app/page.tsx
'use client';

import { useChat } from 'ai/react';
import { useEffect, useRef } from 'react';

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: process.env.NEXT_PUBLIC_API_URL 
      ? `${process.env.NEXT_PUBLIC_API_URL}/chat` 
      : 'http://localhost:8000/api/chat',
  });

  // Auto-scroll to the bottom of the chat
  const messagesEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col w-full max-w-4xl py-12 mx-auto h-screen bg-white">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-slate-800">SynapseRFP</h1>
        <p className="text-slate-500">Agentic Security Questionnaire Responder</p>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto mb-6 p-4 space-y-6 border border-slate-200 rounded-xl bg-slate-50 shadow-inner">
        {messages.length === 0 && (
          <div className="text-center text-slate-400 mt-20">
            Ask a security question to get started.
          </div>
        )}
        
        {messages.map(m => (
          <div key={m.id} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
            <span className="text-xs font-bold text-slate-400 mb-1 ml-1">
              {m.role === 'user' ? 'You' : 'Synapse Agents'}
            </span>
            <div className={`p-4 rounded-2xl max-w-[80%] shadow-sm ${
              m.role === 'user' 
                ? 'bg-blue-600 text-white rounded-tr-none' 
                : 'bg-white text-slate-800 border border-slate-200 rounded-tl-none'
            }`}>
              <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="text-slate-500 animate-pulse text-sm ml-2">
            Agents are analyzing documentation and drafting response...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          className="flex-1 p-4 border border-slate-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-black"
          value={input || ''}
          placeholder="e.g., What is our policy on data encryption at rest?"
          onChange={handleInputChange}
          disabled={isLoading}
        />
        <button 
          type="submit" 
          disabled={isLoading || !input || input.trim() === ''}
          className="px-8 py-4 bg-slate-800 text-white rounded-xl font-bold hover:bg-slate-700 disabled:opacity-50 transition-colors"
        >
          Send
        </button>
      </form>
    </div>
  );
}