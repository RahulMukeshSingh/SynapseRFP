// frontend/app/page.tsx
'use client';

import { useChat } from 'ai/react';
import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';

export default function Chat() {
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

  const { messages, input, handleInputChange, handleSubmit, isLoading, append } = useChat({
    api: `${API_BASE}/chat`,
  });

  const [isExtracting, setIsExtracting] = useState(false);
  const [questionQueue, setQuestionQueue] = useState<string[]>([]);
  const [isQueueRunning, setIsQueueRunning] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Robust Queue Processor
  useEffect(() => {
    // Only fire if the queue is active, the LLM is not loading, and we have questions left
    if (isQueueRunning && !isLoading && questionQueue.length > 0) {
      // Add a 1-second delay between automated questions to let the UI breathe
      // and prevent state race conditions
      const timer = setTimeout(() => {
        const nextQuestion = questionQueue[0];
        setQuestionQueue((prev) => prev.slice(1));
        append({ role: 'user', content: nextQuestion });
      }, 1000);
      
      return () => clearTimeout(timer);
      
    } else if (isQueueRunning && !isLoading && questionQueue.length === 0) {
      // Stop the queue once empty
      setIsQueueRunning(false);
    }
  }, [isLoading, questionQueue, isQueueRunning, append]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsExtracting(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Upload request failed');
      
      const data = await res.json();
      
      if (data.questions && Array.isArray(data.questions) && data.questions.length > 0) {
        setQuestionQueue(data.questions);
        setIsQueueRunning(true); // Start the automated processor
      } else {
        alert('No questions could be extracted from this document.');
      }
    } catch (error) {
      console.error(error);
      alert('Error parsing document. Check console for details.');
    } finally {
      setIsExtracting(false);
      if (fileInputRef.current) fileInputRef.current.value = ''; 
    }
  };

  return (
    <div className="flex flex-col w-full max-w-4xl py-12 mx-auto h-screen bg-white">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-slate-800">SynapseRFP</h1>
        <p className="text-slate-500">Agentic Security Questionnaire Responder</p>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto mb-6 p-4 space-y-6 border border-slate-200 rounded-xl bg-slate-50 shadow-inner">
        {messages.length === 0 && !isExtracting && questionQueue.length === 0 && (
          <div className="text-center text-slate-400 mt-20 flex flex-col items-center gap-4">
            <p>Ask a security question to get started, or upload an RFP document.</p>
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
                : 'bg-white text-slate-800 border border-slate-200 rounded-tl-none prose prose-slate max-w-none'
            }`}>
              {/* Use ReactMarkdown for assistant responses to format lists/bolding properly */}
              {m.role === 'user' ? (
                 <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
              ) : (
                 <ReactMarkdown>{m.content}</ReactMarkdown>
              )}
            </div>
          </div>
        ))}
        
        {isExtracting && (
          <div className="text-blue-500 animate-pulse text-sm font-semibold text-center mt-4">
            📄 Extracting questions from uploaded document...
          </div>
        )}

        {isLoading && !isExtracting && (
          <div className="text-slate-500 animate-pulse text-sm ml-2">
            Agents are analyzing documentation and drafting response...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Status Bar */}
      {isQueueRunning && (
        <div className="mb-3 px-4 py-2 bg-blue-50 text-blue-700 rounded-lg text-sm font-medium animate-pulse text-center border border-blue-200">
          Processing automated queue: {questionQueue.length} question(s) remaining...
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex gap-3 items-center">
        <input 
          type="file" 
          accept=".pdf,.xlsx" 
          ref={fileInputRef}
          onChange={handleFileUpload}
          className="hidden" 
          id="rfp-upload"
        />
        
        <label 
          htmlFor="rfp-upload"
          className={`cursor-pointer px-4 py-4 rounded-xl font-bold transition-colors flex items-center justify-center border-2 border-slate-300 ${
            isExtracting || isLoading || isQueueRunning
            ? 'bg-slate-100 text-slate-400 cursor-not-allowed border-solid' 
            : 'bg-white text-slate-600 hover:bg-slate-100 border-dashed'
          }`}
          title="Upload RFP Document"
        >
          📁 Upload
        </label>

        <input
          className="flex-1 p-4 border border-slate-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-black disabled:bg-slate-50 disabled:text-slate-400"
          value={input || ''}
          placeholder="e.g., What is our policy on data encryption at rest?"
          onChange={handleInputChange}
          disabled={isLoading || isExtracting || isQueueRunning}
        />
        
        <button 
          type="submit" 
          disabled={isLoading || isExtracting || isQueueRunning || !input || input.trim() === ''}
          className="px-8 py-4 bg-slate-800 text-white rounded-xl font-bold hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Send
        </button>
      </form>
    </div>
  );
}