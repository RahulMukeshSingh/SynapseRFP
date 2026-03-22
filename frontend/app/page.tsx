// frontend/app/page.tsx
'use client';

import { useChat } from 'ai/react';
import { useEffect, useRef, useState } from 'react';

export default function Chat() {
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

  // We destructure 'append' so we can programmatically send messages from the queue
  const { messages, input, handleInputChange, handleSubmit, isLoading, append } = useChat({
    api: `${API_BASE}/chat`,
  });

  // State for handling file uploads and the sequential queue
  const [isExtracting, setIsExtracting] = useState(false);
  const [questionQueue, setQuestionQueue] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to the bottom of the chat
  const messagesEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Queue Processor: Watches the queue and triggers the next question when the agent is free
  useEffect(() => {
    if (!isLoading && questionQueue.length > 0) {
      const nextQuestion = questionQueue[0];
      // Remove the question we are about to ask from the queue
      setQuestionQueue((prev) => prev.slice(1));
      // Programmatically send the message as the user
      append({ role: 'user', content: nextQuestion });
    }
  }, [isLoading, questionQueue, append]);

  // Handles the file selection and extraction endpoint
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
        // Load the extracted questions into the queue!
        setQuestionQueue(data.questions);
      } else {
        alert('No questions could be extracted from this document.');
      }
    } catch (error) {
      console.error(error);
      alert('Error parsing document. Check console for details.');
    } finally {
      setIsExtracting(false);
      // Reset the file input so the user can upload the same file again if needed
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
                : 'bg-white text-slate-800 border border-slate-200 rounded-tl-none'
            }`}>
              <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
            </div>
          </div>
        ))}
        
        {/* Loading Indicators */}
        {isExtracting && (
          <div className="text-blue-500 animate-pulse text-sm font-semibold text-center mt-4">
            📄 Extracting questions from uploaded document...
          </div>
        )}

        {isLoading && (
          <div className="text-slate-500 animate-pulse text-sm ml-2">
            Agents are analyzing documentation and drafting response...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Status Bar (Shows if items are queued) */}
      {questionQueue.length > 0 && (
        <div className="mb-3 px-4 py-2 bg-blue-50 text-blue-700 rounded-lg text-sm font-medium animate-pulse text-center">
          Processing automated queue: {questionQueue.length} question(s) remaining...
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex gap-3 items-center">
        {/* Hidden File Input */}
        <input 
          type="file" 
          accept=".pdf,.xlsx" 
          ref={fileInputRef}
          onChange={handleFileUpload}
          className="hidden" 
          id="rfp-upload"
        />
        
        {/* Upload Button */}
        <label 
          htmlFor="rfp-upload"
          className={`cursor-pointer px-4 py-4 rounded-xl font-bold transition-colors flex items-center justify-center border-2 border-slate-300 ${
            isExtracting || isLoading 
            ? 'bg-slate-100 text-slate-400 cursor-not-allowed' 
            : 'bg-white text-slate-600 hover:bg-slate-100 border-dashed'
          }`}
          title="Upload RFP Document"
        >
          📁 Upload
        </label>

        <input
          className="flex-1 p-4 border border-slate-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-black"
          value={input || ''}
          placeholder="e.g., What is our policy on data encryption at rest?"
          onChange={handleInputChange}
          disabled={isLoading || isExtracting}
        />
        
        <button 
          type="submit" 
          disabled={isLoading || isExtracting || !input || input.trim() === ''}
          className="px-8 py-4 bg-slate-800 text-white rounded-xl font-bold hover:bg-slate-700 disabled:opacity-50 transition-colors"
        >
          Send
        </button>
      </form>
    </div>
  );
}