# backend/main.py
import os
import uuid
import tempfile
import json
from typing import cast
from fastapi import FastAPI 
from fastapi import UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from backend.agents.graph import app as graph_app
from backend.agents.state import GraphState
from nemoguardrails import LLMRails, RailsConfig
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredExcelLoader
from backend.core.llm import get_llm

# Load Guardrails on startup


current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory where main.py currently lives
guardrails_path = os.path.join(current_dir, "guardrails") # Append 'guardrails' to it (.../backend/guardrails)
config = RailsConfig.from_path(guardrails_path)
rails = LLMRails(config)


app = FastAPI(title="SynapseRFP API")

# Allow Next.js frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # In production, change it to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: list

async def generate_response(question: str):
    """Generator function to stream actual LLM tokens via astream_events."""
    config = cast(RunnableConfig, {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 25 
    })
    
    initial_state = cast(GraphState, {
        "question": question,
        "sub_tasks": [], "context": [], "retry_count": 0,
        "draft_response": "", "critic_feedback": "", "final_answer": ""      
    })

    # FIX: Use astream_events to capture real-time LLM token generation
    async for event in graph_app.astream_events(initial_state, config, version="v2"):
        kind = event["event"]
        
        # Stream tokens specifically when the LLM is generating the draft
        if kind == "on_chat_model_stream":
            # Extract the token string
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content"):
                token = chunk.content
                if token:
                    yield token
                
        # Optional: Yield system status updates to the frontend
        elif kind == "on_chain_start" and event["name"] == "planner":
            yield "\n\n_[System: Analyzing RFP request and planning search queries...]_ \n\n"
        elif kind == "on_chain_start" and event["name"] == "retriever":
            yield "\n\n_[System: Retrieving relevant parent documents from vector store...]_ \n\n"
        elif kind == "on_chain_start" and event["name"] == "critic":
            yield "\n\n_[System: Critic is evaluating the draft against context...]_ \n\n"



@app.post("/api/upload")
async def upload_rfp_endpoint(file: UploadFile = File(...)):
    """
    Parses an uploaded .pdf or .xlsx RFP document and extracts a list of questions.
    """
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in [".pdf", ".xlsx"]:
        raise HTTPException(status_code=400, detail="Only .pdf and .xlsx files are supported.")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        # Load the document using Unstructured
        if ext == ".pdf":
            loader = UnstructuredPDFLoader(temp_path)
        else:
            loader = UnstructuredExcelLoader(temp_path)
            
        docs = loader.load()
        full_text = "\n".join([d.page_content for d in docs])

        # FIX: Use the centralized LLM factory!
        llm = get_llm() 
        
        prompt = f"""
        You are an RFP parsing assistant. Extract all the security or technical questions from the following document text.
        Return ONLY a raw JSON list of strings. Do not include markdown blocks like ```json.
        
        Text: {full_text[:300000]}  # Limiting to avoid massive token costs
        """
        
        response = llm.invoke(prompt)
        
        # Ensure content is treated purely as a string for json.loads
        content_str = response.content if isinstance(response.content, str) else str(response.content)
        
        try:
            questions = json.loads(content_str)
            return {"questions": questions}
        except json.JSONDecodeError:
            return {"error": "Failed to extract questions cleanly.", "raw_output": content_str}

    finally:
        # Clean up the temp file
        os.remove(temp_path)



@app.post("/api/chat/sync")
async def chat_sync_endpoint(request: ChatRequest):
    """
    Synchronous endpoint that returns the final payload instead of streaming.
    """
    if not request.messages:
        return {"error": "No messages provided."}
        
    user_query = request.messages[-1]["content"]
    
    # NeMo Guardrail Check
    safe_response = await rails.generate_async(messages=[{"role": "user", "content": user_query}])
    
    if isinstance(safe_response, dict):
        response_text = safe_response.get("content", "")
    elif isinstance(safe_response, str):
        response_text = safe_response
    else:
        response_text = str(safe_response)

    if response_text.startswith("GUARDRAIL_BLOCKED:"):
        return {"final_answer": f"_[System: Request blocked by security guardrails.]_\n\n{response_text}"}

    # Configure State
    config = cast(RunnableConfig, {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 25 
    })
    
    initial_state = cast(GraphState, {
        "question": user_query,
        "sub_tasks": [],
        "context": [],
        "retry_count": 0,
        "draft_response": "",   
        "critic_feedback": "",
        "final_answer": ""      
    })

    # Invoke the graph (Wait for it to finish completely)
    final_state = await graph_app.ainvoke(initial_state, config)
    
    # Return the final JSON payload
    return {"final_answer": final_state.get("final_answer", "")}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.messages:
        return StreamingResponse(iter(["Error: No messages provided."]), media_type="text/event-stream")
    
    user_query = request.messages[-1]["content"]
    # 1. NeMo Guardrail Check
    safe_response = await rails.generate_async(messages=[{"role": "user", "content": user_query}])
    
    # --- FIX: Safely extract the string to satisfy Pylance ---
    if isinstance(safe_response, dict):
        response_text = safe_response.get("content", "")
    elif isinstance(safe_response, str):
        response_text = safe_response
    else:
        # Fallback just in case it returns a GenerationResponse object
        response_text = str(safe_response)
    # ---------------------------------------------------------

    # If NeMo blocks it, it will return a pre-canned refusal string instead of passing it to LangGraph
    if response_text.startswith("GUARDRAIL_BLOCKED:"):
        # Yield the blocked response
        async def blocked_stream():
            yield f"_[System: Request blocked by security guardrails.]_\n\n{response_text}"
        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    # If safe, pass to LangGraph
    return StreamingResponse(generate_response(user_query), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)