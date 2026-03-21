# backend/main.py
import os
import uuid
from typing import cast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from backend.agents.graph import app as graph_app
from backend.agents.state import GraphState
from nemoguardrails import LLMRails, RailsConfig

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
    """Generator function to stream LangGraph outputs."""
    config = cast(RunnableConfig, {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 5 
    })
    
    initial_state = cast(GraphState, {
        "question": question,
        "sub_tasks": [],
        "context": [],
        "retry_count": 0,
        "draft_response": "",   
        "critic_feedback": "",  
        "final_answer": ""      
    })

    # Stream graph execution
    async for event in graph_app.astream(initial_state, config, version="v2"):
        for node_name, output in event.items():
            if isinstance(output, dict): # for pylance type checking
                if "draft_response" in output and node_name == "drafter":
                    # Yielding the draft text to the frontend as it's generated
                    yield f"{output['draft_response']}\n\n"
                
                if "critic_feedback" in output:
                    yield f"_[System: Critic Evaluation - {output['critic_feedback']}]_\n\n"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
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