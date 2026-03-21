# backend/main.py
import uuid
from typing import cast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.runnables import RunnableConfig
from backend.agents.graph import app as graph_app
from backend.agents.state import GraphState

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
    # Get the last message from the user
    user_query = request.messages[-1]["content"]
    
    # Return a streaming response back to the Next.js frontend
    return StreamingResponse(generate_response(user_query), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)