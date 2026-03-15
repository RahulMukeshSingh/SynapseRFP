# backend/main.py
# ─────────────────────────────────────────────────────────────
# SynapseRFP — Execution Entry Point
# ─────────────────────────────────────────────────────────────

import uuid
from backend.agents.graph import app
from langchain_core.runnables import RunnableConfig
from backend.agents.state import GraphState
from typing import cast
from langgraph.errors import GraphRecursionError

def run_synapse_rfp(question: str):
    """
    Runs the multi-agent graph for a specific question.
    """
    
    # Initialize the State
    # We give it a unique thread_id so we can track this specific conversation
    config = cast(RunnableConfig, {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 5  # hard cap: prevents infinite Critic -> Drafter/Retriever loops. It will anyways be dormant after 3 retries of 'retry_count' below, but this is an extra safety measure.
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

    print(f"\n🚀 Starting SynapseRFP for: '{question}'\n")
    print("="*50)

    # Stream the Graph Execution
    # 'stream' allows us to see each node's output as it happens
    try:
        for event in app.stream(initial_state, config, version="v2"):
            for node_name, output in event.items():
                print(f"\n[NODE]: {node_name.upper()}")
                
                # Show specific updates based on the node
                if isinstance(output, dict):
                    if "sub_tasks" in output:
                        print(f"Planner created: {output.get('sub_tasks')}")
                    
                    if "draft_response" in output:
                        draft = output.get("draft_response", "")
                        print(f"Draft generated (first 100 chars): {str(draft)[:100]}...")
                    
                    if "critic_feedback" in output:
                        print(f"Critic says: {output.get('critic_feedback')}")
                else:
                    print(f"Node output: {type(output)}")
    except GraphRecursionError:
        print("Recursion limit hit - retry_count logic may have a bug")

    # 3. Get Final State
    final_state = app.get_state(config)
    return final_state.values.get("draft_response")

if __name__ == "__main__":
    # Test Question
    test_q = "What is your policy on data encryption at rest and in transit?"
    result = run_synapse_rfp(test_q)
    
    print("\n" + "="*50)
    print("FINAL VERIFIED RESPONSE:")
    print(result)