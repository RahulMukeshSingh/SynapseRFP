# backend/agents/state.py
# ─────────────────────────────────────────────────────────────
# SynapseRFP — Graph State Definition
# This serves as the shared memory for all agents.
# ─────────────────────────────────────────────────────────────

from typing import Annotated, List, TypedDict, Union
from operator import add

class GraphState(TypedDict):
    """
    Represents the state of our multi-agent RFP responder.
    """
    
    question: str # The original question or RFP requirement from the user

    # Annotated[..., add] means new sub-tasks are APPENDED to the list
    sub_tasks: Annotated[List[str], add] # Planner Agent breaks the question into specific sub-tasks
    context: Annotated[List[str], add] # Retriever Agent stores the raw parent chunks found in Redis here
    draft_response: str  # Drafter Agent stores its initial technical response here
    critic_feedback: str # Critic Agent stores its evaluation (pass, rewrite, retrieve_more) and any specific feedback for the other agents    
    retry_count: int # Important for preventing infinite loops
    final_answer: str # Final verified answer sent back to the user