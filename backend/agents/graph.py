# backend/agents/graph.py
# ─────────────────────────────────────────────────────────────
# SynapseRFP — State Graph definition
# ─────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END
from backend.agents.state import GraphState
from backend.agents.nodes import (
    planner_node, 
    retriever_node, 
    drafter_node, 
    critic_node
)

# Initialize the Graph
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("drafter", drafter_node)
workflow.add_node("critic", critic_node)

# Flow
workflow.set_entry_point("planner")

workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "drafter")
workflow.add_edge("drafter", "critic")

#Logic
def route_after_critic(state: GraphState):
    """
    This function looks at the critic's decision and 
    chooses the next path.
    """
    feedback = state["critic_feedback"]
    
    # Safety Check: If we've tried too many times, just stop
    if state.get("retry_count", 0) >= 3:
        return END

    if "[PASS]" in feedback:
        return END
    elif "[RETRIEVE_MORE]" in feedback:
        return "planner" # Go back to start to get better queries
    else:
        return "drafter" # Go back to writing to fix errors

workflow.add_conditional_edges(
    "critic", # After the critic node, we need to decide where to go based on its feedback
    route_after_critic, # This function will inspect the critic's feedback and return the next node to go to
    {
        "planner": "planner", # {'What route_after_critic returns': 'Which node to send to'}. If the critic says we need to retrieve more, we go back to the planner to generate new queries
        "drafter": "drafter", # If the critic says we need to rewrite, we go back to the drafter to fix the response
        END: END # If the critic approves, we end the workflow and return the draft as the final answer
    }
)

# Compile the Graph
app = workflow.compile()