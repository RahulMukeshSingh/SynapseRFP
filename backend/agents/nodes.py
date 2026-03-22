# backend/agents/nodes.py
# ─────────────────────────────────────────────────────────────
# SynapseRFP — Agent Nodes
# ─────────────────────────────────────────────────────────────

import logging
from typing import List, cast, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from backend.agents.state import GraphState
from backend.config import config
from backend.core.storage import get_retriever
from backend.agents.prompts import PLANNER_PROMPT, DRAFTER_PROMPT, CRITIC_PROMPT
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever # Now langchain_classic.retrievers is used insted of langchain.retrievers
from backend.core.llm import get_llm


logger = logging.getLogger(__name__)


# --- Structured Output Schemas ---

class SearchQueries(BaseModel):
    """Planner output schema"""
    queries: List[str] = Field(description="List of 1-3 specific search queries for the vector DB")

class Evaluation(BaseModel):
    """Critic output schema"""
    decision: Literal["pass", "rewrite", "retrieve_more"] = Field(
        description="Must be one of: 'pass', 'rewrite', or 'retrieve_more'"
    )
    feedback: str = Field(description="Specific reason for the decision and instructions for improvement")

# --- Agent Nodes ---

def planner_node(state: GraphState):
    """
    Analyzes the question and breaks it into search queries.
    """
    logger.info("--- PLANNER AGENT STARTING ---")
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(SearchQueries)
    
    prompt = PLANNER_PROMPT.format(question=state['question'])
    
    result = structured_llm.invoke(prompt)
    structured_result = cast(SearchQueries, result)

    return {"sub_tasks": structured_result.queries}


def retriever_node(state: GraphState):
    """
    Takes the sub_tasks and pulls parent documents from Redis.
    """
    logger.info("--- RETRIEVER AGENT STARTING ---")
    
    retriever = get_retriever(
        openai_api_key=config.openai_api_key,
        pinecone_api_key=config.pinecone_api_key,
        pinecone_index_name=config.pinecone_index_name,
        redis_url=config.redis_url
    )

    # Reranker compressor to ensure we only keep the most relevant context (Top 5) out of potentially 20+ retrieved chunks
    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5, cohere_api_key=config.cohere_api_key) # type: ignore
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever
    )


    # We only search using the NEWEST queries generated in the last turn
    new_queries = state["sub_tasks"][-3:] if state["sub_tasks"] else []
    
    unique_contents = set()
    for query in new_queries:
        logger.info(f"Hybrid Searching + Reranking for: {query}")
        docs = compression_retriever.invoke(query) #returns Parent Document
        for d in docs:
            unique_contents.add(d.page_content)
    
    # Convert set back to list for the state
    return {"context": list(unique_contents)}


def drafter_node(state: GraphState):
    """
    Writes a technical response based on the retrieved context.
    """
    logger.info("--- DRAFTER AGENT STARTING ---")
    
    llm = get_llm()
    
    # We join all the accumulated context into one big block for the LLM
    unique_context = list(dict.fromkeys(state["context"])) # dict.fromkeys() insetad of set() to remove duplicates while preserving order
    context_block = "\n\n".join(unique_context[-10:])  # Keep most recent 10 chunks to avoid overwhelming the LLM context issue.
    
    prompt = DRAFTER_PROMPT.format(
        question=state['question'], 
        context=context_block
    )
    
    response = llm.invoke(prompt)
    
    # We save this as a draft. It's not final until the Critic approves!
    return {"draft_response": response.content}

def critic_node(state: GraphState):
    """
    Fact-checks the draft against the context to prevent hallucinations.
    """
    logger.info("--- CRITIC AGENT STARTING ---")
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(Evaluation)
    

    unique_context = list(dict.fromkeys(state["context"])) # dict.fromkeys() insetad of set() to remove duplicates while preserving order
    context_block = "\n\n".join(unique_context[-10:])  # Keep most recent 10 chunks to avoid overwhelming the LLM context issue.
    prompt = CRITIC_PROMPT.format(
        draft=state['draft_response'], 
        context=context_block
    )
    
    result = structured_llm.invoke(prompt)
    structured_result = cast(Evaluation, result)
    # We update the feedback and increment the retry count
    return {
        "critic_feedback": f"[{structured_result.decision.upper()}] {structured_result.feedback}",
        "retry_count": state.get("retry_count", 0) + 1
    }

def finalize_node(state: GraphState):
    """
    Copies the approved draft_response into final_answer.
    """
    logger.info("--- FINALIZE AGENT STARTING ---")
    
    # We copy the draft to the final answer. 
    # If the graph timed out/exceeded retries, we still provide whatever draft we had.
    draft = state.get("draft_response", "No draft could be generated.")
    
    return {"final_answer": draft}