# backend/tests/eval.py
import os
import sys
import asyncio
import uuid
from typing import cast
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import answer_relevancy, faithfulness, context_precision
from langchain_core.runnables import RunnableConfig

# Ensure backend modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.agents.graph import app as graph_app
from backend.agents.state import GraphState

# 1. Define 20 Mock Questions & Ideal Answers (Ground Truths)
MOCK_RFP_DATA = [
    {
        "question": "How is customer data encrypted at rest?", 
        "ground_truth": "Customer data at rest is encrypted using AES-256 standard."
    },
    {
        "question": "Do you perform annual penetration testing?", 
        "ground_truth": "Yes, we engage a certified third-party firm for annual penetration testing."
    },
    {
        "question": "What is your data retention policy for deleted accounts?", 
        "ground_truth": "Data is retained for 30 days after account deletion before being permanently purged."
    },
    {
        "question": "Is your infrastructure SOC 2 Type II certified?", 
        "ground_truth": "Yes, our infrastructure is hosted on AWS, which maintains SOC 2 Type II compliance."
    },
    {
        "question": "How are employee access rights reviewed?", 
        "ground_truth": "Access rights are reviewed quarterly following the principle of least privilege."
    }
    # Add 15 more questions here to reach the requested 20...
]

async def run_evaluation():
    print(f"Starting evaluation of {len(MOCK_RFP_DATA)} questions...\n")
    
    ragas_dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    # 2. Run each question through your LangGraph Agent
    for item in MOCK_RFP_DATA:
        print(f"Testing: {item['question']}")
        
        config = cast(RunnableConfig, {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 5})
        initial_state = cast(GraphState, {
            "question": item["question"],
            "sub_tasks": [], "context": [], "retry_count": 0,
            "draft_response": "", "critic_feedback": "", "final_answer": ""
        })

        # Invoke the graph
        final_state = await graph_app.ainvoke(initial_state, config)
        
        # 3. Collect Data for Ragas
        ragas_dataset_dict["question"].append(item["question"])
        ragas_dataset_dict["answer"].append(final_state.get("final_answer", ""))
        # Ragas expects contexts as a list of strings
        ragas_dataset_dict["contexts"].append(final_state.get("context", [])) 
        ragas_dataset_dict["ground_truth"].append(item["ground_truth"])

    # 4. Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(ragas_dataset_dict)

    # 5. Run Ragas Evaluation
    print("\nRunning Ragas Evaluation Metrics...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            answer_relevancy,  # Does the answer actually address the question?
            faithfulness,      # Are there hallucinations? (Critic agent should make this score high)
            context_precision  # Did the Retriever pull the right chunks?
        ] # type: ignore
    )

    print("\n=== FINAL EVALUATION SCORES ===")
    print(results)
    
    # Optionally save to file
    results.to_pandas().to_csv("ragas_evaluation_results.csv", index=False) # type: ignore
    print("Detailed results saved to ragas_evaluation_results.csv")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
    