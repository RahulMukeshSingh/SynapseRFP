# SynapseRFP

**SynapseRFP** is an RAG-powered Request for Proposal (RFP) parsing and response assistant. It allows users to upload RFP documents, automatically extracts complex technical and security questions, and utilizes a robust Retrieval-Augmented Generation (RAG) agentic workflow to draft, critique, and finalize answers. 

Built with safety in mind, it integrates **NeMo Guardrails** to ensure secure and appropriate AI interactions, alongside **LangGraph** for advanced multi-step reasoning.

## Features

* **Automated RFP Extraction:** Upload `.pdf` or `.xlsx` documents and automatically extract technical and security questions using the `unstructured` library.
* **Agentic RAG Workflow:** Uses LangGraph to orchestrate a multi-actor system (Planner, Retriever, Drafter, Critic) to ensure high-quality, context-aware responses.
* **Security & Safety First:** Integrated with NVIDIA's **NeMo Guardrails** to filter inputs and block unsafe or off-topic prompts.
* **Real-time Streaming:** Streams LLM tokens and system thought-processes (e.g., *[System: Retrieving relevant documents...]* ) directly to the frontend.
* **Multi-LLM Support:** Seamlessly switch between OpenAI, Mistral, and Cohere using environment configurations.
* **Modern Frontend:** A responsive Next.js frontend built with React, Tailwind CSS, and Vercel's AI SDK.

## Tech Stack

**Backend:**
* **Framework:** Python, FastAPI
* **AI & Orchestration:** LangChain, LangGraph
* **Safety:** NeMo Guardrails
* **Document Parsing:** Unstructured
* **Vector Store:** Pinecone
* **State Management:** Redis
* **Evaluations** Ragas
* **Observability** LangSmith

**Frontend:**
* **Framework:** Next.js (App Router), React
* **Styling:** Tailwind CSS
* **AI UI:** `ai` (Vercel AI SDK), `react-markdown`

**Infrastructure:**
* Docker & Docker Compose

---

## Prerequisites

Before you begin, ensure you have the following installed:
* [Docker](https://www.docker.com/get-started) and Docker Compose
* API Keys for your preferred LLM provider (OpenAI, Mistral, or Cohere)
* A [Pinecone](https://www.pinecone.io/) API key and Index for the Vector Database

## Environment Variables

Create a `.env` file in the root directory (where `docker-compose.yml` is located). Use the following template and fill in your keys:

```env
# LLM Provider Configuration
LLM_PROVIDER=openai # Options: openai, mistral, cohere
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
MISTRAL_API_KEY=your_mistral_api_key
COHERE_API_KEY=your_cohere_api_key

# Vector Database (Pinecone)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rfp-responder

# Redis (leave as default for Docker deployment)
REDIS_URL=redis://redis:6379

# LangSmith Tracing (Optional - for debugging agent workflows)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=synapserfp
```

---

## How to Run (Recommended: Docker)

The easiest way to run the entire stack (Frontend, Backend, and Redis) is using Docker Compose.

1. **Clone the repository** and navigate to the project root.
2. **Ensure your `.env` file** is created and populated.
3. **Build and start the containers:**
   ```bash
   docker-compose up --build
   ```
4. **Access the application:**
   * **Frontend UI:** `http://localhost:3000`
   * **Backend API:** `http://localhost:8000`

To stop the application, run:
```bash
docker-compose down
```

---

## Local Development Setup (Without Docker)

If you wish to run the services locally for active development, you will need to run the Redis server, the Python backend, and the Next.js frontend separately.

### 1. Start Redis
Ensure you have a local Redis server running on port `6379`.

### 2. Backend Setup
```bash
cd backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Run the Next.js development server
npm run dev
```

---

## API Endpoints

Once the backend is running, the following core endpoints are available:

* **`POST /api/upload`**: Upload an RFP document (`.pdf` or `.xlsx`). The server parses the document and returns a JSON list of extracted technical/security questions.
* **`POST /api/chat`**: The main streaming chat endpoint. Accepts a list of messages, validates them against NeMo Guardrails, and streams back the LangGraph agent's thought process and the final LLM response.
* **`POST /api/chat/sync`**: A synchronous version of the chat endpoint that waits for the LangGraph workflow to complete and returns the final JSON payload.

---

## Project Structure

```text
SynapseRFP/
├── docker-compose.yml       # Orchestrates frontend, backend, and redis
├── backend/
│   ├── main.py              # FastAPI application & routing
│   ├── config.py            # Centralized environment configurations
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # Backend container instructions
│   ├── agents/              # LangGraph orchestration (graph, nodes, state)
│   ├── core/                # LLM factory and Vector store interactions
│   ├── guardrails/          # NeMo Guardrails configuration and prompts
│   └── ingestion/           # Document parsing logic
└── frontend/
    ├── package.json         # Node.js dependencies
    ├── Dockerfile           # Frontend container instructions
    ├── app/                 # Next.js App Router
    └── public/              # Static frontend assets
```