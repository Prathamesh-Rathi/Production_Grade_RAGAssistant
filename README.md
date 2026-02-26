## RAG GenAI Assistant – README

### Architecture Overview

This project is a **production-style GenAI chat assistant** built with:

- Backend: Flask (Python)
- Frontend: HTML/CSS/JS (single-page chat UI)
- LLM: Groq Chat API (Llama‑3 family)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector Store: FAISS (in-memory, cosine similarity)
- Storage: `docs.json` knowledge base

High-level architecture (text diagram):

```text
User (Browser)
    │
    ▼
Beautiful Chat UI (index.html)
    │  POST /api/chat  (message, sessionId)
    ▼
Flask Backend (app.py)
    │
    ├─ Conversation Manager (per session history)
    │
    ├─ RAG Retrieval Layer
    │     • docs.json → chunk_text()
    │     • all-MiniLM-L6-v2 embeddings
    │     • FAISS IndexFlatIP + cosine similarity
    │
    ├─ Prompt Builder
    │     • Injects: retrieved chunks + chat history + guardrails
    │
    └─ Groq LLM (chat.completions)
          • model: e.g. llama-3.3-70b-versatile
          • temperature: 0.2
          • max_tokens: 300
```

This follows the standard RAG pattern: **embed at build time, retrieve + augment at runtime.** [squirro](https://squirro.com/squirro-blog/rag-architecture)

***

### RAG Workflow

End-to-end RAG flow for each user message: [latenode](https://latenode.com/blog/ai-frameworks-technical-infrastructure/rag-retrieval-augmented-generation/rag-diagram-guide-visual-architecture-of-retrieval-augmented-generation)

1. **User Message**
   - Frontend sends `{ message, sessionId }` to `POST /api/chat`.

2. **Conversation History Lookup**
   - Backend loads the last ~5 (user, assistant) pairs for that `sessionId` to maintain context (pronouns, follow‑ups).

3. **Retrieval (Semantic Search)**
   - The query is embedded with `all-MiniLM-L6-v2`.
   - FAISS performs cosine similarity search against all document chunks.
   - Top‑k (3) chunks above a similarity threshold are selected as context. [nanonets](https://nanonets.com/blog/retrieval-augmented-generation-workflows/)

4. **Fallback Decision**
   - If no chunk passes the threshold, the system returns a safe response:
     > "I don't have enough information in the knowledge base to answer that…"

5. **Augmented Prompt Construction**
   - Retrieved chunks are formatted into a **Context** section.
   - Conversation history is formatted into a **Chat History** section.
   - A **System Prompt** gives strict rules about using only the provided context and how to behave. [apxml](https://apxml.com/courses/prompt-engineering-llm-application-development/chapter-6-integrating-llms-external-data-rag/combining-retrieved-context-prompts)

6. **Generation**
   - The augmented prompt + user question are sent to Groq’s chat completion model.
   - The LLM generates a grounded answer.

7. **Response + Logging**
   - Backend returns:
     - `reply` (assistant message)
     - `retrieved` (number of chunks used)
     - `tokensUsed` (LLM usage)
   - Frontend displays the message and stats in the UI.

***

### Embedding Strategy

We use `sentence-transformers/all-MiniLM-L6-v2` as the embedding model: [blog.csdn](https://blog.csdn.net/gitblog_00929/article/details/150532722)

- **Type:** Sentence-level transformer model.
- **Dimension:** 384‑dimensional embeddings, compact but semantically rich.
- **Advantages:**
  - Lightweight (~90MB), fast enough for real‑time RAG on CPU. [blog.csdn](https://blog.csdn.net/gitblog_00929/article/details/150532722)
  - Good performance on semantic similarity tasks (question–answer, FAQ, docs). [huggingface](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Usage in this project:**
  - Each document (or chunk) in `docs.json` is encoded into a 384‑dim vector at startup.
  - User queries are encoded in the same space at runtime so they’re comparable.

This choice keeps **cost at zero** (no paid embedding API) while providing decent semantic search quality for a small internal KB. [huggingface](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

***

### Similarity Search Explanation

We use **FAISS IndexFlatIP** + **cosine similarity** for vector search:

1. **Indexing:**
   - All document embeddings are L2‑normalized.
   - FAISS `IndexFlatIP` (inner product) is used.
   - After normalization, inner product between vectors equals cosine similarity. [escape-force](https://www.escape-force.com/post/understanding-cosine-similarity-and-its-role-in-llm-models-with-retrieval-augmented-generation-rag)

2. **Querying:**
   - The query embedding is also L2‑normalized.
   - We search for top‑k nearest neighbors by **maximum inner product** (cosine similarity). [docs.raga](https://docs.raga.ai/ragaai-catalyst/ragaai-metric-library/additional-metrics/evaluation/cosine-similarity)

3. **Thresholding:**
   - Only chunks with similarity ≥ 0.3 are considered valid.
   - This filters out weak matches and reduces noise while still being tolerant enough for a small corpus. [milvus](https://milvus.io/ai-quick-reference/how-do-you-tune-similarity-thresholds-to-reduce-false-positives)

4. **Why cosine similarity:**
   - Scale‑invariant: focuses on direction (semantics), not magnitude.
   - Standard metric for embedding‑based RAG retrieval. [escape-force](https://www.escape-force.com/post/understanding-cosine-similarity-and-its-role-in-llm-models-with-retrieval-augmented-generation-rag)

***

### Prompt Design Reasoning

The prompt is designed to **strictly ground the LLM** in retrieved context and handle multi‑turn chat: [community.openai](https://community.openai.com/t/prompt-engineering-for-rag/621495)

1. **System Prompt Sections**
   - **Role:** “You are a helpful support assistant for a SaaS product.”
   - **Rules:**
     - Use **only** the context for factual answers.
     - Use **chat history only** for continuity (follow‑ups, pronouns).
     - If answer not in context, explicitly say:
       > "I don't have information about that in the knowledge base."
   - **Guidelines:** Be concise, do not invent features/prices/policies, reference document titles when possible.

2. **Context Block**
   - Retrieved chunks are formatted like:
     ```text
      [geeksforgeeks](https://www.geeksforgeeks.org/nlp/rag-architecture/) Password Reset:
     <chunk content>

      [latenode](https://latenode.com/blog/ai-frameworks-technical-infrastructure/rag-retrieval-augmented-generation/rag-diagram-guide-visual-architecture-of-retrieval-augmented-generation) API Rate Limits:
     <chunk content>
     ```
   - This gives the model **clean, labeled passages** to reason over. [apxml](https://apxml.com/courses/prompt-engineering-llm-application-development/chapter-6-integrating-llms-external-data-rag/combining-retrieved-context-prompts)

3. **Chat History Block**
   - Last few turns are included as:
     ```text
     User: ...
     Assistant: ...
     ```
   - Helps resolve pronouns (“it”, “that plan”) without mixing chat history and knowledge base. [smith.langchain](https://smith.langchain.com/hub/teddynote/rag-prompt-chat-history)

4. **User Message**
   - Sent as the final `user` role message, which keeps the model focused on the current question while still seeing context + history.

This structure follows recommended RAG prompt patterns (system instructions + formatted context + question), improving grounding and reducing hallucinations. [ibm](https://www.ibm.com/architectures/patterns/genai-rag)

***

### Setup Instructions

1. **Clone and create project**

   ```bash
   git clone <your-repo-url>
   cd rag-assistant
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add Groq API Key**

   In `app.py`, set:

   ```python
   GROQ_API_KEY = "gsk_YourRealGroqKeyHere"
   ```

   - Generate a key from Groq Console → API Keys. [console.groq](https://console.groq.com/docs/models)
   - Ensure the chosen `model` (e.g. `llama-3.3-70b-versatile`) is available in your account. [console.groq](https://console.groq.com/docs/models)

4. **Verify files**

   - `app.py` – Flask backend + RAG + Groq integration
   - `templates/index.html` – Chat UI
   - `docs.json` – 5–10 domain documents for knowledge base
   - `requirements.txt` – Python dependencies

5. **Run the app**

   ```bash
   python app.py
   ```

   Open in browser:

   ```text
   http://localhost:5000
   ```

6. **Test queries**

   Try questions grounded in docs:

   - “How can I reset my password?”
   - “What are the subscription plans?”
   - “What are the API rate limits?”
   - “What is the support email?”
   - Follow‑up: “And what is its response time?”

   Also test out‑of‑scope:

   - “Who is your CEO?”
   - Expect the fallback message (insufficient knowledge base info).
  
     

https://github.com/user-attachments/assets/6cbf161c-ef4e-4af1-848c-da0a537df692


