from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from groq import Groq
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer  # local embeddings[web:23][web:26]

app = Flask(__name__)
CORS(app)

# ===== Groq Setup =====
# Put your real Groq key here (demo only, not secure for production)[web:4][web:11]
GROQ_API_KEY = "your_Groq_key"

groq_client = Groq(api_key=GROQ_API_KEY)

# ===== Embedding + Vector Store (RAG) =====

class EmbeddingEngine:
    def __init__(self):
        # all-MiniLM-L6-v2: 384-dim sentence embeddings, good for semantic search[web:23][web:9]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.build_index()

    def chunk_text(self, text, max_words=80):
        """
        Simple word-based chunking.
        ~80 words is roughly 300–400 tokens depending on language.[web:46]
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        return chunks

    def build_index(self):
        # Load documents from docs.json
        with open('docs.json', 'r', encoding='utf-8') as f:
            docs = json.load(f)

        texts = []
        meta = []

        for doc in docs:
            # Chunk each document
            chunks = self.chunk_text(doc["content"])
            for chunk in chunks:
                texts.append(chunk)
                meta.append({
                    "title": doc["title"],
                    "content": chunk
                })

        # Compute embeddings
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')

        # FAISS index using cosine similarity: L2 normalize + inner product[web:7][web:29][web:45]
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.chunks = meta

    def search(self, query, top_k=3, threshold=0.3):
        """
        Semantic search with cosine similarity threshold.
        Only return chunks with score >= threshold.[web:50][web:45]
        """
        query_emb = self.model.encode([query])
        query_emb = np.array(query_emb).astype('float32')
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)
        print("Scores:", scores[0])  # debug

        results = []
        for score, idx in zip(scores[0], indices[0]):
            score = float(score)
            if score < threshold:
                continue
            results.append({
                "title": self.chunks[idx]["title"],
                "content": self.chunks[idx]["content"],
                "score": score
            })
        return results

# Initialize embedding engine once at startup
embedding_engine = EmbeddingEngine()

# In-memory conversation history: {sessionId: [(user, assistant), ...]}
conversations = {}

# ===== Routes =====

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json or {}
        message = data.get('message', '').strip()
        session_id = data.get('sessionId', 'default_session')

        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        # Initialize session history if missing
        if session_id not in conversations:
            conversations[session_id] = []

        # STEP 1: Retrieve relevant chunks from vector store
        retrieved = embedding_engine.search(
            message,
            top_k=3,
            threshold=0.3
        )
        print("Retrieved:", retrieved)  # debug

        # STEP 2: If no chunks are similar enough, return safe fallback[web:50][web:49]
        if not retrieved:
            safe_reply = (
                "I don't have enough information in the knowledge base to answer that. "
                "Please check the official documentation or contact support."
            )
            conversations[session_id].append((message, safe_reply))
            if len(conversations[session_id]) > 5:
                conversations[session_id] = conversations[session_id][-5:]
            return jsonify({
                'reply': safe_reply,
                'retrieved': 0,
                'tokensUsed': 0,
                'topTitle': None
            })

        # STEP 3: Build context string from retrieved chunks
        context_blocks = []
        for i, chunk in enumerate(retrieved, start=1):
            context_blocks.append(
                f"[{i}] {chunk['title']}:\n{chunk['content']}"
            )
        context_text = "\n\n".join(context_blocks)

        # STEP 4: Build chat history text (last 3–5 exchanges)[web:64][web:66]
        history_pairs = conversations[session_id][-5:]
        history_lines = []
        for user_msg, assistant_msg in history_pairs:
            history_lines.append(f"User: {user_msg}")
            history_lines.append(f"Assistant: {assistant_msg}")
        history_text = "\n".join(history_lines) if history_lines else "None."

        # STEP 5: Construct grounded system prompt including history[web:59][web:63]
        system_prompt = f"""You are a helpful support assistant for a SaaS product.

You must follow these rules:
- Use ONLY the information from the context to answer factual questions.
- Use the chat history only to maintain conversation continuity (pronouns, follow-ups, etc.).
- If the answer is not clearly in the context, say:
  "I don't have information about that in the knowledge base."

Context:
{context_text}

Chat History (most recent first):
{history_text}

Guidelines:
- Be concise and clear.
- Do not invent features, prices, or policies not mentioned in the context.
- When possible, mention which document you used, like "According to Password Reset"."""  # noqa: E501

        # STEP 6: Call Groq LLM with context + history[web:3][web:16][web:63]
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ensure valid model ID for your Groq account[web:16]
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.2,
            max_tokens=300
        )

        reply_text = completion.choices[0].message.content.strip()
        total_tokens = completion.usage.total_tokens if completion.usage else 0
        top_title = retrieved[0]["title"] if retrieved else None

        # STEP 7: Store this turn in history and keep sliding window[web:64]
        conversations[session_id].append((message, reply_text))
        if len(conversations[session_id]) > 5:
            conversations[session_id] = conversations[session_id][-5:]

        return jsonify({
            'reply': reply_text,
            'retrieved': len(retrieved),
            'tokensUsed': total_tokens,
            'topTitle': top_title
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
