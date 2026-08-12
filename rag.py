"""
rag.py
Core retrieval-augmented generation logic:
  1. Embed the user's question
  2. Retrieve the most relevant chunks from ChromaDB
  3. Build a grounded prompt
  4. Call the LLM (Anthropic Claude by default) and return the answer
     plus the sources used, so the frontend can show "answered from: ..."
"""

import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "chroma_db"
COLLECTION_NAME = "iit_grad_assistant"
TOP_K = 4  # how many chunks to retrieve per question

SYSTEM_PROMPT = """You are the International Grad Student Assistant for Illinois \
Institute of Technology (Illinois Tech). You answer questions from prospective \
and current international graduate students about OPT, CPT, registration, \
full-time status, and financial aid.

Rules you must follow:
1. Answer ONLY using the CONTEXT provided below. Do not use outside knowledge, \
   even if you think you know the answer.
2. If the context does not contain the answer, say clearly: "I don't have \
   verified information on that — please check with the Office of Global \
   Services (OGS) or the Office of the Registrar directly." Do not guess.
3. This tool provides general information only. It is NOT official legal or \
   immigration advice. For anything involving visa status, deadlines that \
   affect your status, or anything you are unsure about, always tell the \
   student to confirm with OGS before acting.
4. Keep answers concise and directly useful — students are often checking \
   this between classes.
5. When a rule depends on the specific term/semester (like a CPT deadline), \
   remind the student that exact dates should be re-verified on the official \
   OGS page, since they change every semester.
"""


def get_collection():
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedder)


def retrieve(question: str, top_k: int = TOP_K):
    """Return the top_k most relevant chunks for the question."""
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=top_k)

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for text, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "section": meta.get("section", ""),
            "score": 1 - dist,  # cosine distance -> similarity, higher is better
        })
    return chunks


def build_prompt(question: str, chunks: list):
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {c['source']} — {c['section']}]\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    return f"""CONTEXT:
{context}

STUDENT QUESTION: {question}

Answer the student's question using only the CONTEXT above, following your rules."""


def generate_answer(question: str, chunks: list) -> str:
    """
    Calls Claude with the grounded prompt. Requires ANTHROPIC_API_KEY to be
    set in the environment (see .env.example).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(question, chunks)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def answer_question(question: str, top_k: int = TOP_K):
    """
    Full pipeline used by the API: retrieve -> generate -> return answer +
    sources, so the frontend can display what the answer was grounded in.
    """
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return {
            "answer": "I don't have any information loaded yet — has the "
                      "knowledge base been ingested? Run ingest.py first.",
            "sources": [],
        }

    answer = generate_answer(question, chunks)
    sources = sorted({c["source"] for c in chunks})
    return {"answer": answer, "sources": sources, "chunks_used": chunks}


if __name__ == "__main__":
    # Quick manual test: python rag.py "your question here"
    import sys

    q = " ".join(sys.argv[1:]) or "How many credits do I need to stay full-time?"
    result = answer_question(q)
    print("\nQ:", q)
    print("\nA:", result["answer"])
    print("\nSources:", result["sources"])
