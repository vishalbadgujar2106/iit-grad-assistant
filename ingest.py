"""
ingest.py
Reads every .md file in data/, splits it into overlapping chunks,
embeds each chunk locally (no API key needed for this step), and
stores everything in a persistent local ChromaDB collection.

Run this once at setup, and again any time you add/update files in data/.

Usage:
    python ingest.py
"""

import os
import glob
import re

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "iit_grad_assistant"

# Chunking config: word-based sliding window. Simple and effective for
# short, dense FAQ/policy docs like the ones in this project.
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40


def load_documents():
    """Read every markdown file in DATA_DIR, tagged with its source filename."""
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        docs.append({"source": source, "text": text})
    return docs


def split_into_sections(text):
    """
    Split a markdown doc on '## ' headers first, so each chunk stays
    topically coherent (e.g. 'Application Steps' doesn't get merged
    with 'Fees'). Falls back to the whole doc if there are no headers.
    """
    parts = re.split(r"\n(?=## )", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_section(section_text, max_words=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Word-based sliding window chunking, used if a section is still too long."""
    words = section_text.split()
    if len(words) <= max_words:
        return [section_text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunks(docs):
    """Turn loaded documents into a flat list of (id, text, metadata) chunks."""
    all_chunks = []
    for doc in docs:
        sections = split_into_sections(doc["text"])
        for sec_idx, section in enumerate(sections):
            # Grab the header text (first line) for metadata/debugging.
            header_line = section.split("\n", 1)[0].lstrip("#").strip()
            for chunk_idx, chunk_text in enumerate(chunk_section(section)):
                chunk_id = f"{doc['source']}::{sec_idx}::{chunk_idx}"
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": doc["source"],
                        "section": header_line or "intro",
                    },
                })
    return all_chunks


def main():
    print("Loading documents from data/ ...")
    docs = load_documents()
    if not docs:
        raise SystemExit(
            f"No .md files found in '{DATA_DIR}/'. Add your knowledge base "
            "documents there first."
        )
    print(f"  Loaded {len(docs)} document(s): {[d['source'] for d in docs]}")

    print("Chunking documents ...")
    chunks = build_chunks(docs)
    print(f"  Produced {len(chunks)} chunks.")

    print("Loading local embedding model (first run downloads ~90MB, cached after) ...")
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    print(f"Writing to local ChromaDB at ./{DB_DIR} ...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Fresh start each run so re-ingesting after editing data/ doesn't leave
    # stale chunks behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"Done. {collection.count()} chunks embedded and stored in '{COLLECTION_NAME}'.")
    print("You can now run: uvicorn app:app --reload")


if __name__ == "__main__":
    main()
