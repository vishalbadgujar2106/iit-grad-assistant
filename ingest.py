"""
ingest.py
Reads every .md file in data/ and splits it into section-sized chunks that
the retriever in rag.py searches over.

There's no separate build step or vector database anymore: rag.py calls
build_chunks() once at startup and keeps the index in memory. The whole
knowledge base is only a few thousand words, so this takes milliseconds and
keeps the deploy small enough for serverless hosts like Vercel.

Run this directly to inspect how the docs get chunked:
    python ingest.py
"""

import glob
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Word-based sliding window, only used if a single section is unusually long.
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40


def load_documents(data_dir=DATA_DIR):
    """Read every markdown file in data_dir, tagged with its source filename."""
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": os.path.basename(path), "text": text})
    return docs


def split_into_sections(text):
    """
    Split a markdown doc on '## ' headers so each chunk stays topically
    coherent (e.g. 'Application Steps' doesn't get merged with 'Fees').
    """
    parts = re.split(r"\n(?=## )", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_section(section_text, max_words=CHUNK_SIZE_WORDS,
                  overlap=CHUNK_OVERLAP_WORDS):
    """Word-based sliding window chunking, used if a section is still too long."""
    words = section_text.split()
    if len(words) <= max_words:
        return [section_text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def parse_doc_header(intro):
    """Pull the title and 'Last verified' date out of a doc's intro block."""
    title = intro.split("\n", 1)[0].lstrip("#").strip()
    match = re.search(r"^Last verified:\s*([^\n]+)", intro, re.MULTILINE)
    verified = match.group(1).strip() if match else ""
    return title, verified


def build_chunks(docs):
    """
    Turn loaded documents into a flat list of chunks. The '# Title / Source /
    Last verified' intro of each doc is not a chunk of its own (it has no
    answerable content); instead its title is attached to every section so a
    chunk like 'Application Steps' still knows it is about CPT, not OPT.
    """
    all_chunks = []
    for doc in docs:
        sections = split_into_sections(doc["text"])
        title, verified = "", ""
        if sections and sections[0].startswith("# "):
            title, verified = parse_doc_header(sections[0])
            sections = sections[1:]

        for sec_idx, section in enumerate(sections):
            header_line = section.split("\n", 1)[0].lstrip("#").strip()
            for chunk_idx, chunk_text in enumerate(chunk_section(section)):
                all_chunks.append({
                    "id": f"{doc['source']}::{sec_idx}::{chunk_idx}",
                    "text": chunk_text,
                    "source": doc["source"],
                    "title": title,
                    "section": header_line,
                    "verified": verified,
                })
    return all_chunks


def main():
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No .md files found in '{DATA_DIR}'.")
    chunks = build_chunks(docs)
    print(f"{len(docs)} document(s), {len(chunks)} chunks:\n")
    for c in chunks:
        words = len(c["text"].split())
        print(f"  {c['id']:28} {words:4} words  {c['section']}")


if __name__ == "__main__":
    main()
