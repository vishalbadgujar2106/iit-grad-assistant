# International Grad Student Assistant — Illinois Tech

A RAG chatbot that answers international grad students' questions about OPT, CPT, registration, full-time status, and financial aid at Illinois Tech. I built this after spending way too much time digging through scattered FAQ pages for answers I needed as an international student myself — figured a lot of other students hit the same wall every semester.

Built by [Vishal K Badgujar](https://www.linkedin.com/in/vishal-k-badgujar-b8115a200/) — [Portfolio](https://vishal-portfolio-ten-sooty.vercel.app/)

## A note before anything else

This is not official immigration advice, and it shouldn't be treated that way. It's a tool for general orientation — the kind of thing you'd ask a senior student, not an immigration lawyer. Anything touching your actual visa status or a real deadline should go through the Office of Global Services (OGS) directly. The bot is instructed to say "I don't know" rather than guess, and to point status-related questions back to OGS, but I wouldn't want anyone treating an AI's answer as the final word on something that could affect their status. The disclaimer shows up in the chat UI itself for this reason, not just buried down here.

## How it's put together

There are two halves to this. Offline, the documents in `data/` get split into chunks, turned into embeddings, and stored in a local ChromaDB. At runtime, every question gets embedded too, matched against those stored chunks to pull the most relevant ones, and then handed to Claude along with the question so it can answer using the actual retrieved text instead of guessing from memory.

```
Offline (run once, re-run when data/ changes):
  data/*.md  →  chunk by section  →  embed (local model)  →  store in ChromaDB

Runtime (every question):
  User question  →  embed  →  vector search (top 4 chunks)
                 →  build grounded prompt  →  Claude API  →  answer + sources
```

The system prompt in `rag.py` is what enforces the "only answer from context, admit when you don't know" behavior — worth reading if you're curious how that's actually implemented rather than just claimed.

## What's in here

```
iit-grad-assistant/
├── data/                   # The knowledge base — 5 topic docs, written from
│                            # real IIT pages (sources listed further down)
├── ingest.py               # Chunks + embeds + stores data/ into ChromaDB
├── rag.py                  # Retrieval + prompt construction + Claude call
├── app.py                  # FastAPI backend (/chat endpoint)
├── frontend/index.html     # Standalone chat UI, no build step
├── eval/
│   ├── test_questions.json # 10 test Q&A pairs, including one deliberately
│   │                        # off-topic question to check the bot refuses
│   │                        # instead of making something up
│   └── evaluate.py         # Scores answers with RAGAS
├── .github/workflows/eval.yml  # Runs the eval suite on every push
├── Dockerfile
└── requirements.txt
```

## Getting it running

Install the dependencies:
```bash
pip install -r requirements.txt
```

Copy the env file and drop in your own Anthropic API key (get one at console.anthropic.com):
```bash
cp .env.example .env
```

Build the vector store:
```bash
python ingest.py
```
First run pulls down a small local embedding model, maybe 90MB, then it's cached for next time. Re-run this whenever you edit anything in `data/`.

Start the backend:
```bash
uvicorn app:app --reload
```
That runs at `http://localhost:8000` — `/health` should just return `{"status": "ok"}` if it's up.

Then just open `frontend/index.html` in a browser. No build tooling, no npm install, it's plain HTML/JS.

## Checking that it's actually accurate

```bash
python eval/evaluate.py
```

This runs all 10 test questions through the real pipeline and scores each answer on faithfulness (did it actually stick to the retrieved docs, or wander off?) and relevancy (did it answer what was asked?). If either score drops below the threshold set in the script, it exits with a failing status code — that's what `.github/workflows/eval.yml` hooks into so a bad change can't get merged without someone noticing.

I added this mainly because "it seemed to work when I tried it" isn't something I wanted to just take on faith for something touching visa-related info.

## Deploying it somewhere real

```bash
docker build -t iit-grad-assistant .
docker run -p 8000:8000 --env-file .env iit-grad-assistant
```
Push that image to ECR and run it on Fargate or Lambda if you want it actually live, and update `API_URL` in `index.html` to point at wherever the backend ends up.

## Where the data came from, and why it'll go stale

The five files in `data/` are written from Illinois Tech's own OGS, Registrar, and Financial Aid pages, current as of August 2026. University deadlines and fees shift every semester, so treat this as a starting point, not something to trust blindly a year from now. Each file has a "last verified" date at the top for exactly this reason — check the source pages before relying on anything time-sensitive, and re-run `ingest.py` after updating.

Sources:
- [OPT — Office of Global Services](https://www.iit.edu/global-services/student-employment/optional-practical-training-opt)
- [CPT — Office of Global Services](https://www.iit.edu/global-services/student-employment/curricular-practical-training-cpt)
- [Registration & Full-Time Status FAQs](https://www.iit.edu/gaa/students/faqs)
- [Financial Aid — International Students](https://www.iit.edu/financial-aid/international-students)

## Ideas for taking this further

A few things I'd add if I kept building on this: actual tracing (LangSmith or similar) so I could see what's happening per-query instead of flying blind; some way to auto-refresh the knowledge base instead of manually editing markdown every semester; conversation memory so follow-up questions don't need full context repeated; and probably swapping ChromaDB for pgvector if this ever needed to live alongside a real Postgres app.

## Stack

Python, FastAPI, ChromaDB, Sentence-Transformers, Claude API, RAGAS, Docker, GitHub Actions.
