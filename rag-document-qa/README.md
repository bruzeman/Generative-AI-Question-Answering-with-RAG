# 📚 Enterprise Document QA

A private, open-source Retrieval-Augmented Generation (RAG) system for asking natural-language questions about internal PDF documents — built with **LangChain**, **Pinecone**, **HuggingFace** models, and **Streamlit**.

Answers are generated *strictly* from your own document set — no external knowledge is used, and the bot is instructed to say "I don't know" when the answer isn't in the retrieved context.

---

## Architecture

```
                ┌──────────────────┐
   PDFs  ──────▶│  ingestion.py     │──▶ chunks ──▶ HuggingFace embeddings ──▶ Pinecone index
 (data/raw)     └──────────────────┘

                ┌──────────────────┐
 User query ───▶│  rag_pipeline.py  │──▶ retriever (top-k) ──▶ HuggingFace Chat LLM ──▶ answer
                └──────────────────┘

                ┌──────────────────┐
                │     main.py       │  Streamlit chat UI, wraps rag_pipeline.py
                └──────────────────┘
```

| File | Role |
|---|---|
| `src/ingestion.py` | Loads PDFs, cleans + chunks text, embeds with HuggingFace, uploads to Pinecone |
| `src/rag_pipeline.py` | Builds the retrieval + generation chain used to answer questions |
| `main.py` | Streamlit chat interface — the app users actually interact with |

---

## How it works

### 1. Ingestion (`ingestion.py`)
- Loads every PDF in `data/raw/` via `PyPDFDirectoryLoader`.
- **Strips repeated header/footer boilerplate** (e.g. running section headers, standalone page numbers) before chunking, so this noise doesn't dominate small chunk embeddings.
- Splits documents with `RecursiveCharacterTextSplitter`:
  - `chunk_size=1200`, `chunk_overlap=200` — large enough to preserve real sentence/paragraph context, especially important for long reports (100+ pages) where narrow chunks fragment meaning.
- Embeds chunks using the free, local `all-MiniLM-L6-v2` HuggingFace sentence-transformer model (384-dim vectors).
- Creates (if needed) and populates a Pinecone serverless index (`rag-document-qa-hf`, cosine similarity, AWS `us-east-1`).

> ⚠️ **Re-running ingestion**: the script only *creates* the index if it doesn't already exist — running it again on an existing index **appends** new vectors rather than replacing old ones. If you change chunking parameters or source PDFs, delete the index first:
> ```python
> from pinecone import Pinecone
> pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
> pc.delete_index("rag-document-qa-hf")
> ```

### 2. Retrieval + Generation (`rag_pipeline.py`)
- Connects to the same Pinecone index using the same embedding model (must match ingestion exactly, or similarity search breaks).
- Retriever pulls the **top 8** most similar chunks (`search_kwargs={"k": 8}`) per query — tuned up from an initial `k=3`, which was too narrow for broad/summary-style questions on long documents.
- LLM: `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Inference Providers, wrapped in `ChatHuggingFace` (required — this model is chat-only, not raw text-generation).
  - `provider="auto"` — routes the request across **all** Inference Providers enabled on your HuggingFace account (Together, Fireworks, Cerebras, etc.) instead of defaulting to a single provider that may not host the model.
- A strict system prompt instructs the model to answer *only* from retrieved context and explicitly say "I don't know" otherwise — this is what prevents hallucinated answers when the document doesn't cover a topic.
- Built using LangChain's `create_stuff_documents_chain` + `create_retrieval_chain`.

### 3. Frontend (`main.py`)
- Simple Streamlit chat UI.
- Caches the RAG chain with `@st.cache_resource` so the model/vectorstore connection is only initialized once per session.
- Maintains chat history in `st.session_state`.

---

## Setup

### Prerequisites
- Python 3.12
- A [Pinecone](https://www.pinecone.io/) account and API key
- A [HuggingFace](https://huggingface.co/settings/tokens) account and **fine-grained token** with:
  - ✅ "Make calls to Inference Providers" permission enabled
- At least one Inference Provider enabled in your [HF provider settings](https://huggingface.co/settings/inference-providers/settings)

### Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

### Add your documents

Place PDF files in:

```
data/raw/
```

### Run ingestion (one-time, or whenever documents change)

```bash
python src/ingestion.py
```

### Launch the app

```bash
streamlit run main.py
```

---

## Example questions to test

**Narrow factual** (should work reliably)
- "Who is the CEO of [Company]?"
- "What was the net profit for FY[year]?"

**Section-specific** (tests whether retrieval finds the right passage)
- "What did the Chairman say in his report?"
- "What are the company's top and emerging risks?"

**Broad/summary** (hardest — see Known Limitations below)
- "Summarize the key financial results for the year."

**Out-of-scope** (should trigger the "I don't know" guardrail, not a hallucination)
- Ask about a fact or company not covered in the ingested documents.

---

## Known limitations

- **Whole-document summarization is inherently weak with plain top-k retrieval.** A query like "summarize the findings and conclusion" doesn't semantically match any single passage well, since no one chunk *is* the summary. Retrieval-based RAG can occasionally splice together numbers from different sections (e.g. a segment note vs. the consolidated group total) and present them as one coherent answer — always spot-check figures used in summary answers against the source document.
- For true whole-document summarization, a **map-reduce or refine summarization chain** (processing the document section-by-section rather than via similarity search) would be a more reliable approach than the current retrieval chain.
- HuggingFace Inference Providers change which models they host somewhat frequently. If you see a `model_not_supported` error, check the model's page on huggingface.co for its current "Inference Providers" status before assuming the code is broken.

---

## Tech stack

- [LangChain](https://python.langchain.com/) — orchestration
- [Pinecone](https://www.pinecone.io/) — vector database
- [HuggingFace](https://huggingface.co/) — embeddings (`all-MiniLM-L6-v2`) + LLM (`Qwen2.5-7B-Instruct`) via Inference Providers
- [Streamlit](https://streamlit.io/) — chat UI