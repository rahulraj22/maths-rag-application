# ✦ Maths RAG application

> A conversational AI tutor that lets you ask questions around Maths chapters (e.g. linear transformations and linear algebra) - powered by your own course notes, not just the model's memory.

---

## ✦ What is this?

Point any math PDF at it - linear algebra, calculus, real analysis, probability, whatever you're studying - and it becomes a conversational tutor for that material. Ask follow-up questions, request proofs, ask for examples. It remembers what you discussed earlier in the conversation.

The key difference from just asking ChatGPT is that every answer is grounded in your actual course notes. The model only tells you what's in the document, and says so clearly if something isn't covered there.

`Note`: Currently its knowledge corpus is limited to `lintransf.pdf`, later we will allow pdf upload and on the go vector embeddings and user's can ask any questions out of it.

---

## ✦ How it works

The pipeline has a few moving parts that work together:

```
PDF (lintransf.pdf)
    └── Docling parser (structure-aware, pypdfium2 backend)
            └── HybridChunker (respects section boundaries)
                    └── ChromaDB (local vector store, persisted to disk)
                            └── Similarity search (with optional section filter)
                                    └── gpt-4o-mini (streamed response)
                                            └── Streamlit chat UI (KaTeX math rendering)
```

**Parsing** — [Docling](https://github.com/docling-project/docling) reads the PDF with layout awareness. It understands headings, paragraphs, formula blocks, and list items as distinct structural elements — not just a wall of text.

**Chunking** — Docling's `HybridChunker` splits the content along natural section boundaries first, then merges or splits further based on token count. This means a theorem and its proof tend to stay in the same chunk, rather than getting cut in half.

**Embedding & Storage** — Each chunk is embedded using OpenAI's `text-embedding-3-small` model and stored in a local [Chroma](https://www.trychroma.com/) vector database. This is persisted to `chroma_db/` on disk, so it only needs to be built once.

**Retrieval** — When you ask a question, the top 5 most semantically similar chunks are fetched from Chroma. You can optionally filter by section heading so retrieval is scoped to just the parts of the notes you care about.

**Generation** — The retrieved context, your current question, and the full conversation history so far are all sent to `gpt-4o-mini`. Responses are streamed token by token. Math is rendered live using KaTeX.

---

## ✦ Features

- **Conversational memory** - follow-up questions like "can you show the proof?" work because the model sees the full prior conversation
- **Streamed responses** - answers appear word by word; a *Thinking...* indicator shows while context is being retrieved
- **Proper math rendering** - all LaTeX expressions render correctly in the browser via KaTeX (`$...$` inline, `$$...$$` block)
- **Section filtering** - sidebar lets you restrict retrieval to specific sections of the notes (e.g. only search within "4.2 Examples" or "4.1 Linear transformations")
- **Cached parsing** - the PDF is parsed once and saved to `docs_cache.pkl`; subsequent runs skip parsing entirely and load in seconds
- **Clear history** - one-click button in the sidebar to reset the conversation

---

## ✦ Project structure

```
rag_maths/
├── app.py              # Streamlit chat UI
├── rag.py              # Backend: parsing, embedding, retrieval, LLM chain
├── script.ipynb        # Exploration notebook (development scratchpad)
├── lintransf.pdf       # Source PDF — linear transformations lecture notes
├── docs_cache.pkl      # Cached parsed chunks (auto-generated on first run)
├── chroma_db/          # Persisted vector store (auto-generated on first run)
└── .env                # OPENAI_API_KEY goes here
```

---

## ✦ Getting started

**1. Clone and install dependencies**

```bash
cd projects/rag_maths
pip install streamlit langchain-openai langchain-chroma langchain-docling langchain-core python-dotenv docling
```

**2. Add your OpenAI API key**

Create a `.env` file in `rag_maths/`:

```
OPENAI_API_KEY=sk-...
```

**3. Run the app**

```bash
streamlit run app.py
```

On the first run, the app will parse the PDF and build the vector store — this takes a minute or two. After that, everything loads from cache and starts instantly.

---

## ✦ Tech stack

| Layer | Tool |
|---|---|
| PDF parsing | [Docling](https://github.com/docling-project/docling) + pypdfium2 backend |
| Chunking | Docling `HybridChunker` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | [ChromaDB](https://www.trychroma.com/) (local, persisted) |
| LLM | OpenAI `gpt-4o-mini` (streamed) |
| Orchestration | [LangChain](https://www.langchain.com/) LCEL |
| UI | [Streamlit](https://streamlit.io/) with KaTeX math rendering |

---

## ✦ Notes & limitations

- **Formula placeholders** — because formula enrichment is disabled for speed (it requires a heavy transformer model running on CPU), some formulas in the source PDF appear as `<!-- formula-not-decoded -->` in the retrieved context. The LLM handles this gracefully but cannot reconstruct the exact original formula. Re-enabling `do_formula_enrichment=True` in `rag.py` fixes this at the cost of significantly longer parsing time.

## ✦ Future Feature Addition
- **Tool Call** — Right now the LLM can only answer from retrieved context. Giving it tools would open up a lot more. For example, a `solve_equation` tool that calls a symbolic math library like SymPy, a `plot_transformation` tool that visualises how a matrix transformation distorts a vector space, or a `verify_proof_step` tool that checks whether a logical step holds. The model could decide on its own when to call these mid-conversation — so instead of just explaining what an eigenvector is, it could actually compute one for a matrix the user provides.

- **MCP Server Connection** — Connecting the app to an [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server would let it tap into external knowledge sources beyond the PDF at runtime — things like a Wolfram Alpha server for symbolic computation, a Wikipedia server for background definitions, or a custom notes server that a student maintains themselves. The LLM would treat these as additional context providers, pulling from whichever source is most relevant to the question without the user having to switch between tools manually.
