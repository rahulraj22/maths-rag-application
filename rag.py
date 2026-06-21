import os
import pickle

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_docling import DoclingLoader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "lintransf.pdf")
CACHE_FILE = os.path.join(BASE_DIR, "docs_cache.pkl")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

SYSTEM_PROMPT = """You are a helpful math tutor specializing in linear algebra and linear transformations.
Answer the question using only the provided context from the course notes.
Use the conversation history to understand follow-up questions and references to previous answers.

CRITICAL — Math formatting rules (strictly follow):
- ALWAYS use $...$ for inline math. Example: $T(x+y) = T(x) + T(y)$
- ALWAYS use $$...$$ on its own line for block/display equations. Example: $$\\ker(T) = \\{{v \\in V : T(v) = 0\\}}$$
- NEVER use \\(...\\) or \\[...\\] delimiters — they will not render.
- Every math symbol, variable, or expression — no matter how short — must be wrapped in $...$

If the context does not contain enough information to answer, say so clearly.

Context from course notes:
{context}"""


def _build_loader():
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_formula_enrichment=False,
        do_code_enrichment=False,
        generate_page_images=False,
        generate_picture_images=False,
    )
    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )
    return DoclingLoader(file_path=FILE_PATH, converter=converter)


def _load_docs():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    loader = _build_loader()
    docs = loader.load()
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(docs, f)
    return docs


def _clean_metadata(docs):
    for doc in docs:
        dl_meta = doc.metadata.get("dl_meta", {})
        headings = dl_meta.get("headings", [])
        doc.metadata = {
            "source": doc.metadata.get("source", ""),
            "heading": headings[0] if headings else "",
        }
    return docs


def _build_vectorstore(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(FAISS_INDEX_DIR):
        return FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    clean_docs = _clean_metadata(docs)
    vectorstore = FAISS.from_documents(clean_docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)
    return vectorstore


def build_vectorstore():
    docs = _load_docs()
    return _build_vectorstore(docs)


def get_all_headings(vectorstore: FAISS) -> list[str]:
    """Return sorted list of unique section headings stored in the vector index."""
    docs = vectorstore.docstore._dict.values()
    headings = sorted(
        {doc.metadata.get("heading", "") for doc in docs if doc.metadata.get("heading")}
    )
    return headings


def retrieve(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
    headings: list[str] | None = None,
) -> str:
    """Run similarity search with an optional heading filter, return formatted context."""
    filter_fn = (lambda meta: meta.get("heading") in headings) if headings else None
    docs = vectorstore.similarity_search(query, k=k, filter=filter_fn)
    return "\n\n".join(doc.page_content for doc in docs)


def build_llm_chain():
    """Return a chain that takes {context, question, chat_history} and streams an answer."""
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()
