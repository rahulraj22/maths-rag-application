import re
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag import build_vectorstore, get_all_headings, retrieve, build_llm_chain


def fix_latex(text: str) -> str:
    """Convert LLM output delimiters to Streamlit-compatible KaTeX delimiters."""
    text = re.sub(r'\\\[(.+?)\\\]', lambda m: f'$$\n{m.group(1)}\n$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', lambda m: f'${m.group(1)}$', text, flags=re.DOTALL)
    return text


def get_chat_history() -> list:
    """Convert session messages to LangChain message objects, excluding the latest user message."""
    history = []
    for msg in st.session_state.get("messages", [])[:-1]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


st.set_page_config(
    page_title="Math RAG — Linear Transformations",
    page_icon="∑",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading course notes and vector store...")
def get_vectorstore():
    return build_vectorstore()


@st.cache_resource(show_spinner=False)
def get_llm_chain():
    return build_llm_chain()


vectorstore = get_vectorstore()
llm_chain = get_llm_chain()
all_headings = get_all_headings(vectorstore)

# --- Sidebar ---
with st.sidebar:
    st.header("Filter by Section")
    st.caption("Narrow retrieval to specific sections of the notes. Leave empty to search all.")
    selected_headings = st.multiselect(
        label="Sections",
        options=all_headings,
        default=[],
        placeholder="All sections",
    )
    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main chat UI ---
st.title("∑ Linear Transformations Q&A")
st.caption("Ask questions about linear transformations, null spaces, eigenvalues, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(fix_latex(msg["content"]))

if query := st.chat_input("Ask a question about linear transformations..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        placeholder.markdown("*Thinking...*")

        context = retrieve(
            vectorstore,
            query,
            k=5,
            headings=selected_headings if selected_headings else None,
        )
        for chunk in llm_chain.stream({
            "context": context,
            "question": query,
            "chat_history": get_chat_history(),
        }):
            full_response += chunk
            placeholder.markdown(fix_latex(full_response) + "▌")

        placeholder.markdown(fix_latex(full_response))

    st.session_state.messages.append({"role": "assistant", "content": fix_latex(full_response)})
