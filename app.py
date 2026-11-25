import streamlit as st
from dotenv import load_dotenv

from src.ingestion import parse_video_id, get_transcript_data, VideoError
from src.chunking import chunk_transcript
from src.embedding import get_embedding_model
from src.faiss_index import get_or_create_vector_store
from src.reranker import get_reranking_retriever
from src.context_builder import get_prompt_template
from src.generator import get_llm
from src.pipeline import run_rag, get_sources, is_refusal, fmt_time
from src.summarizer import needs_summary, build_summary_chain

load_dotenv()  

MAX_QUESTIONS = 10


# cached heavy objects

@st.cache_resource(show_spinner=False)
def load_video(video_id: str):
    """Fetch transcript, build or load the index. Runs once per video."""
    text, cues = get_transcript_data(video_id)
    store = get_or_create_vector_store(
        get_embedding_model(),
        lambda: chunk_transcript(text, cues, video_id),
        f"data/{video_id}",
    )
    return text, get_reranking_retriever(store, k=4, fetch_k=20)


@st.cache_resource(show_spinner=False)
def get_models():
    llm = get_llm()
    return get_prompt_template(), llm, build_summary_chain(llm)


# page

st.set_page_config(page_title="TubeRAG", page_icon="🎬")
st.title("🎬 TubeRAG")
st.caption("Chat with any YouTube video that has English captions. "
           "Answers come only from the transcript.")

ss = st.session_state
ss.setdefault("video_id", None)
ss.setdefault("messages", [])
ss.setdefault("asked", 0)
ss.setdefault("start", 0)

url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
if st.button("Load video"):
    try:
        vid = parse_video_id(url)
        with st.spinner("Reading the transcript and building the index..."):
            load_video(vid)
        ss.video_id, ss.messages, ss.start = vid, [], 0
    except VideoError as e:
        st.error(str(e))

if ss.video_id:
    text, retriever = load_video(ss.video_id)
    prompt, llm, summary_chain = get_models()

    st.video(f"https://youtu.be/{ss.video_id}", start_time=ss.start)

    # chat history
    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                st.caption("Sources (approximate times). Click to jump:")
                cols = st.columns(len(m["sources"]))
                for col, (t, _) in zip(cols, m["sources"]):
                    if col.button(f"▶ {fmt_time(t)}", key=f"src-{i}-{t}"):
                        ss.start = t
                        st.rerun()

    # new question
    if ss.asked >= MAX_QUESTIONS:
        st.info("You've reached the question limit for this session.")
    elif q := st.chat_input("Ask about the video"):
        with st.chat_message("user"):
            st.markdown(q)
        ss.messages.append({"role": "user", "content": q})
        ss.asked += 1

        with st.spinner("Thinking..."):
            try:
                if needs_summary(q):
                    answer = summary_chain.invoke({"transcript": text, "question": q})
                    sources = []
                else:
                    result = run_rag(retriever, prompt, llm, q)
                    answer = result["answer"]
                    sources = [] if is_refusal(answer) else get_sources(result["docs"][:3])
            except Exception:
                answer, sources = "Something went wrong. Please try again.", []

        ss.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()