"""The one RAG path: retrieve -> build context -> generate.
Both main.py and the pipeline eval call this, so they can never drift apart."""

from src.context_builder import format_docs
from src.generator import generate_answer


def run_rag(retriever, prompt, llm, question: str) -> dict:
    """Returns the answer AND the retrieved chunks (the eval and source citations need both)."""
    docs = retriever.invoke(question)
    final_prompt = prompt.invoke({"context": format_docs(docs), "question": question})
    return {"answer": generate_answer(llm, final_prompt), "docs": docs}


def fmt_time(t: int) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def get_sources(docs):
    """Distinct (start_time, url) pairs, in relevance order."""
    seen, out = set(), []
    for d in docs:
        t = d.metadata.get("start_time")
        if t is None or t in seen:
            continue
        seen.add(t)
        out.append((int(t), d.metadata["url"]))
    return out


def format_sources(docs) -> str:
    """Text version of the sources, used by the terminal app."""
    return "\n".join(f"  [{fmt_time(t)}] {url}" for t, url in get_sources(docs))


def is_refusal(answer: str) -> bool:
    """True when the model said it doesn't know (so we show no sources)."""
    return answer.replace("’", "'").lower().startswith("i don't know")