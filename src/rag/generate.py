"""
Generate an answer from retrieved context using a local Ollama model.

Requires Ollama running locally (https://ollama.com) with a small model pulled:
    ollama pull qwen2.5:1.5b
"""
import ollama

PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the context does not contain the answer, say "I don't know based on the given context."

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, chunks: list[dict], model: str = "qwen2.5:1.5b") -> str:
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


if __name__ == "__main__":
    from src.rag.retrieve import Retriever

    retriever = Retriever()
    question = "What does RAG combine?"
    chunks = retriever.retrieve(question, top_k=3)
    answer = generate_answer(question, chunks)
    print(f"Q: {question}\nA: {answer}")
