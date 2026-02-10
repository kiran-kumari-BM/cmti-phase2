from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

# ---------- Load Embedding Model ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Load Local LLM ----------
model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)


# ---------- Helper: Chunk Text ----------
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# ---------- Main RAG Function ----------
def ask_question(document_text, question):

    chunks = chunk_text(document_text)

    if not chunks:
        return "No content found.", ""

    embeddings = embedder.encode(chunks)

    question_embedding = embedder.encode([question])[0]

    similarities = np.dot(embeddings, question_embedding)

    best_index = int(np.argmax(similarities))
    best_chunk = chunks[best_index]

    prompt = f"""
You are an AI assistant answering based only on the provided document.

Document:
{best_chunk}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, best_chunk