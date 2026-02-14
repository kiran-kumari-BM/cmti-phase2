import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- Global Lazy Variables ----------
embedder = None
tokenizer = None
model = None
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_models():
    global embedder, tokenizer, model

    if embedder is None:
        print("Loading embedding model...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if tokenizer is None or model is None:
        print("Loading LLM...")
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(device)


def chunk_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def ask_question(document_text, question):

    if not document_text.strip():
        return "No content found.", ""

    # 🔥 Load models only when needed
    load_models()

    chunks = chunk_text(document_text)

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    question_embedding = embedder.encode([question], convert_to_numpy=True)[0]

    similarities = np.dot(embeddings, question_embedding)
    best_chunk = chunks[int(np.argmax(similarities))]

    prompt = f"""
Answer the question based only on the context below.

Context:
{best_chunk}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150
    )

    generated_text= tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("Answer:")[-1].strip()
    return answer, best_chunk