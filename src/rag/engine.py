import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_txt(path, chunk_size=800, overlap=200):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def load_pdf(path, chunk_size=800, overlap=200):
    reader = PdfReader(path)
    text = "\n".join([p.extract_text() or "" for p in reader.pages])
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def build_db(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("orion_docs")
    except:
        pass
    collection = client.create_collection("orion_docs")
    
    embeddings = model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"DB criado com {len(chunks)} chunks")
    return collection, model

def ask(question, collection, model):
    q_emb = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    docs = results['documents'][0]
    context = "\n\n".join(docs)
    
    prompt = f"""
    Use APENAS o contexto abaixo. Responda e ao final cite a fonte.
    Contexto: {context}
    Pergunta: {question}
    """
    
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, docs

if __name__ == "__main__":
    chunks = load_txt("data/flamengo.txt")
    collection, model = build_db(chunks)
    resposta, fontes = ask("Quantas Libertadores tem o Flamengo?", collection, model)
    print("\n--- RESPOSTA RAG ---")
    print(resposta)
    print("\n--- FONTES ---")
    print(fontes)