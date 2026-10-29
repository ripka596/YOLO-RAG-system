import pickle, chromadb, torch, os
from pathlib import Path
from rank_bm25 import BM25Okapi
from ast_chunker import parse_code
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA = Path("data")
CPP_DIR = DATA / "code_cpp"
PY_DIR = DATA / "code_py"
DB = "chroma_db"
CODE_COLLECTION = "YOLOcodes"
BM25_CODE = "bm25/bm25_code.pkl"
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"


def collect_chunks():
    all_chunks = []
    target_dirs = [CPP_DIR, PY_DIR]

    for dir in target_dirs:
        if not dir.exists():
            continue
        for file_path in dir.rglob("*"):
            if file_path.is_file():
                all_chunks.extend(parse_code(file_path))
    return all_chunks

def ingest_code():
    raw_chunks = collect_chunks()
    if not raw_chunks:
        return

    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3072,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for chunk in raw_chunks:
        text = chunk["text"]
        if len(text) < 4000:
            chunks.append(chunk)
        else:
            sub_texts = code_splitter.split_text(text)
            for i, sub_text in enumerate(sub_texts):
                node_type = chunk["metadata"]["node_type"]
                entity_name = chunk["metadata"]["entity_name"]
                content = f"// [Part {i+1}/{len(sub_texts)}] of {node_type} '{entity_name}':\n"

                chunks.append({
                    "text": content + sub_text,
                    "metadata": chunk["metadata"]
                })

    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"code_{i}" for i in range(len(chunks))]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDER, device=device)
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=DB)
    try:
        client.delete_collection(CODE_COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(name=CODE_COLLECTION, metadata={"hnsw:space": "cosine"})

    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)

    os.makedirs("bm25", exist_ok=True)
    with open(BM25_CODE, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": ids,
            "docs": documents,
            "meta": metadatas
            }, f)

if __name__ == "__main__":
    ingest_code()