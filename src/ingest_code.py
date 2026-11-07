import pickle, chromadb, torch, os, re
from pathlib import Path
from rank_bm25 import BM25Okapi
from ast_chunker import parse_code
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import create_bm25_index, upload_to_chromadb

DATA = Path("data")
CPP_DIR = DATA / "code_cpp"
PY_DIR = DATA / "code_py"
DB = "chroma_db"
CODE_COLLECTION = "YOLOcodes"
BM25_CODE = "bm25/bm25_code.pkl"

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

    client = chromadb.PersistentClient(path=DB)
    try:
        client.delete_collection(CODE_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(name=CODE_COLLECTION, metadata={"hnsw:space": "cosine"})

    upload_to_chromadb(collection, ids, documents, metadatas)
    create_bm25_index(ids, documents, metadatas, BM25_CODE)

if __name__ == "__main__":
    ingest_code()