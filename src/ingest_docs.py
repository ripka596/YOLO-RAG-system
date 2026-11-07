from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb, torch, pickle, os, re
from utils import upload_to_chromadb, create_bm25_index

DATA_DIR = Path("data/docs")
CHROMA_DB_DIR = Path("chroma_db")
COLLECTION_NAME = "YOLOdocs"
BM25_INDEX_PATH = "bm25/bm25_docs.pkl"

def ingest_docs():
    docs = []
    for md_file in DATA_DIR.rglob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        docs.append({"text": text, "source": str(md_file)})

    if not docs:
        print(f"Не найдено .md файлов в {DATA_DIR}")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n## ", "\n# ", "\n\n", "\n", " ", ""]
    )

    documents, metadatas, ids = [], [], []
    for doc in docs:
        for i, chunk in enumerate(splitter.split_text(doc["text"])):
            documents.append(chunk)
            metadatas.append({"source": doc["source"], "chunk_id": i})
            ids.append(f"{doc['source']}::{i}")

    print (f"Создано {len(documents)} чанков из {len(docs)} документов.")

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    upload_to_chromadb(collection, ids, documents, metadatas)
    create_bm25_index(ids, documents, metadatas, BM25_INDEX_PATH)


if __name__ == "__main__":
    ingest_docs()