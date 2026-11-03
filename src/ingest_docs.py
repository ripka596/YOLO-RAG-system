from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb, torch, pickle, os, re

DATA_DIR = Path("data/docs")
CHROMA_DB_DIR = Path("chroma_db")
COLLECTION_NAME = "YOLOdocs"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BM25_INDEX_PATH = "bm25/bm25_index.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    BATCH_SIZE = 5000
    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end]
        )

        print(f"Добавлено {end if end < len(documents) else len(documents)} из {len(documents)} чанков в ChromaDB")


    bm25 = BM25Okapi([re.findall(r'\w+', d.lower()) for d in documents]) #каждый документ делим на слова

    os.makedirs("bm25", exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            'bm25': bm25,
            'ids': ids,
            'docs': documents,
            'meta': metadatas,
        }, f)
    print(f"Загружено {len(documents)} чанков в bm25")

if __name__ == "__main__":
    ingest_docs()