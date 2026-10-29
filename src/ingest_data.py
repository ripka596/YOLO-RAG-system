from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import torch

DATA_DIR = Path("data/docs")
CHROMA_DB_DIR = Path("chroma_db")
COLLECTION_NAME = "YOLOdocs"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

device = "cuda" if torch.cuda.is_available() else "cpu"

def update_chromaDB():
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

    chunks = []
    for doc in docs:
        for i, chunk in enumerate(splitter.split_text(doc["text"])):
            chunks.append({
                "text": chunk,
                "source": doc["source"],
                "chunk_id": i
            })

    print (f"Создано {len(chunks)} чанков из {len(docs)} документов.")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [f"{c['source']}::{c['chunk_id']}" for c in chunks]

    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

    BATCH_SIZE = 5000

    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE

        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end]
        )

        print(f"Добавлено {end if end < len(chunks) else len(chunks)} из {len(chunks)} чанков в ChromaDB")


if __name__ == "__main__":
    update_chromaDB()