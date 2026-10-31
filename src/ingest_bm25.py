import pickle, re, chromadb, os
from rank_bm25 import BM25Okapi

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "YOLOdocs"
BM25_INDEX_PATH = "bm25/bm25_index.pkl"

def build_bm25():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    data = collection.get(include=['documents', 'metadatas'])
    docs = data['documents']

    bm25 = BM25Okapi([re.findall(r'\w+', d.lower()) for d in docs]) #каждый документ делим на слова

    os.makedirs("bm25", exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f: #пересоздаём файл индексов bm_25
        pickle.dump({
            'bm25': bm25,
            'ids': data['ids'],
            'docs': docs,
            'meta': data['metadatas'],
        }, f)

    print(f"Загружено {len(docs)} чанков в bm25")

if __name__ == "__main__":
    build_bm25()