from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch, re, os, pickle


def tokenize_bm25(text):
    """Токенизация для BM25"""
    tokens = []
    for word in re.findall(r'[\w-]+', text):
        cleaned_word = word.strip("-_")
        if not cleaned_word:
            continue
        tokens.append(cleaned_word.lower())
        parts = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z]|\b)', cleaned_word)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts if p)

    return tokens


def upload_to_chromadb(collection, ids, documents, metadatas):
    "Векторизация и загрузка документов в ChromaDB"

    EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
    batch_size = 5000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SentenceTransformer(EMBEDDER, device=device)
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size]
        )
        
    print(f"Добавлено {len(documents)} документов в коллекцию '{collection.name}' ChromaDB")


def create_bm25_index(ids, documents, metadatas, index_path):
    """Создание и сохранение BM25 индекса"""

    tokenized_docs = [tokenize_bm25(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    os.makedirs("bm25", exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": ids,
            "docs": documents,
            "meta": metadatas
            }, f)
        
    print(f"BM25 индекс сохранен в {index_path}")

def rrf_docs(*lcand, k=60):
    """
    RRF объединение результатов
    """
    scores = {}
    meta = {}

    for cand in lcand:
        for rank, (chunk, metadata) in enumerate(cand):
            key = (chunk, metadata.get('source', ''))
            if key not in scores:
                scores[key] = 0
                meta[key] = (chunk, metadata)
            scores[key] += 1 / (k + rank + 1)
            
    return [meta[key] for key in sorted(scores.keys(), key=scores.get, reverse=True)]

def rr_docs(*lcand, rerank_k):
    seen = set()
    results = []

    n = len(lcand)
    quota = (rerank_k//n) + 1
    for cand in lcand:
        for chunk, meta in cand[:quota]:
            key = (chunk, meta.get('source', ''))
            if key not in seen:
                seen.add(key)
                results.append((chunk, meta))
                if len(results) >= rerank_k:
                    return results
    return results