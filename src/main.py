import pickle, ollama, chromadb, torch, gradio as gr, numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from itertools import chain
from typing import Optional

from dataclasses import dataclass, replace
from multi_query import multi_query
from router import get_route
from utils import tokenize_bm25, rrf_docs, rr_docs


CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
LLM_MODEL_NAME = "qwen3_5:latest"

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
reranker = CrossEncoder(RERANK_MODEL_NAME, device=device)

@dataclass
class SearchConfig:
    mode: str = "concatenate"  # "concatenate", "rrf", "score", "vector", "bm25"
    use_reranker: bool = True
    score_threshold: Optional[float] = None
    top_k: int = 8
    rerank_k: int = 20
    use_router: bool = False
    multi_query: bool = True

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

collection_docs = client.get_collection(name="YOLOdocs")
with open("bm25/bm25_docs.pkl", "rb") as f:
    bm25_data_docs = pickle.load(f)

collection_code = client.get_collection(name="YOLOcodes")
with open("bm25/bm25_code.pkl", "rb") as f:
    bm25_data_code = pickle.load(f)



def get_bm25(db_type):
    """Возвращает нужные индексы bm25"""
    data = bm25_data_docs if db_type == "docs" else bm25_data_code
    return data['bm25'], data['ids'], data['docs'], data['meta'] 


def rrf(vec_ids, bm_ids):
    """
    RRF объединение результатов
    """
    k = 60 #константа сглаживания
    scores = {}
    for rank, doc_id in enumerate(vec_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=scores.get, reverse=True)


def score_fusion(vec_ids, vec_dists, bm_ids, bm_scores):
    """
    Слияние по скорам
    """
    alpha = 0.5 #весовой коэффициент, баланс между векторным и текстовым результатами
    def normalize(values):
        if len(values) == 0:
            return []
        mn, mx = min(values), max(values)
        if mx == mn:
            return [0.5] * len(values)
        return [(v-mn) / (mx - mn) for v in values]

    vec_scores = normalize([-d for d in vec_dists])
    normalize_bm = normalize(bm_scores)

    scores_results = {}
    for doc_id, s in zip(vec_ids, vec_scores):
        scores_results[doc_id] = alpha * s
    for doc_id, s in zip(bm_ids, normalize_bm):
        scores_results[doc_id] = scores_results.get(doc_id, 0) + (1 - alpha) * s

    return sorted(scores_results.keys(), key=scores_results.get, reverse=True)


def rerank(user_question: str, docs_metas: list, config: SearchConfig = None):
    """
    Реранкинг результатов поиска, возвращает top_k лучших результатов или фильтрует по порогу score_threshold
    """
    if config is None:
        config = SearchConfig()

    if not docs_metas:
        return []
    docs, metas = zip(*docs_metas)

    pairs = [[user_question, doc] for doc in docs]

    scores = reranker.predict(pairs)
    reranked_results = np.argsort(-np.array(scores))[:config.top_k]
    if config.score_threshold is not None:
        results = [(docs[i], metas[i]) for i in reranked_results if scores[i] >= config.score_threshold]
    else:
        results = [(docs[i], metas[i]) for i in reranked_results]

    return results


def hybrid_search(user_question: str, n_results: int, db_target="docs", config: SearchConfig = None):
    """
    Гибридный поиск BM25 + векторный
    mode:
     "concatenate" - объединение двух поисков
     "rrf" - слияние по рангам
     "score" - слияние по скорам
     "vector" - только векторный поиск
     "bm25" - только текстовый поиск
    По умолчанию - RRF
    Используется реранкер
    Возвращает rerank_k кандидатов
    """

    collection = collection_docs if db_target == "docs" else collection_code
    target_bm_model, target_bm_ids, target_bm_docs, target_bm_meta = get_bm25(db_target)

    vec_ids, vec_docs, vec_meta, vec_dists = [], [], [], []
    bm_ids, bm_docs, bm_meta, bm_scores = [], [], [], []

    if config.mode != "bm25":
    # Векторный поиск
        query_vector = embedder.encode(user_question).tolist()
        vector_results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        vec_ids, vec_docs, vec_meta, vec_dists = vector_results['ids'][0], vector_results['documents'][0], vector_results['metadatas'][0], vector_results['distances'][0]

    if config.mode != "vector":
    # поиск BM25
        words_query = tokenize_bm25(user_question) #вопрос пользователя по словам
        bm25_all_scores = target_bm_model.get_scores(words_query)
        bm25_results = np.argsort(-bm25_all_scores)[:n_results]

        bm_ids = [target_bm_ids[i] for i in bm25_results]
        bm_docs = [target_bm_docs[i] for i in bm25_results]
        bm_meta = [target_bm_meta[i] for i in bm25_results]
        bm_scores = [bm25_all_scores[i] for i in bm25_results]

    match config.mode:
        case "concatenate":
            all_ids = list(chain.from_iterable(zip(vec_ids, bm_ids)))
        case "rrf":
            all_ids = rrf(vec_ids, bm_ids)
        case "score":
            all_ids = score_fusion(vec_ids, vec_dists, bm_ids, bm_scores)
        case "vector":
            all_ids = vec_ids
        case "bm25":
            all_ids = bm_ids
        case _:
            raise ValueError(f"Неизвестный способ объединения {config.mode}")

    all_docs = {}
    all_meta = {}
    for id, doc, meta in zip(vec_ids+bm_ids, vec_docs+bm_docs, vec_meta+bm_meta):
        if id not in all_docs:
            all_docs[id] = doc
            all_meta[id] = meta

    seen_ids = set()
    unique_docs = []
    unique_meta = []

    for id in all_ids:
        if id not in seen_ids and id in all_docs:
            seen_ids.add(id)
            unique_docs.append(all_docs[id])
            unique_meta.append(all_meta[id])

    if not unique_docs:
        return []
    results = list(zip(unique_docs[:config.rerank_k], unique_meta[:config.rerank_k]))
    return results


def routing(user_question: str, config: SearchConfig = None, score_fuze = True):
    """Создание альтернативных вопросов, маршрутизация и отправка в поиск, связывает hybrid_search() и rerank()"""
    if config is None:
        config = SearchConfig()

    if config.multi_query:
        all_queries = multi_query(user_question)
        #print(f"\nАльтернативные вопросы:")
        #for i, question in enumerate(all_queries):
        #    print(f"{i + 1}. {question}")
    else:
        all_queries = [user_question]

    route = get_route(user_question) if config.use_router else "BOTH"
    results = []
    for q in all_queries:

        if route in ["DOCS", "BOTH"]:
            results.append(hybrid_search(q, n_results=25, config=config, db_target="docs"))

        if route in ["CODE", "BOTH"]:
            results.append(hybrid_search(q, n_results=25, config=config, db_target="code"))

    if len(results) == 1:
        merged_results = results[0]
    else:
        if score_fuze:
            merged_results = rr_docs(*results, rerank_k=config.rerank_k)
        else:
            merged_results = rrf_docs(*results)

    documentation = rerank(user_question, merged_results[:config.rerank_k], config=config) if config.use_reranker else merged_results[:config.top_k]

    return documentation

    
def rag_query(user_question: str, history: list):
    """
    RAG-пайплайн с функцией гибридного поиска
    """
    try:
        documentation = routing(user_question)

        if not documentation:
            return "Не найдено результатов"
        
        context = "\n\n".join([doc for doc, _ in documentation])
        metadatas = [meta for _, meta in documentation]

        sources = "\n".join(sorted(set([m.get('source') or m.get('file_name') or 'неизвестно' for m in metadatas if m])))
        
        prompt = f"""
        ВОПРОС: {user_question}

        КОНТЕКСТ
        {context}
        КОНЕЦ КОНТЕКСТА 

        ОТВЕТ:"""

        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "Ты - ассистент по документации. Отвечай строго по контексту и кратко (не более 5 предложений), код в контексте можешь привести в ответе. Если ответа нет в контексте, скажи, что ответа нет."},
                {"role": "user", "content": prompt}
            ],
            think=False,
            keep_alive="30m",
            options={
                'temperature': 0.2
            }
        )

        return f"{response['message']['content']}\n\n---\n**Источники:**\n{sources}"

    except chromadb.errors.ChromaError as e:
        print(f"Ошибка БД: {e}")
        return "Ошибка базы данных!"

    except Exception as e:
        print(f"Ошибка выполнения RAG-пайплайна: {e}")
        return "Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз."


if __name__ == "__main__":
    gr.ChatInterface(
        fn=rag_query,
        title="Ассистент по документации YOLO",
        description="Задайте вопрос, и я постараюсь найти ответ в документации YOLO.",
        examples=[
            "Как использовать YOLO для обнаружения объектов?",
            "Какие модели YOLO поддерживаются?",
            "Как настроить обучение модели YOLO?",
            "Где найти документацию по YOLO?",
            "Как использовать YOLO с Python?"
        ]
    ).launch()