import json, time, os, statistics, random, pandas as pd, ollama
from main import hybrid_search, LLM_MODEL_NAME, bm25_docs, bm25_meta

GROUND_TRUTH = "metrics/ground_truth.json"
N_QUESTIONS = 25
N_RUNS = 3

configs = [
    ("vector", False, None),
    ("vector", True, None),
    ("vector", True, 0),
    ("vector", True, 6),
    ("bm25", False, None),
    ("bm25", True, None),
    ("bm25", True, 0),
    ("rrf", False, None),
    ("rrf", True, None),
    ("rrf", True, 0),
    ("concatenate", False, None),
    ("concatenate", True, None),
    ("concatenate", True, 0),
    ("score", False, None),
    ("score", True, None),
    ("score", True, 0),
    ("score", True, 6)
]


def calc_metrics(chunks: list, ground_truth: list, k: int = 5):
    """Рассчёт Recall@5 и RR для одного вопроса по совпадениям чанков."""
    chunks = chunks[:k]
    if not ground_truth:
        return 0.0, 0.0

    matched_gt_indices = set()
    first_rank = 0

    for rank, (doc, meta) in enumerate(chunks, start=1):
        source = (meta.get("source", ""))
        text = " ".join((doc or "").lower().split())

        for idx, gt in enumerate(ground_truth):
            gt_source = gt["source"]
            gt_span = " ".join(gt["span"].lower().split())

            if gt_span and source == gt_source and gt_span in text:
                matched_gt_indices.add(idx)
                if first_rank == 0:
                    first_rank = rank

    recall = len(matched_gt_indices) / len(ground_truth)
    rr = (1.0 / first_rank) if first_rank > 0 else 0.0
    return round(recall, 3), round(rr, 3)

def gen_dataset(num_q: int = N_QUESTIONS):
    """Генерация вопроса на чанк и точная цитата для ответа"""
    docs = [i for i, doc in enumerate(bm25_docs) if doc and len(doc.strip()) > 300 and bm25_meta[i]]

    random.seed(33)
    random.shuffle(docs)

    dataset = []

    for idx in docs:
        if len(dataset) >= num_q:
            break

        text = bm25_docs[idx]
        source = bm25_meta[idx].get("source", "")

        try:
            response = ollama.chat(
                model=LLM_MODEL_NAME,
                messages=[{
                    "role": "system", 
                    "content": """
                        Прочитай чанк документации и придумай к нему вопрос (question) на русском языке, ответ на который будет в этом чанке. Также выведи цитату (quote) из этого чанка, которая будет подтверждать ответ.
                        Отвечай строго в JSON: {"question": question, "span": quote}
                    """},
                    {"role": "user", "content": f"Чанк документации: {text}"}
                ],
                format="json",
                think=True,
                options={"temperature": 0.4},
            )
            data = json.loads(response["message"]["content"])
            question = data.get("question", "").strip()
            span = data.get("span", "").strip()

            check_chunk = " ".join(text.lower().split())
            check_span = " ".join(span.lower().split())

            if question and len(check_span) >= 15 and check_span in check_chunk:
                dataset.append(
                    {
                        "question": question,
                        "ground_truth": [
                            {"source": source, "span": span}
                        ],
                    }
                )

        except Exception as e:
            print(f"Ошибка генерации для чанка {idx}: {e}")
            continue

    with open(GROUND_TRUTH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    return dataset


if __name__ == "__main__":
    if os.path.exists(GROUND_TRUTH):
        with open(GROUND_TRUTH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = gen_dataset(N_QUESTIONS)

    warmup_q = dataset[0]["question"]
    
    results = {}
    for mode, reranker, threshold in configs:
        label = f"{mode}{' + rerank' if reranker else ''}{f' (score >={threshold})' if threshold is not None else ''}"

        for _ in range(3):
            hybrid_search(warmup_q, n_results=25, mode=mode, use_reranker=reranker, score_threshold=threshold)

        recalls, rrs, times, counts = [], [], [], []

        for item in dataset:
            chunks = hybrid_search(item["question"], n_results=25, mode=mode, use_reranker=reranker, score_threshold=threshold)
            recall, rr = calc_metrics(chunks, item["ground_truth"])
            recalls.append(recall)
            rrs.append(rr)
            counts.append(len(chunks))

            for _ in range(N_RUNS):
                start = time.perf_counter()
                hybrid_search(item["question"], n_results=25, mode=mode, use_reranker=reranker, score_threshold=threshold)
                times.append(time.perf_counter() - start)

        avg_time = round(sum(times) / len(times), 4)
        std_time = round((statistics.stdev(times) if len(times) > 1 else 0.0), 4)

        results[label] = {
                "Recall@5": round(sum(recalls) / len(recalls), 3),
                "MRR": round(sum(rrs) / len(rrs), 3),
                "Avg chunks": round(sum(counts) / len(counts), 1),
                "Время (s)": avg_time,
                "Отклонение (s)": std_time
            }
        
    df = pd.DataFrame(results)
    df.to_csv("metrics/metrics_results.csv", index_label="Метрика", encoding="utf-8-sig")
    df.to_excel("metrics/metrics_results.xlsx", index_label="Метрика")
