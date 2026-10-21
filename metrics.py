import json, time, os, random, pandas as pd, ollama
from main import hybrid_search, LLM_MODEL_NAME, bm25_docs, bm25_meta

GROUND_TRUTH = "ground_truth.json"
N_QUESTIONS = 25

configs = [
    ("vector", False),
    ("vector", True),
    ("bm25", False),
    ("bm25", True),
    ("rrf", False),
    ("rrf", True),
    ("concatenate", False),
    ("concatenate", True),
    ("score", False),
    ("score", True),
]


def calc_metrics(chunks: list, ground_truth: list, k: int = 5):
    """Рассчёт Recall@5 и RR для одного вопроса по совпадениям чанков."""
    top_k = chunks[:k]
    if not ground_truth:
        return 0.0, 0.0

    matched_gt_indices = set()
    first_rank = 0

    for rank, (doc, meta) in enumerate(top_k, start=1):
        source = (meta.get("source", ""))
        text = " ".join((doc or "").lower().split())

        for idx, gt in enumerate(ground_truth):
            gt_source = gt["source"]
            gt_span = " ".join(gt["span"].lower().split())

            if gt_span and source.endswith(gt_source) and gt_span in text:
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
    sampled = random.sample(docs, len(docs))

    dataset = []

    for idx in sampled:
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

    for _ in range(3): #прогрев
        hybrid_search(dataset[0]["question"], n_results = 25)
    
    rows = []
    for mode, reranker in configs:
        label = f"{mode}{' + rerank' if reranker else ''}"
        recalls, rrs, times = [], [], []

        for item in dataset:
            start = time.perf_counter()
            chunks = hybrid_search(item["question"], n_results=25, mode=mode, use_reranker=reranker)
            times.append(time.perf_counter() - start)

            recall, rr = calc_metrics(chunks, item["ground_truth"])
            recalls.append(recall)
            rrs.append(rr)
        rows.append(
            {
                "Метод": label,
                "Recall@5": round(sum(recalls) / len(recalls), 3),
                "MRR": round(sum(rrs) / len(rrs), 3),
                "Время": round(sum(times) / len(times), 4)
            }
        )
    pd.DataFrame(rows).to_csv("metrics.results.csv", index=False, encoding="utf-8-sig")
