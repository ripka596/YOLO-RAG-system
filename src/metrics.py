import json, time, os, statistics, random, pandas as pd, ollama
from main import LLM_MODEL_NAME, bm25_data_docs, bm25_data_code, routing

GROUND_TRUTH = "metrics/ground_truth.json"
N_QUESTIONS = 30
N_RUNS = 3

configs = [
    ("vector", False, None),
    ("vector", True, 6),
    ("bm25", False, None),
    ("rrf", True, None),
    ("rrf", True, 0),
    ("concatenate", True, None),
    ("concatenate", True, 0),
    ("score", True, None),
    ("score", True, 0),
    ("score", True, 6)
]


def calc_metrics(chunks: list, ground_truth: list, k: int = 5):
    """Рассчёт Recall@5 и RR для одного вопроса по цитатам."""
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

def gen_questions(bm25_data, db_name, num_q):
    """Генерация вопросов по конкретной базе (документация или код)"""
    docs = bm25_data['docs']
    meta = bm25_data['meta']

    valid_idx = [i for i, doc in enumerate(docs) if doc and len(doc.strip()) > 300 and meta[i]]
    random.seed(26)
    random.shuffle(valid_idx)

    dataset = []
    db_type = "исходного кода" if db_name == "code" else "документации"

    for idx in valid_idx:
        if len(dataset) >= num_q:
            break
        text = docs[idx]
        source = meta[idx].get("source", "")

        try:
            response = ollama.chat(
                model=LLM_MODEL_NAME,
                messages=[{
                    "role": "system",
                    "content": f"""
                    Прочитай чанк {db_type} и придумай к нему вопрос (question) на русском языке, ответ на который будет в этом чанке. Также выведи цитату (quote) из этого чанка, которая будет подтверждать ответ.
                    Отвечай строго в JSON: {{"question": question, "span": quote}}
                    """},
                    {"role": "user", "content": f"Чанк: {text}"}
                    ],
                    format="json",
                    think=False,
                    options={"temperature": 0.3}
            )
            data = json.loads(response["message"]["content"])
            question = data.get("question", "").strip()
            span = data.get("span", "").strip()

            check_chunk = " ".join(text.lower().split())
            check_span = " ".join(span.lower().split())

            if question and len(check_span) >= 15 and check_span in check_chunk:
                dataset.append({
                        "question": question,
                        "ground_truth": [{"source": source, "span": span}]
                            })

        except Exception as e:
            print(f"Ошибка генерации для чанка {idx}: {e}. \n Ответ нейросети:\n'{response["message"]["content"]}'\n\nТекст: {text[:200]}")
            continue
    return dataset


def gen_dataset(num_q: int = N_QUESTIONS):
    """Генерация смешанного датасета"""
    os.makedirs("metrics", exist_ok=True)
    dataset = []
    half = num_q // 2

    dataset.extend(gen_questions(bm25_data_docs, "docs", half))
    dataset.extend(gen_questions(bm25_data_code, "code", half))

    with open(GROUND_TRUTH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print("Датасет готов")
    return dataset


if __name__ == "__main__":
    if os.path.exists(GROUND_TRUTH):
        with open(GROUND_TRUTH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = gen_dataset(N_QUESTIONS)

    warmup_q = dataset[0]["question"]
    
    results = {}
    results_router = {}
    for mode, reranker, threshold in configs:
        label = f"{mode}{' + rerank' if reranker else ''}{f' (score >={threshold})' if threshold is not None else ''}"
        for use_router in [True, False]:
            router_status = "С роутером" if use_router else "Без роутера"

            for _ in range(3):
                routing(warmup_q, mode=mode, use_reranker=reranker, score_threshold=threshold, use_router=use_router)

            recalls, rrs, times, counts = [], [], [], []

            for item in dataset:
                chunks = routing(item["question"], mode=mode, use_reranker=reranker, score_threshold=threshold, use_router=use_router)
                recall, rr = calc_metrics(chunks, item["ground_truth"])
                recalls.append(recall)
                rrs.append(rr)
                counts.append(len(chunks))

                for _ in range(N_RUNS):
                    start = time.perf_counter()
                    routing(item["question"], mode=mode, use_reranker=reranker, score_threshold=threshold, use_router=use_router)
                    times.append(time.perf_counter() - start)

            avg_time = round(sum(times) / len(times), 4)
            std_time = round((statistics.stdev(times) if len(times) > 1 else 0.0), 4)

            data = {
                    "Recall@5": round(sum(recalls) / len(recalls), 3),
                    "MRR": round(sum(rrs) / len(rrs), 3),
                    "Avg chunks": round(sum(counts) / len(counts), 1),
                    "Время (s)": avg_time,
                    "Отклонение (s)": std_time
                }
            if use_router:
                results[label] = data
            else:
                results_router[label] = data
        
    df = pd.DataFrame(results)
    df_router = pd.DataFrame(results_router)

    xlsx_path = "metrics/metrics_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Результаты", startrow=0, index_label="Метрика")
        worksheet = writer.sheets["Результаты"]
        start_row2 = len(df) + 4

        worksheet.write_string(0, 0, "Без роутера")
        worksheet.write_string(start_row2-2, 0 , "С роутером")

        df_router.to_excel(writer, sheet_name="Результаты", startrow=start_row2, index_label="Метрика")

    df.to_csv("metrics/results.csv", index_label="Метрика", encoding="utf-8-sig")
    df_router.to_csv("metrics/router_results.csv", index_label="Метрика", encoding="utf-8-sig")
    print("Метрики загружены в папку metrics!")
