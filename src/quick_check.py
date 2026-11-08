import json, os, time, pandas as pd
import main
from main import SearchConfig, routing
from metrics import GROUND_TRUTH, calc_metrics
from multi_query import multi_query as mq

CACHE_PATH = "metrics/multiq_cache.json"

with open(GROUND_TRUTH, encoding="utf-8") as f:
    dataset = json.load(f)

os.makedirs("metrics", exist_ok=True)
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, encoding="utf-8") as f:
        mq_cache = json.load(f)
    print(f"Кэш загружен: {len(mq_cache)} записей")
else:

    mq_cache = {}
    print("Строим кэш multi-query")
    for i, item in enumerate(dataset, 1):
        mq_cache[item["question"]] = mq(item["question"])
        print(f"{i}/{len(dataset)}")
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(mq_cache, f, ensure_ascii=False, indent=2)


def cached_multi_query(question):
    """Возвращает альтернативные вопросы из кэша без вызова LLM"""
    if question in mq_cache:
        return mq_cache[question]
    return [question]

main.multi_query = cached_multi_query

base_configs = [
    ("router + mq", True, True),
    ("router, no mq", True, False),
    ("no router + mq", False, True),
    ("no router, no mq", False, False)
]


warmup_q = dataset[0]["question"]
all_results = {}

for mode in ["concatenate", "score", "score_k"]:
    configs = []
    use_score_k = (mode == "score_k")
    for name, use_router, use_mq in base_configs:
        cfg = SearchConfig(
            mode=mode if mode == "concatenate" else "score", use_reranker=True,
            score_threshold=0, top_k=8,
            rerank_k=75, use_router=use_router,
            multi_query=use_mq
            )
        configs.append((name, cfg))
    results = {}
    for name, cfg in configs:
        routing(warmup_q, config=cfg, score_fuze=use_score_k)

        recalls, rrs, counts, times, recalls8, rrs8 = [], [], [], [], [], []

        for item in dataset:
            start = time.perf_counter()
            chunks = routing(item["question"], config=cfg, score_fuze=use_score_k)
            times.append(time.perf_counter() - start)

            recall, rr = calc_metrics(chunks, item["ground_truth"])
            recalls.append(recall)
            rrs.append(rr)
            recall8, rr8 = calc_metrics(chunks, item["ground_truth"], k=8)
            recalls8.append(recall8)
            rrs8.append(rr8)
            counts.append(len(chunks))

        results[name] = {
            "Recall@5": round(sum(recalls) / len(recalls), 3),
            "MRR@5": round(sum(rrs) / len(rrs), 3),
            "Recall@8": round(sum(recalls8) / len(recalls8), 3),
            "MRR@8": round(sum(rrs8) / len(rrs8), 3),
            "Avg chunks": round(sum(counts) / len(counts), 1),
            "Latency (s)": round(sum(times) / len(times), 2),
        }
        print(f"[{name}] готово")

    all_results[mode] = pd.DataFrame(results)

df_concat = all_results["concatenate"]
df_score = all_results["score"]
df_score_k = all_results["score_k"]

print("\n=== Concatenate + rerank (score>=0) AND rrf_docs ===")
print(df_concat.to_string())
print("\n=== Score + rerank (score>=0) AND rrf_docs ===")
print(df_score.to_string())
print("\n=== Score + rerank (score>=0) AND score_k ===")
print(df_score_k.to_string())

df_concat.to_csv(f"metrics/quick_check_concat.csv", index_label="Метрика", encoding="utf-8-sig")
df_score.to_csv(f"metrics/quick_check_score.csv", index_label="Метрика", encoding="utf-8-sig")
df_score_k.to_csv(f"metrics/quick_check_score_k.csv", index_label="Метрика", encoding="utf-8-sig")