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
all_results = {
    "concatenate": {},
    "concatenate_RR": {},
    "score": {},
    "score_RR": {}
}

for rerank_k in [20, 40, 75]:
    for mode in ["concatenate", "concatenate_RR", "score", "score_RR"]:
        configs = []
        use_score_docs = (mode == "score_RR" or mode == "concatenate_RR")
        for name, use_router, use_mq in base_configs:
            cfg = SearchConfig(
                mode="concatenate" if "concatenate" in mode else "score", use_reranker=True,
                score_threshold=0, top_k=8,
                rerank_k=rerank_k, use_router=use_router,
                multi_query=use_mq
                )
            configs.append((name, cfg))
        results = {}
        for name, cfg in configs:
            routing(warmup_q, config=cfg, score_fuze=use_score_docs)

            recalls, rrs, counts, times, recalls8, rrs8 = [], [], [], [], [], []

            for item in dataset:
                start = time.perf_counter()
                chunks = routing(item["question"], config=cfg, score_fuze=use_score_docs)
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

        all_results[mode][rerank_k] = pd.DataFrame(results)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("\n=====РЕЗУЛЬТАТЫ=====")
for mode, rk_dict in all_results.items():
    print(f"\n\n==Метод: {mode}==")
    df = list(rk_dict.values())
    keys = [f"rerank_k={rk}" for rk in rk_dict.keys()]
    result = pd.concat(df, axis=1, keys=keys)
    print(result)
    result.to_csv(f"metrics/quick_check_{mode}.csv", encoding="utf-8-sig")