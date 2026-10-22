import json, pandas as pd, time
from main import hybrid_search
from metrics import GROUND_TRUTH, calc_metrics


with open(GROUND_TRUTH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


MODE = "score"
TOP_K_LIST = [3, 5, 7, 9, 11, 13, 15]
RERANK_K_LIST = [5, 10, 15, 20, 25]
THRESHOLDS = [-2, 0, 2, 4, 6, 8]

results = {}
configs = []
for rerank_k in RERANK_K_LIST:
    for top_k in TOP_K_LIST:
        if top_k <= rerank_k:
            configs.append((rerank_k, top_k, None))

    for th in THRESHOLDS:
        configs.append((rerank_k, rerank_k, th))

for _ in range(3):
    hybrid_search(dataset[0]["question"], n_results=25, mode=MODE)

for rerank_k, top_k, th in configs:
    if th is None:
        label = f"rk={rerank_k} | top_k={top_k}"
    else:
        label = f"rk={rerank_k} | score>={th}"

    recalls, rrs, counts, times = [], [], [], []

    for item in dataset:
        start = time.perf_counter()
        chunks = hybrid_search(item["question"], n_results=25, mode=MODE, score_threshold=th, top_k=top_k, rerank_k=rerank_k)
        times.append(time.perf_counter() - start)

        eval_k = top_k if top_k is not None else len(chunks)
        recall, rr = calc_metrics(chunks, item["ground_truth"], k=eval_k)

        recalls.append(recall)
        rrs.append(rr)
        counts.append(len(chunks))
    results[label] = {
        "Recall": round(sum(recalls) / len(recalls), 3),
        "MRR": round(sum(rrs) / len(rrs), 3),
        "Avg chunks": round(sum(counts) / len(counts), 1),
        "Время (s)": round(sum(times) / len(times), 4),
    }

df = pd.DataFrame(results)
df.to_csv("metrics/sweep_metrics.csv", index_label="Метрика", encoding="utf-8-sig")
df.to_excel("metrics/sweep_metrics.xlsx", index_label="Метрика")