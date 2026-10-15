from main import hybrid_search
import time, json

questions = [
    "Как настроить работу с OBB датасетами и чем они отличаются от стандартного формата сегментации?",
    "Что делать, если при запуске валидации на CUDA падает ошибка Out of Memory?",
    "Как запустить инференс на весах ONNX с сохранением результатов в TXT, но без отображения окон на экране?",
    "Где физически на диске лежат дефолтные конфигурационные файлы для предобученных моделей в репозитории?",
    "Как заставить нейросеть предсказывать объекты на лету прямо с веб-камеры компьютера?",
    "Как настроить обучение модели YOLO?",
    "Для чего используется движок ollama?"
]

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

MODES = ["rrf + rerank", "concatenate + rerank", "score + rerank"]
output = {}

for q in questions:
    print(f"\n{'+'*15}\nЗапрос:\n{q}")
    candidate_sources = set()
    for mode, reranker in configs:
        label = f"{mode}{' + rerank' if reranker else ''}"
        start = time.time()
        results = hybrid_search(q, n_results=25, mode=mode, use_reranker=reranker)
        end = time.time() - start

        sources = [meta['source'] for _, meta in results]
        print(f"\n[{label}] {end:.2f} сек")
        for i, s in enumerate(sources, 1):
            print(f" {i}: {s}")

        if label in MODES:
            candidate_sources.update(sources)
        
    output[q] = sorted(candidate_sources)

with open("ground_truth_candidates.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2) 