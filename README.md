<div align="center">

# RAG-система для документации YOLO

**Гибридный поиск, роутер, реранкер**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)
![Qwen](https://img.shields.io/badge/Qwen-%236950EF?style=flat&logo=qwen&logoColor=white)

</div>

## 📖 О проекте

Локальный RAG-ассистент по экосистеме Ultralytics YOLO, объединяющий поиск по официальной документации и исходному коду (Python и C++).

- **Маршрутизация запросов:** LLM определяет, куда отправлять вопрос - в код, документацию или в оба источника, экономя время и контекст.
- **AST-чанкинг:** Код разбивается по функциям, классам и методам с помощью `tree-sitter`.
- **Гибридный поиск + финальное слияние мультизапросов:** Слияние векторного поиска и BM25. Для финального слияния реализован RRF или Round Robin.
- **Cross-Encoder реранкинг:** Пересортировка кандидатов перед подачей в контекст LLM.
<div align="center">

![Interface](image.png)

</div>

## 🤖 Стек
- **LLM:** Qwen3.5-4B-UD_Q5-K-XL
- **Эмбеддер:** Qwen3-Embedding-0.6B
- **Реранкер:** Qwen3-Reranker-0.6B
- **Векторная БД:** ChromaDB
- **Полнотекстовый поиск:** BM25 (rank-bm25)
- **Парсинг кода:** tree-sitter (c++, python)
- **Интерфейс:** Gradio

**Требования к железу:** Проект протестирован на NVIDIA RTX 4070 SUPER (12 GB VRAM). Для инференса LLM и построения индексов настоятельно рекомендуется наличие дискретной видеокарты.

## ⚒️ Установка

1. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

2. Установить Ollama и модель:
    - Скачать веса [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
    - Создать папку с .GGUF и файлом Modelfile с текстом 
    ```bash
    FROM ./путь_к_файлу/model-name.gguf
    ```
    - Импортировать модель в Ollama:
    ```bash
    ollama create qwen3_5 -f ./Modelfile
    ```

3. Скачать документацию и построить индексы:
   ```bash
   python src/updateData.py
   ```

4. Запустить веб-интерфейс:
   ```bash
   python src/main.py
   ```

## ⚙️ Структура
- `src/main.py` - RAG-пайплайн + Gradio
- `src/router.py` - LLM-маршрутизатор запросов
- `src/ast_chunker.py` - AST-парсер кода
- `src/ingest_docs.py` - чанкинг и загрузка в ChromaDB
- `src/ingest_code.py` - индексация кода в ChromaDB и BM25
- `src/updateData.py` - скачивание документации и построение индексов
- `src/utils.py` - дополнительные инструменты
- `src/metrics.py` - генерация тестового датасета и расчёт метрик
- `src/quick_check.py` - тестирование мультизапросов и методов слияния
- `src/sweep.py` - расчёт оптимальных параметров 
- `metrics/` - ground_truth и таблицы с результатами

## 🧮 Метрики
![Таблица метрик](metrics/metrics.png)
Лучший результат точности без мультизапросов - score + rerank (score>=6) без роутера (Recall: 0.733, MRR: 0.653 при 4.1 чанках в LLM)

### Тестирование Multi-Query и RRF/Round Robin (score>=0)
![Влияние Multi-Query](<metrics/TEST mq, router, RR or RRF.png>)
Альтернативные вопросы улучшают качество ответов, но значительно (~1.5 раза) увеличивают время поиска.
Алгоритм round robin также даёт выигрыш над RRF в некоторых конфигурациях.
Лучший результат точности - score + rerank при rerank_k=75 с мультизапросами, без роутера. (Recall@8: 0.867, MRR: 0.704, Время: 2.630 c). Рекомендуется: concatenate + rerank + Round Robin. (Recall@8: 0.800, MRR: 0.723, Время: 0.940 c)

## 🔧 Настройка поиска в `main.py`
- `n_results` - количество чанков для поиска
- `mode` - алгоритм слияния: `"rrf"`, `"concatenate"`, `"score"`, `"vector"`, `"bm25"`
- `use_reranker`: `True` / `False` - параметр использования реранкера
- `score_threshold`: `None` / `k` - отсечение по скору реранкера (по умолчанию: 0). Если `None`, используется `top_k`.
- `top_k` - количество чанков после реранкера (по умолчанию: 8)
- `rerank_k` - количество чанков до реранкера
- `multi_query`: `True` / `False` - использование альтернативных вопросов
- `db_target` - выбор целевой базы данных ("docs" или "code")
- `score_fuze`: `True` / `False` - финальное слияние мультизапросов (`routing()`)

### Стандартная настройка
`main.py` использует следующие параметры поиска по умолчанию (`routing()`):
- `n_results`: 25 (docs)
- `n_results`: 25 (code)
- `mode`: concatenate
- `use_reranker`: True
- `score_threshold`: 0
- `top_k`: 8
- `rerank_k`: 20
- `use_router`: False
- `multi-query`: True
- Финальное слияние - Round Robin (score_fuze = True)