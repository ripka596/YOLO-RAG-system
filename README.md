# RAG-система для документации YOLO

Retrieval-Augmented Generation для поиска по официальной документации Ultralytics YOLO.

## Стек
- LLM: Qwen3.5-4B-UD_Q5-K-XL
- Эмбеддер: Qwen3-Embedding-0.6B
- Реранкер: Qwen3-Reranker-0.6B
- Векторная БД: ChromaDB
- Полнотекстовый поиск: BM25 (rank-bm25)
- Интерфейс: Gradio

## Установка

1. Установить зависимости:
   `pip install -r requirements.txt`

2. Установить Ollama и модель:
    - Скачать https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
    - Создать папку с .GGUF и файлом Modelfile с текстом 
    ```
    FROM ./путь_к_файлу/model-name.gguf
    ```
    - Импортировать модель в Ollama:
    `ollama create qwen3_5 -f ./Modelfile`

3. Скачать документацию и построить индексы:
   `python updateData.py`

4. Запустить веб-интерфейс:
   `python main.py`

## Структура
- `main.py` - RAG-пайплайн + Gradio
- `ingest.py` - чанкинг и загрузка в ChromaDB
- `ingest_bm25.py` - построение BM25-индекса
- `updateData.py` - скачивание документации и построение индексов
- `tests.py` - прогон конфигураций

## Метрики
| **Метод** | **Recall@5** | **MRR** |
| --- | --- | --- |
| RRF + rerank | 0.644 | 0.917 |
| Score + rerank | 0.658 | 0.867 |
| Concatenate + rerank | 0.661 | 0.833 |

### Смена метода производится в `main.py` в функции `hybrid_search()`
### Параметр - `mode="rrf"`, доступны "rrf", "concatenate", "score"
