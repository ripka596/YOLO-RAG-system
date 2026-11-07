import json, ollama
from pydantic import BaseModel
from typing import Literal

LLM = "qwen3_5:latest"

class Response(BaseModel):
    route: Literal["DOCS", "CODE", "BOTH"]

def get_route(user_q: str):
    """Классификация запроса:
    "DOCS", "CODE" или "BOTH" 
    """
    try:
        response = ollama.chat(
            model=LLM,
            messages=[{
                "role": "system", "content": '''Ты - маршрутизатор для RAG-системы.
                Есть две базы:
                - "DOCS" (официальная документация, туториалы, метрики).
                - "CODE" (исходный код на Python и C++).
                Определи маршрут для вопроса пользователя.
                Правила:
                - Вопрос про установку, обучение, датасеты, теорию -> "DOCS".
                - Вопрос про функции, реализацию или явно про код -> "CODE".
                - Если вопрос требует понимания и концепции, и кода или если ты сомневаешься -> "BOTH".
                Отвечай строго в формате JSON с одним ключом route и значением "DOCS", "CODE" или "BOTH". 
                Примеры ответа:
                Вопрос: "Как установить ultralytics?" -> {"route": "DOCS"}
                Вопрос: "Где находится класс YOLOv8Predictor?" -> {"route": "CODE"}
                Вопрос: "Какие параметры принимает функция forward?" -> {"route": "CODE"}
                Вопрос: "Как работает NMS?" -> {"route": "BOTH"}'''},
                {"role": "user", "content": f"Вопрос пользователя: {user_q}"}
            ],
            think=False,
            format=Response.model_json_schema(),
            options={"temperature": 0.0, "num_predict": 20}
        )

        route = Response.model_validate_json(response["message"]["content"]).model_dump()["route"]
        return route
    
    except Exception as e:
        print(f"Ошибка внутри роутера: {e}")
        return "BOTH"

if __name__ == "__main__":
    test_queries = [
        "Как настроить параметры обучения?",
        "Где реализован класс YOLO?",
        "Объясни архитектуру и покажи код"
    ]
    for q in test_queries:
        print(f"Вопрос: {q} -> Маршрут: {get_route(q)}")