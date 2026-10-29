import json, ollama

LLM = "qwen3_5:latest"

def get_route(user_q: str):
    """Классификация запроса:
    "DOCS", "CODE" или "BOTH" 
    """
    try:
        response = ollama.chat(
            model=LLM,
            messages=[{
                "role": "system", "content": 'Ты - маршрутизатор для RAG-системы. Твоя задача определить куда направить запрос пользователя: DOCS - Документация, CODE - Исходный код или оба варианта - BOTH, только если пользователь явно просит и теорию, и кусок кода. Отвечай строго в формате JSON с одним ключом route и значением "DOCS", "CODE" или "BOTH". Пример ответа: {"route": "CODE"} '},
                {"role": "user", "content": f"Вопрос пользователя: {user_q}"}
            ],
            think=False,
            format="json",
            options={"temperature": 0.0, "num_predict": 20}
        )
        result = json.loads(response["message"]["content"])
        route = result.get("route", "BOTH").upper()
        return route
    except Exception as e:
        print(f"Ошибка роутера: {e}. Ответ нейросети: {response}")
        return "BOTH"

if __name__ == "__main__":
    test_queries = [
        "Как настроить параметры обучения?",
        "Где реализован класс YOLO?",
        "Объясни архитектуру и покажи код"
    ]
    for q in test_queries:
        print(f"Вопрос: {q} -> Маршрут: {get_route(q)}")