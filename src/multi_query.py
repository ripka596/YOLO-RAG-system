from pydantic import BaseModel, Field
import ollama

class QueryRequests(BaseModel):
    queries: list[str] = Field(description="Список из 3 альтернативных вопросов")

def multi_query(user_q: str):
    """
    Создание нескольких вопросов
    """
    try:
        response = ollama.chat(
            model="qwen3_5:latest",
            messages=[
                {"role": "system", "content": "Напиши 3 альтернативных вопроса, которые помогут найти ответ в документации YOLO. Хотя бы один из них должен быть на английском языке. Английские названия только на английском языке. Ответь в формате JSON"},
                {"role": "user", "content": f"Вопрос: {user_q}"}
            ],
            think=False,
            format=QueryRequests.model_json_schema(),
            options={'temperature': 0.4}
        )

        validated_response = QueryRequests.model_validate_json(response['message']['content'])
        return [user_q] + validated_response.queries
    except Exception as e:
        print(f"Ошибка MultiQuery: {e}")
        return [user_q]

if __name__ == "__main__":
    user_question = "Как использовать YOLO для обнаружения объектов на видео?"
    alternative_questions = multi_query(user_question)
    print("Сгенерированные альтернативные вопросы:")
    for i, question in enumerate(alternative_questions):
        print(f"{i + 1}. {question}")