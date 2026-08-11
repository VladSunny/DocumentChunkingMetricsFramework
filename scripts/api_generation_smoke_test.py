import numpy as np
from dotenv import load_dotenv
import os

from chunking_metrics import generate_statements_api, calculate_embeddings, concept_unity, generate_questions_api

CHUNK = (
    "Чанкирование делит исходный документ на фрагменты для последующего поиска. "
    "Смысловая цельность помогает векторной модели точнее кодировать содержание фрагмента. "
    "Утверждения, извлечённые из цельного фрагмента, обычно описывают близкие понятия."
    "Кукареку Кукареку у куриц большие яйца"
    "у меня большой-большой дом"
)

load_dotenv()
api_key = os.getenv("API_KEY")


statements = generate_statements_api(
    chunk=CHUNK,
    model_name="deepseek-v4-flash",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    statement_count=4,
)
statement_embeddings = calculate_embeddings(statements, device="cpu")
score = concept_unity(statement_embeddings)

for index, statement in enumerate(statements, start=1):
    print(f"{index}. {statement}")
print(f"embeddings_shape={statement_embeddings.shape}")
print(f"concept_unity={score:.6f}")

questions = generate_questions_api(
    chunk=CHUNK,
    model_name="deepseek-v4-flash",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    question_count=4,
)

print(questions)
