import os

import numpy as np
from dotenv import load_dotenv

from chunking_metrics.metrics import concept_unity, semantic_independence
from chunking_metrics.preparations import api, local

CHUNK = (
    "Чанкирование делит исходный документ на фрагменты для последующего поиска. "
    "Смысловая цельность помогает векторной модели точнее кодировать содержание фрагмента. "
    "Утверждения, извлечённые из цельного фрагмента, обычно описывают близкие понятия."
)
ADDITIONAL_CHUNK = "Семантически связанные фрагменты помогают дополнить контекст ответа на вопрос."
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
ITEM_COUNT = 4


def main() -> None:
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is required for the API smoke test")

    statements = api.generate_statements(
        chunk=CHUNK,
        model_name=MODEL_NAME,
        api_key=api_key,
        base_url=BASE_URL,
        statement_count=ITEM_COUNT,
    )
    statement_embeddings = local.calculate_embeddings(statements, device="cpu")
    unity_score = concept_unity(statement_embeddings)

    questions = api.generate_questions(
        chunk=CHUNK,
        model_name=MODEL_NAME,
        api_key=api_key,
        base_url=BASE_URL,
        question_count=ITEM_COUNT,
    )
    standalone_answers = api.generate_answers(
        questions,
        CHUNK,
        model_name=MODEL_NAME,
        api_key=api_key,
        base_url=BASE_URL,
        max_new_tokens=1024,
    )
    contextual_answers = api.generate_answers(
        questions,
        CHUNK,
        model_name=MODEL_NAME,
        api_key=api_key,
        base_url=BASE_URL,
        additional_chunks_by_question=[[ADDITIONAL_CHUNK] for _ in questions],
        max_new_tokens=1024,
    )
    standalone_embeddings = local.calculate_embeddings(standalone_answers, device="cpu")
    contextual_embeddings = local.calculate_embeddings(contextual_answers, device="cpu")
    independence_score = semantic_independence(standalone_embeddings, contextual_embeddings)

    assert len(standalone_answers) == len(questions)
    assert len(contextual_answers) == len(questions)
    assert np.isfinite(unity_score)
    assert np.isfinite(independence_score)
    assert 0.0 <= independence_score <= 1.0

    for index, statement in enumerate(statements, start=1):
        print(f"statement_{index}={statement}")
    for question, standalone_answer, contextual_answer in zip(
        questions, standalone_answers, contextual_answers, strict=True
    ):
        print(f"question={question}")
        print(f"standalone={standalone_answer}")
        print(f"contextual={contextual_answer}")
    print(f"concept_unity={unity_score:.6f}")
    print(f"semantic_independence={independence_score:.6f}")


if __name__ == "__main__":
    main()
