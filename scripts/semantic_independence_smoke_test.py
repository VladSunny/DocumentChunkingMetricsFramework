import numpy as np

from chunking_metrics.metrics import semantic_independence
from chunking_metrics.preparation import calculate_embeddings, generate_answers_local

CHUNK = (
    "Чанкирование делит исходный документ на фрагменты для последующего поиска. "
    "Смысловая цельность помогает векторной модели точнее кодировать содержание фрагмента."
)
QUESTIONS = [
    "Для чего документ делят на фрагменты?",
    "Что помогает векторной модели точнее кодировать содержание фрагмента?",
]
ADDITIONAL_CHUNKS_BY_QUESTION = [
    ["Полученные фрагменты индексируют и находят по запросу пользователя."],
    ["Семантически связанный текст обычно образует более устойчивое представление."],
]


def main() -> None:
    standalone_answers = generate_answers_local(QUESTIONS, CHUNK)
    contextual_answers = generate_answers_local(
        QUESTIONS,
        CHUNK,
        additional_chunks_by_question=ADDITIONAL_CHUNKS_BY_QUESTION,
    )
    standalone_embeddings = calculate_embeddings(standalone_answers, device="cpu")
    contextual_embeddings = calculate_embeddings(contextual_answers, device="cpu")
    score = semantic_independence(standalone_embeddings, contextual_embeddings)

    assert len(standalone_answers) == len(QUESTIONS)
    assert len(contextual_answers) == len(QUESTIONS)
    assert standalone_embeddings.shape == contextual_embeddings.shape
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0

    for question, standalone_answer, contextual_answer in zip(
        QUESTIONS, standalone_answers, contextual_answers, strict=True
    ):
        print(f"question={question}")
        print(f"standalone={standalone_answer}")
        print(f"contextual={contextual_answer}")
    print(f"semantic_independence={score:.6f}")


if __name__ == "__main__":
    main()
