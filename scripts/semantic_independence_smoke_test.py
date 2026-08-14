import numpy as np

from chunking_metrics.metrics import semantic_independence
from chunking_metrics.preparations import local

CHUNK = (
    "Чанкирование делит исходный документ на фрагменты для последующего поиска. "
    "Смысловая цельность помогает векторной модели точнее кодировать содержание фрагмента."
)
QUESTIONS = [
    "Для чего документ делят на фрагменты?",
    "Что помогает векторной модели точнее кодировать содержание фрагмента?",
]
CANDIDATE_CHUNKS = [
    "Полученные фрагменты индексируют и находят по запросу пользователя.",
    "Семантически связанный текст обычно образует более устойчивое представление.",
    "Дополнительный контекст может уточнить смысл найденного фрагмента.",
    "Размер чанка влияет на количество текста, передаваемого языковой модели.",
]


def main() -> None:
    additional_chunks_by_question = local.retrieve_relevant_chunks(QUESTIONS, CANDIDATE_CHUNKS)
    standalone_answers = local.generate_answers(QUESTIONS, CHUNK)
    contextual_answers = local.generate_answers(
        QUESTIONS,
        CHUNK,
        additional_chunks_by_question=additional_chunks_by_question,
    )
    standalone_embeddings = local.calculate_embeddings(standalone_answers, device="cpu")
    contextual_embeddings = local.calculate_embeddings(contextual_answers, device="cpu")
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
