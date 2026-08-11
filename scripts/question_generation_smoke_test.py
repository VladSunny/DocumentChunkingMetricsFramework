from dotenv import load_dotenv

from chunking_metrics.preparation import generate_questions_local

CHUNK = (
    "Чанкирование делит исходный документ на фрагменты для последующего поиска. "
    "Смысловая независимость показывает, достаточно ли одного фрагмента для ответа. "
    "Вопросы для проверки должны быть отвечаемы только по содержимому выбранного фрагмента."
)
QUESTION_COUNT = 5


def main() -> None:
    load_dotenv()
    questions = generate_questions_local(
        CHUNK,
        question_count=QUESTION_COUNT,
        device="cpu",
    )

    assert len(questions) == QUESTION_COUNT
    assert all(question.strip() for question in questions)

    for index, question in enumerate(questions, start=1):
        print(f"{index}. {question}")


if __name__ == "__main__":
    main()
