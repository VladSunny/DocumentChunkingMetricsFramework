from dotenv import load_dotenv

from chunking_metrics.preparations import local

CHUNK = (
    "Чанкирование делит исходный документ на фрагменты для последующего поиска. "
    "Смысловая независимость показывает, достаточно ли одного фрагмента для ответа. "
    "Вопросы для проверки должны быть отвечаемы только по содержимому выбранного фрагмента."
)
QUESTION_COUNT = 5


def main() -> None:
    load_dotenv()
    questions = local.generate_questions(
        CHUNK,
        question_count=QUESTION_COUNT,
        max_regenerations=3,
        # device="",
    )

    assert len(questions) == QUESTION_COUNT
    assert all(question.strip() for question in questions)

    for index, question in enumerate(questions, start=1):
        print(f"{index}. {question}")


if __name__ == "__main__":
    main()
