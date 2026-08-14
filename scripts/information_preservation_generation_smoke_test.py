# ruff: noqa: RUF001

import os

from dotenv import load_dotenv

from chunking_metrics.preparations import api

SEGMENT = (
    "В 2024 году библиотека открыла новый читальный зал на втором этаже. "
    "Зал рассчитан на 120 посетителей и работает ежедневно с девяти утра до восьми вечера. "
    "Для работы в зале читателям требуется действующий библиотечный билет."
)
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is required for the API smoke test")

    true_statement, false_statements = api.generate_information_preservation_statements(
        SEGMENT,
        model_name=MODEL_NAME,
        api_key=api_key,
        base_url=BASE_URL,
        max_new_tokens=512,
    )

    assert true_statement.strip()
    assert len(false_statements) == 3
    assert all(statement.strip() for statement in false_statements)
    assert len(set(false_statements)) == 3
    assert true_statement not in false_statements

    print(f"true_statement={true_statement}")
    for index, statement in enumerate(false_statements, start=1):
        print(f"false_statement_{index}={statement}")


if __name__ == "__main__":
    main()
