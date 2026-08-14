# ruff: noqa: RUF001

import os

from dotenv import load_dotenv

from chunking_metrics.preparations import api

TRUE_STATEMENT = "Читальный зал рассчитан на 120 посетителей."
FALSE_STATEMENTS = [
    "Читальный зал рассчитан на 12 посетителей.",
    "Читальный зал рассчитан на 200 посетителей.",
    "Читальный зал рассчитан на 320 посетителей.",
]
RELEVANT_CHUNKS = [
    (
        "В 2024 году библиотека открыла новый читальный зал на втором этаже. "
        "Зал рассчитан на 120 посетителей и работает ежедневно с девяти утра до восьми вечера. "
        "Для работы в зале читателям требуется действующий библиотечный билет."
    )
]
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
SEED = 42


def main() -> None:
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is required for the API smoke test")

    score = api.evaluate_information_preservation(
        TRUE_STATEMENT,
        FALSE_STATEMENTS,
        RELEVANT_CHUNKS,
        model_name=MODEL_NAME,
        api_key=api_key,
        base_url=BASE_URL,
        seed=SEED,
    )

    print(f"information_preservation_test={score}")
    assert score == 1, "the model did not select the true statement"


if __name__ == "__main__":
    main()
