import numpy as np
from dotenv import load_dotenv

from chunking_metrics import calculate_embeddings, concept_unity, generate_statements

CHUNK = (
    "Чанкирование делит исходный документ на фрагменты для последующего поиска. "
    "Смысловая цельность помогает векторной модели точнее кодировать содержание фрагмента. "
    "Утверждения, извлечённые из цельного фрагмента, обычно описывают близкие понятия."
)
STATEMENT_COUNT = 5


def main() -> None:
    load_dotenv()
    statements = generate_statements(
        CHUNK,
        statement_count=STATEMENT_COUNT,
        device="cpu",
    )
    statement_embeddings = calculate_embeddings(statements, device="cpu")
    score = concept_unity(statement_embeddings)

    assert len(statements) == STATEMENT_COUNT
    assert all(statement.strip() for statement in statements)
    assert statement_embeddings.ndim == 2
    assert statement_embeddings.shape[0] == STATEMENT_COUNT
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0

    for index, statement in enumerate(statements, start=1):
        print(f"{index}. {statement}")
    print(f"embeddings_shape={statement_embeddings.shape}")
    print(f"concept_unity={score:.6f}")


if __name__ == "__main__":
    main()
