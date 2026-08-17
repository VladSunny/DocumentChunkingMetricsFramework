from itertools import pairwise

import numpy as np

from chunking_metrics.metrics import boundary_clarity, chunk_score, semantic_dispersion
from chunking_metrics.preparations import local
import dotenv

dotenv.load_dotenv()

CHUNKS = [
    "Чанкирование делит документ на фрагменты для последующего поиска.",
    "Семантическое рассеяние оценивает разнообразие эмбеддингов чанков.",
    "Перплексия соседних фрагментов помогает оценить логическую независимость границ.",
]


def main() -> None:
    chunk_embeddings = local.calculate_embeddings(CHUNKS)
    semantic_dispersion_score = semantic_dispersion(chunk_embeddings)

    unconditional_perplexities = np.array(
        [local.calculate_perplexity(chunk) for chunk in CHUNKS],
        dtype=np.float64,
    )
    conditional_perplexities = np.array(
        [
            local.calculate_perplexity(current_chunk, context=previous_chunk)
            for previous_chunk, current_chunk in pairwise(CHUNKS)
        ],
        dtype=np.float64,
    )
    logical_independence_by_boundary = boundary_clarity(
        unconditional_perplexities,
        conditional_perplexities,
    )
    logical_independence_score = float(np.mean(logical_independence_by_boundary))
    score = chunk_score(logical_independence_score, semantic_dispersion_score)

    assert chunk_embeddings.ndim == 2
    assert chunk_embeddings.shape[0] == len(CHUNKS)
    assert np.all(np.isfinite(chunk_embeddings))
    assert unconditional_perplexities.shape == (len(CHUNKS),)
    assert conditional_perplexities.shape == (len(CHUNKS) - 1,)
    assert np.all(np.isfinite(unconditional_perplexities))
    assert np.all(np.isfinite(conditional_perplexities))
    assert len(logical_independence_by_boundary) == len(CHUNKS) - 1
    assert np.all(np.isfinite(logical_independence_by_boundary))
    assert np.isfinite(semantic_dispersion_score)
    assert np.isfinite(logical_independence_score)
    assert np.isfinite(score)
    assert np.isclose(
        score,
        0.3 * logical_independence_score + 0.7 * semantic_dispersion_score,
    )

    print(f"embeddings_shape={chunk_embeddings.shape}")
    print(f"logical_independence={logical_independence_score:.6f}")
    print(f"semantic_dispersion={semantic_dispersion_score:.6f}")
    print(f"chunk_score={score:.6f}")


if __name__ == "__main__":
    main()
