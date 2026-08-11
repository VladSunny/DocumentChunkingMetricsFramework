import numpy as np

from chunking_metrics.metrics import contextual_coherence, intrachunk_cohesion
from chunking_metrics.preparation import calculate_embeddings

SENTENCES = [
    "Иван подписал договор на следующий день.",
    "Документ вступил в силу сразу после подписания.",
    "Сегодня в городе ожидается солнечная погода.",
]


def main() -> None:
    sentence_embeddings = calculate_embeddings(SENTENCES, device="cpu")
    single_embedding = calculate_embeddings(SENTENCES[0], device="cpu")

    assert sentence_embeddings.shape[0] == len(SENTENCES)
    assert sentence_embeddings.ndim == 2
    assert single_embedding.shape == (sentence_embeddings.shape[1],)
    assert np.all(np.isfinite(sentence_embeddings))
    assert np.allclose(np.linalg.norm(sentence_embeddings, axis=1), 1.0, atol=1e-5)
    assert np.allclose(single_embedding, sentence_embeddings[0], atol=1e-5)

    cohesion = intrachunk_cohesion([sentence_embeddings])
    coherence = contextual_coherence(single_embedding, sentence_embeddings[1:])
    assert np.isfinite(cohesion)
    assert np.isfinite(coherence)

    print(f"embeddings_shape={sentence_embeddings.shape}")
    print(f"intrachunk_cohesion={cohesion:.6f}")
    print(f"contextual_coherence={coherence:.6f}")


if __name__ == "__main__":
    main()
