import os

import numpy as np
from dotenv import load_dotenv

from chunking_metrics.preparations import api

TEXTS = [
    "Чанкирование делит документ на фрагменты для последующего поиска.",
    "Качественные эмбеддинги помогают находить семантически близкие фрагменты.",
]
MODEL_NAME = "text-embedding-v4"
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("EMB_API_KEY")
    if not api_key:
        raise RuntimeError("EMB_API_KEY is required for the API smoke test")
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME must be set in the smoke test")
    if not BASE_URL:
        raise RuntimeError("BASE_URL must be set in the smoke test")

    embeddings = api.calculate_embeddings(
        TEXTS,
        model_name=MODEL_NAME,
        api_key=api_key,
        base_url=BASE_URL,
        batch_size=2,
    )
    single_embedding = api.calculate_embeddings(
        TEXTS[0],
        model_name=MODEL_NAME,
        api_key=api_key,
        base_url=BASE_URL,
    )

    print(embeddings)
    print(single_embedding)

    # assert embeddings.ndim == 2
    # assert embeddings.shape[0] == len(TEXTS)
    # assert embeddings.dtype == np.float32
    # assert single_embedding.shape == (embeddings.shape[1],)
    # assert single_embedding.dtype == np.float32
    # assert np.all(np.isfinite(embeddings))
    # assert np.all(np.isfinite(single_embedding))
    # assert np.allclose(single_embedding, embeddings[0], atol=1e-5)

    print(f"embeddings_shape={embeddings.shape}")
    print(f"single_embedding_shape={single_embedding.shape}")


if __name__ == "__main__":
    main()
