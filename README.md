# Document Chunking Metrics Framework

`chunking_metrics` is a Python library for reference-free evaluation of document chunking.
It measures structural and semantic properties of chunks without requiring gold boundaries,
reference chunks, annotated questions, or human relevance labels.

## Metric implementation status

Five of the eleven metrics in the project plan are currently implemented:

- [x] [Size Compliance (SC)](docs/chunking_metrics.md#1-size-compliance-sc) — `size_compliance`
- [ ] [Block Integrity (BI)](docs/chunking_metrics.md#2-block-integrity-bi)
- [x] [Intrachunk Cohesion (ICC)](docs/chunking_metrics.md#3-intrachunk-cohesion-icc) — `intrachunk_cohesion`
- [x] [Contextual Coherence (DCC)](docs/chunking_metrics.md#4-contextual-coherence-dcc) — `contextual_coherence`
- [ ] [Coreference Integrity (RC)](docs/chunking_metrics.md#5-coreference-integrity-rc)
- [x] [Boundary Clarity (BC)](docs/chunking_metrics.md#6-boundary-clarity-bc) — `boundary_clarity`
- [ ] [ChunkScore](docs/chunking_metrics.md#7-chunkscore)
- [x] [HOPE Concept Unity](docs/chunking_metrics.md#8-hope-concept-unity) — `concept_unity`
- [ ] [HOPE Semantic Independence](docs/chunking_metrics.md#9-hope-semantic-independence)
- [ ] [HOPE Information Preservation](docs/chunking_metrics.md#10-hope-information-preservation)
- [ ] [HOPE Aggregate](docs/chunking_metrics.md#11-hope-aggregate)

A checked metric is implemented, exported from the package, and covered by tests. Functions that
exist only as `NotImplementedError` placeholders are not considered implemented.

## Overview

The library evaluates complementary aspects of a chunking strategy:

- size compliance with a chosen granularity;
- semantic cohesion within individual chunks;
- semantic coherence between a chunk and its local context;
- independence across neighboring chunk boundaries.

These metrics are not interchangeable and should normally be considered together. For example, a
fixed-size strategy can receive a perfect Size Compliance score while splitting semantic or
structural dependencies at poor locations.

The package also provides helpers for generating statements, normalized sentence embeddings, and
causal-language-model perplexities. Detailed definitions, formulas, input requirements, and
limitations for both implemented and planned metrics are available in the
[metric plan](docs/chunking_metrics.md).

## Installation

The package requires Python 3.10 or newer. Install the project and its locked development
dependencies from a local checkout with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev
```

Run Python commands inside the environment with `uv run`, for example:

```bash
uv run python -c "import chunking_metrics; print(chunking_metrics.__all__)"
```

## Use in another project

You can install the library directly from a local checkout without publishing it to a package
index. For the commands below, assume the library and consuming project are sibling directories:

```text
workspace/
├── DocumentChunkingMetricsFramework/
└── my-project/
```

### Editable local dependency

Use an editable dependency while developing the library and the consuming project together. From
the `my-project/` directory, run:

```bash
uv add --editable ../DocumentChunkingMetricsFramework
```

`uv` records the local source in the consuming project's `pyproject.toml` and installs it into the
project environment. Changes made under `src/chunking_metrics/` are then available without
rebuilding the package.

For a project managed with `pip`, activate its virtual environment and run:

```bash
python -m pip install -e ../DocumentChunkingMetricsFramework
```

Import the package by its Python module name, `chunking_metrics`. Verify an `uv` installation with:

```bash
uv run python -c "from chunking_metrics import size_compliance; print(size_compliance([100], 50, 150))"
```

For an activated `pip` environment, use the same check without the `uv run` prefix:

```bash
python -c "from chunking_metrics import size_compliance; print(size_compliance([100], 50, 150))"
```

### Built wheel

Use a wheel when the consuming project should depend on a fixed build instead of live source files.
First build the library from this repository:

```bash
uv build
```

Then add the generated wheel from the consuming project's directory:

```bash
uv add ../DocumentChunkingMetricsFramework/dist/chunking_metrics-0.1.0-py3-none-any.whl
```

The equivalent installation in an activated `pip` environment is:

```bash
python -m pip install ../DocumentChunkingMetricsFramework/dist/chunking_metrics-0.1.0-py3-none-any.whl
```

## Quick start

### Array-only metric

Size Compliance operates directly on precomputed chunk lengths and does not require a model:

```python
from chunking_metrics import size_compliance

chunk_lengths = [180, 240, 510, 320]
score = size_compliance(chunk_lengths, min_size=200, max_size=500)

print(score)  # 0.5
```

### Model-backed metrics

The preparation helpers can produce all model-dependent inputs needed by the currently implemented
semantic and boundary metrics:

```python
import numpy as np

from chunking_metrics import (
    boundary_clarity,
    calculate_embeddings,
    calculate_perplexity,
    concept_unity,
    contextual_coherence,
    generate_statements,
    intrachunk_cohesion,
)

sentence_groups = [
    ["Чанкирование делит документ на фрагменты.", "Размер фрагмента влияет на поиск."],
    ["Когезия измеряет смысловую связность.", "Связанные предложения стоит хранить вместе."],
    ["Ясность границ измеряет зависимость.", "Сильные границы разделяют независимые фрагменты."],
]
chunks = [" ".join(sentences) for sentences in sentence_groups]

# Embed sentences for ICC, and whole chunks for DCC.
sentence_embeddings = [calculate_embeddings(sentences) for sentences in sentence_groups]
chunk_embeddings = calculate_embeddings(chunks)

icc = intrachunk_cohesion(sentence_embeddings)
dcc = contextual_coherence(chunk_embeddings[1], chunk_embeddings[[0, 2]])

# HOPE Concept Unity uses statements generated from one chunk.
statements = generate_statements(chunks[0])
statement_embeddings = calculate_embeddings(statements)
cu = concept_unity(statement_embeddings)

# BC expects one unconditional perplexity per chunk and one conditional
# perplexity for every chunk after the first.
unconditional = np.array([calculate_perplexity(chunk) for chunk in chunks])
conditional = np.array(
    [
        calculate_perplexity(current, context=previous)
        for previous, current in zip(chunks, chunks[1:])
    ]
)
bc = boundary_clarity(unconditional, conditional)

print(
    {
        "intrachunk_cohesion": icc,
        "contextual_coherence": dcc,
        "concept_unity": cu,
        "boundary_clarity": bc,
    }
)
```

The first invocation of each preparation helper may download its default model. Pass `model_name`
to a helper to use a different compatible Hugging Face model.

## Public API

### Metrics

| Function | Inputs | Returns |
| --- | --- | --- |
| `size_compliance(lengths, min_size, max_size)` | Chunk lengths and an inclusive valid range | Fraction of chunks inside the range |
| `intrachunk_cohesion(embs)` | One `(sentence_count, embedding_dim)` matrix per chunk | Mean sentence-to-chunk-centroid cosine similarity |
| `contextual_coherence(chunk_embs, context_embs)` | One `(embedding_dim,)` chunk vector and an `(item_count, embedding_dim)` context matrix | Cosine similarity between the chunk and mean context vector |
| `boundary_clarity(uncond_ppls, cond_ppls)` | `K` unconditional and `K - 1` preceding-context-conditioned perplexities | Mean conditional-to-unconditional perplexity ratio |
| `concept_unity(statements_embs)` | One `(statement_count, embedding_dim)` matrix | Mean clipped pairwise cosine similarity between generated statements |

### Input preparation

| Function | Purpose | Returns |
| --- | --- | --- |
| `calculate_embeddings(texts, model_name=..., device=None, batch_size=32)` | Encode one text or a sequence with a Sentence Transformers model | A normalized `float32` vector or matrix |
| `calculate_perplexity(text, model_name=..., context=None, device=None)` | Score target text with an optional preceding context excluded from the loss | Causal-language-model perplexity as `float` |
| `generate_questions(chunk, model_name=..., prompt=..., question_count=5, temperature=0.7, max_new_tokens=256, device=None)` | Generate an exact number of questions answerable from one chunk | A list of `question_count` non-empty strings |
| `generate_statements(chunk, model_name=..., prompt=..., statement_count=5, temperature=0.7, max_new_tokens=256, device=None)` | Extract an exact number of factual statements from one chunk | A list of `statement_count` non-empty strings |

## Model and runtime notes

- Embeddings default to `cointegrated/rubert-tiny2`, perplexity to
  `ai-forever/rugpt3small_based_on_gpt2`, and statement and question generation to
  `Qwen/Qwen2.5-1.5B-Instruct`. These defaults support Russian-language text; use `model_name` to
  select compatible models for other languages.
- When `device` is omitted, CUDA is preferred, followed by Apple MPS and CPU. Explicit `cpu`,
  `cuda[:index]`, and `mps` values are supported when available.
- The most recently loaded embedding model and causal language model are cached for the lifetime of
  the Python process. Perplexity, statement generation, and question generation share the
  causal-language-model cache, so switching model identifiers replaces its cached entry.
- Embedding inputs that exceed the model limit are truncated after a warning. For perplexity,
  excessive preceding context is truncated from the left; a target that cannot fit raises
  `ValueError`.
- Statement and question generation are stochastic, perform no retries, and require a tokenizer
  with a chat template. The model must return a JSON array containing exactly the requested number
  of non-empty strings. Chunks are not truncated; a prompt and response budget that exceed the
  model context window raise `ValueError`.
- The default statement prompt is defined in `chunking_metrics.prompts`. Supply a custom format
  string through `prompt`; it must contain both `{chunk}` and `{statement_count}`. The chunk is
  substituted as a JSON string, and literal braces in the template must be escaped as `{{` and
  `}}`:

  ```python
  custom_prompt = (
      "Return exactly {statement_count} short claims supported by {chunk} as a JSON array."
  )
  statements = generate_statements(chunks[0], prompt=custom_prompt)
  ```

- The default question prompt is also defined in `chunking_metrics.prompts`. A custom question
  prompt must contain `{chunk}` and `{question_count}`:

  ```python
  custom_prompt = "Return exactly {question_count} questions answerable from {chunk} as a JSON array."
  questions = generate_questions(chunks[0], prompt=custom_prompt)
  ```

- Preparation helpers reject invalid arguments with `TypeError` or `ValueError`. Metric functions
  currently return `0.0` for the invalid shapes or ranges that they explicitly validate.

Run the complete Concept Unity pipeline manually with the built-in Russian example:

```bash
uv run python scripts/concept_unity_smoke_test.py
```

The first run downloads both generation and embedding models and can take considerably longer than
the isolated unit tests. The smoke script is intentionally not part of the pytest suite.

Run question generation manually with the default model:

```bash
uv run python scripts/question_generation_smoke_test.py
```

This smoke script may download and run the generation model and is intentionally not part of the
pytest suite. Unit tests replace model loading with local fakes.

## Development

Install development dependencies and run the complete validation suite:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Source code follows a `src/` layout under `src/chunking_metrics/`, with tests in `tests/` and
experiments in `scripts/`.

## Documentation

- [Metric plan and reference](docs/chunking_metrics.md)
- [Deep research report](docs/deep-research-report.md)
- [Domain-agnostic automatic evaluation paper](docs/Domain-agnosticAutomaticEvaluationTextChunking.pdf)
