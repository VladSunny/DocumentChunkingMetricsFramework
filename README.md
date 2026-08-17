# Document Chunking Metrics Framework

`chunking_metrics` is a Python 3.10+ library for reference-free evaluation of document
chunking. It measures structural and semantic properties of chunks without gold boundaries,
reference chunks, annotated questions, or human relevance labels.

## Project plan

Calculation readiness, preparation-helper availability, and a complete end-to-end pipeline are
tracked separately. A checked calculation item means that the public metric function is
implemented. A checked preparation item means that a corresponding public helper exists in
`chunking_metrics.preparations`; it does not by itself make the full metric pipeline complete.

- [Size Compliance (SC)](docs/chunking_metrics.md#1-size-compliance-sc)
  - [x] Calculation
  - [ ] Preparation: calculate chunk lengths
- [Block Integrity (BI)](docs/chunking_metrics.md#2-block-integrity-bi)
  - [ ] Calculation
  - [ ] Preparation: extract structural spans
- [Intrachunk Cohesion (ICC)](docs/chunking_metrics.md#3-intrachunk-cohesion-icc)
  - [x] Calculation
  - [ ] Preparation: split chunks into sentences
  - Embeddings
    - [x] Local
    - [x] API
- [Contextual Coherence (DCC)](docs/chunking_metrics.md#4-contextual-coherence-dcc)
  - [x] Calculation
  - [ ] Preparation: extract context windows
  - Embeddings
    - [x] Local
    - [x] API
- [Coreference Integrity (RC)](docs/chunking_metrics.md#5-coreference-integrity-rc)
  - [ ] Calculation
  - Coreference resolution
    - [ ] Local
    - [ ] API
- [Boundary Clarity (BC)](docs/chunking_metrics.md#6-boundary-clarity-bc)
  - [x] Calculation
  - Perplexity/scoring
    - [x] Local
    - [ ] API
- [ChunkScore](docs/chunking_metrics.md#7-chunkscore)
  - [ ] Calculation
  - Perplexity
    - [x] Local
    - [ ] API
  - Embeddings
    - [x] Local
    - [x] API
- [HOPE Concept Unity](docs/chunking_metrics.md#8-hope-concept-unity)
  - [x] Calculation
  - Statements
    - [x] Local
    - [x] API
  - Embeddings
    - [x] Local
    - [x] API
- [HOPE Semantic Independence](docs/chunking_metrics.md#9-hope-semantic-independence)
  - [x] Calculation
  - Questions
    - [x] Local
    - [x] API
  - Retrieval
    - [x] Local
    - [ ] API
  - Answers
    - [x] Local
    - [x] API
  - Answer embeddings
    - [x] Local
    - [x] API
- [HOPE Information Preservation](docs/chunking_metrics.md#10-hope-information-preservation)
  - [ ] Calculation
  - [ ] Preparation: sample document segments
  - Statements
    - [ ] Local
    - [x] API
  - Retrieval
    - [x] Local
    - [ ] API
  - Evaluation
    - [ ] Local
    - [x] API
  - [ ] Preparation: aggregate evaluation results

`block_integrity`, `coreference_integrity`, and `chunk_score` currently exist only as
`NotImplementedError` placeholders. Information Preservation has API generation and evaluation
helpers, but no calculation function or complete preparation pipeline. HOPE Aggregate must also be
calculated by the caller.

## Features

- Six implemented calculation functions for size, cohesion, context, boundaries, Concept Unity,
  and Semantic Independence, plus three planned metric placeholders.
- Twelve public preparation helpers: six local and six API-based helpers for the implemented
  model-backed steps. The roadmap above identifies the remaining preparation work needed for
  complete end-to-end pipelines.
- Ten public prompt constants, including system and user prompts for all generation workflows.
- Local Hugging Face inference and provider-neutral OpenAI-compatible Chat Completions and
  Embeddings helpers.

For formulas, assumptions, and planned metrics, see the
[metric reference](docs/chunking_metrics.md). For one example of every implemented function and
complete document-level pipelines, use the
**[practical usage guide](docs/usage-examples.md)**.

## Installation

Install the project and its locked development dependencies from a local checkout:

```bash
uv sync --dev
```

To use the checkout from a sibling project, run either:

```bash
uv add --editable ../DocumentChunkingMetricsFramework
```

or, in an activated `pip` environment:

```bash
python -m pip install -e ../DocumentChunkingMetricsFramework
```

For a fixed artifact, build and install the wheel:

```bash
uv build
uv add ../DocumentChunkingMetricsFramework/dist/chunking_metrics-0.1.0-py3-none-any.whl
```

## Quick start

Size Compliance needs no model:

```python
import numpy as np

from chunking_metrics.metrics import size_compliance

chunk_lengths = [180, 240, 510, 320]
scores = size_compliance(chunk_lengths, min_size=200, max_size=500)
print(scores)  # [0.0, 1.0, 0.0, 1.0]

# Aggregate explicitly when a document-level score is needed.
document_score = float(np.mean(scores))
print(document_score)  # 0.5
```

A model-backed metric is prepared explicitly. For example, Intrachunk Cohesion expects one
sentence-embedding matrix per chunk:

```python
from chunking_metrics.metrics import intrachunk_cohesion
from chunking_metrics.preparations import local

sentences_by_chunk = [
    ["Chunking splits a document.", "Chunk size affects retrieval."],
    ["Cohesion measures relatedness.", "Related sentences belong together."],
]
embeddings_by_chunk = [local.calculate_embeddings(sentences) for sentences in sentences_by_chunk]
scores_by_chunk = intrachunk_cohesion(embeddings_by_chunk)
print(scores_by_chunk)
```

Model helpers may download their default Hugging Face models on first use. The
[practical usage guide](docs/usage-examples.md) covers model selection, API configuration, every
callable, custom prompts, and complete SC/ICC/DCC/BC/HOPE pipelines.

## Public API

Import through the three public modules shown below. The package root exports these modules, not
their individual members.

```python
from chunking_metrics import metrics, preparations, prompts
```

### Metrics (`chunking_metrics.metrics`)

| Signature | Result |
| --- | --- |
| `size_compliance(lengths: Iterable[int], min_size: int, max_size: int) -> list[float]` | One inclusive-range compliance score per chunk |
| `block_integrity(*args: Any, **kwargs: Any) -> None` | Placeholder; raises `NotImplementedError` |
| `intrachunk_cohesion(embs: Iterable[np.ndarray]) -> list[float]` | Sentence-to-centroid similarity for each chunk |
| `contextual_coherence(chunk_embs: np.ndarray, context_embs: np.ndarray) -> float` | Chunk-to-mean-context similarity |
| `coreference_integrity(*args: Any, **kwargs: Any) -> None` | Placeholder; raises `NotImplementedError` |
| `boundary_clarity(uncond_ppls: np.ndarray[float], cond_ppls: np.ndarray[float]) -> list[float]` | Conditioned/unconditioned perplexity ratio for each boundary |
| `chunk_score(*args: Any, **kwargs: Any) -> None` | Placeholder; raises `NotImplementedError` |
| `concept_unity(statements_embs: np.ndarray) -> float` | Mean clipped pairwise statement similarity |
| `semantic_independence(standalone_answer_embs: np.ndarray, contextual_answer_embs: np.ndarray) -> float` | Mean clipped similarity of corresponding answers |

### Input preparation (`chunking_metrics.preparations`)

#### Local helpers (`chunking_metrics.preparations.local`)

| Signature | Purpose |
| --- | --- |
| `calculate_perplexity(text: str, model_name: str = DEFAULT_PERPLEXITY_MODEL, *, context: str \| None = None, device: str \| None = None) -> float` | Calculate target perplexity, optionally conditioned on preceding text |
| `calculate_embeddings(texts: str \| Sequence[str], model_name: str = DEFAULT_EMBEDDING_MODEL, *, device: str \| None = None, batch_size: int = 32) -> np.ndarray` | Encode one text or a sequence as normalized embeddings |
| `retrieve_relevant_chunks(queries: Sequence[str], candidate_chunks: Sequence[str], model_name: str = DEFAULT_EMBEDDING_MODEL, *, top_k: int = 3, device: str \| None = None, batch_size: int = 32) -> list[list[str]]` | Rank candidate chunks independently for each query |
| `generate_answers(questions: Sequence[str], chunk: str, model_name: str = DEFAULT_STATEMENT_MODEL, *, additional_chunks_by_question: Sequence[Sequence[str]] \| None = None, prompt: str = DEFAULT_ANSWER_PROMPT, temperature: float = 0.0, max_new_tokens: int = 128, device: str \| None = None) -> list[str]` | Answer questions with a local causal chat model |
| `generate_statements(chunk: str, model_name: str = DEFAULT_STATEMENT_MODEL, *, prompt: str = DEFAULT_STATEMENT_PROMPT, statement_count: int = 5, temperature: float = 0.7, max_new_tokens: int = 256, device: str \| None = None) -> list[str]` | Generate statements with a local causal chat model |
| `generate_questions(chunk: str, model_name: str = DEFAULT_STATEMENT_MODEL, *, prompt: str = DEFAULT_QUESTION_PROMPT, question_count: int = 5, temperature: float = 0.7, max_new_tokens: int = 256, device: str \| None = None) -> list[str]` | Generate questions with a local causal chat model |

#### API helpers (`chunking_metrics.preparations.api`)

| Signature | Purpose |
| --- | --- |
| `calculate_embeddings(texts: str \| Sequence[str], model_name: str, api_key: str, base_url: str \| None = None, *, dimensions: int \| None = None, batch_size: int = 2048) -> np.ndarray` | Encode one text or a sequence as unnormalized embeddings through an OpenAI-compatible API |
| `generate_answers(questions: Sequence[str], chunk: str, model_name: str, api_key: str, base_url: str \| None = None, *, additional_chunks_by_question: Sequence[Sequence[str]] \| None = None, prompt: str = DEFAULT_ANSWER_PROMPT, temperature: float = 0.0, max_new_tokens: int = 128) -> list[str]` | Answer questions through an OpenAI-compatible API |
| `generate_statements(chunk: str, model_name: str = "", api_key: str = "", base_url: str = "", *, prompt: str = DEFAULT_STATEMENT_PROMPT, statement_count: int = 5, temperature: float = 0.7, max_new_tokens: int = 256) -> list[str]` | Generate statements through an OpenAI-compatible API |
| `generate_information_preservation_statements(segment: str, model_name: str = "", api_key: str = "", base_url: str \| None = None, *, prompt: str = DEFAULT_INFORMATION_PRESERVATION_PROMPT, temperature: float = 0.7, max_new_tokens: int = 256) -> tuple[str, list[str]]` | Generate one true and three false statements |
| `evaluate_information_preservation(true_statement: str, false_statements: Sequence[str], relevant_chunks: Sequence[str], model_name: str, api_key: str, base_url: str \| None = None, *, prompt: str = DEFAULT_INFORMATION_PRESERVATION_EVALUATION_PROMPT, temperature: float = 0.0, max_new_tokens: int = 32, seed: int \| None = None) -> int` | Score one retrieved multiple-choice test as `0` or `1` |
| `generate_questions(chunk: str, model_name: str = "", api_key: str = "", base_url: str = "", *, prompt: str = DEFAULT_QUESTION_PROMPT, question_count: int = 5, temperature: float = 0.7, max_new_tokens: int = 256) -> list[str]` | Generate questions through an OpenAI-compatible API |

### Prompts (`chunking_metrics.prompts`)

| Workflow | System prompt | User prompt |
| --- | --- | --- |
| Statements | `DEFAULT_STATEMENT_SYSTEM_PROMPT` | `DEFAULT_STATEMENT_PROMPT` |
| Questions | `DEFAULT_QUESTION_SYSTEM_PROMPT` | `DEFAULT_QUESTION_PROMPT` |
| Answers | `DEFAULT_ANSWER_SYSTEM_PROMPT` | `DEFAULT_ANSWER_PROMPT` |
| Information Preservation generation | `DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT` | `DEFAULT_INFORMATION_PRESERVATION_PROMPT` |
| Information Preservation evaluation | `DEFAULT_INFORMATION_PRESERVATION_EVALUATION_SYSTEM_PROMPT` | `DEFAULT_INFORMATION_PRESERVATION_EVALUATION_PROMPT` |

Only user-prompt constants are accepted through a helper's `prompt=` argument. Custom templates
must preserve the required placeholders:

| Helper family | Required placeholders |
| --- | --- |
| Statement generation | `{chunk}`, `{statement_count}` |
| Question generation | `{chunk}`, `{question_count}` |
| Answer generation | `{question}`, `{chunk}`, `{additional_chunks}` |
| Information Preservation generation | `{segment}` |
| Information Preservation evaluation | `{statements}`, `{relevant_chunks}` |

Values are serialized into prompts as JSON. Escape literal braces in a custom format string as
`{{` and `}}`.

## Model and runtime notes

- Embeddings default to `cointegrated/rubert-tiny2`; perplexity defaults to
  `ai-forever/rugpt3small_based_on_gpt2`; local generation defaults to
  `Qwen/Qwen2.5-1.5B-Instruct`. Select compatible models for the language and task.
- With no `device`, CUDA is preferred, then Apple MPS, then CPU. Supported explicit values are
  `cpu`, `cuda[:index]`, and `mps` when available.
- The most recently loaded embedding model and causal language model are cached. Perplexity and
  all local generation helpers share the causal-model cache.
- Embedding inputs over the model limit are truncated after a warning. Perplexity discards the
  oldest context tokens when necessary but rejects a target that cannot fit.
- Local statement and question generation require a tokenizer chat template. Generation helpers
  make no retries, and local prompts are not truncated.
- API helpers use OpenAI-compatible Chat Completions and Embeddings endpoints. Supply an explicit
  provider model and API key; set a base URL for non-OpenAI providers. Callers own provider errors,
  retries, rate limits, and costs.
- Preparation helpers raise `TypeError` or `ValueError` for invalid inputs. Metric functions
  generally return `0.0` for invalid shapes or values that they explicitly validate.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

## Documentation

- **[Practical usage guide](docs/usage-examples.md)**
- [Metric plan and reference](docs/chunking_metrics.md)
- [Deep research report](docs/deep-research-report.md)
- [Domain-agnostic automatic evaluation paper](docs/Domain-agnosticAutomaticEvaluationTextChunking.pdf)
