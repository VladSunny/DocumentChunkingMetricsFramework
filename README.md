# Document Chunking Metrics Framework

`chunking_metrics` is a Python 3.10+ library for reference-free evaluation of document
chunking. It measures structural and semantic properties of chunks without gold boundaries,
reference chunks, annotated questions, or human relevance labels.

Version `0.1.0` implements 7 of the 10 metrics in the project plan:

- [x] [Size Compliance (SC)](docs/chunking_metrics.md#1-size-compliance-sc)
- [ ] [Block Integrity (BI)](docs/chunking_metrics.md#2-block-integrity-bi)
- [x] [Intrachunk Cohesion (ICC)](docs/chunking_metrics.md#3-intrachunk-cohesion-icc)
- [x] [Contextual Coherence (DCC)](docs/chunking_metrics.md#4-contextual-coherence-dcc)
- [ ] [Coreference Integrity (RC)](docs/chunking_metrics.md#5-coreference-integrity-rc)
- [x] [Boundary Clarity (BC)](docs/chunking_metrics.md#6-boundary-clarity-bc)
- [ ] [ChunkScore](docs/chunking_metrics.md#7-chunkscore)
- [x] [HOPE Concept Unity](docs/chunking_metrics.md#8-hope-concept-unity)
- [x] [HOPE Semantic Independence](docs/chunking_metrics.md#9-hope-semantic-independence)
- [x] [HOPE Information Preservation](docs/chunking_metrics.md#10-hope-information-preservation)

Checked entries are usable today. `block_integrity`, `coreference_integrity`, and `chunk_score`
exist only as `NotImplementedError` placeholders. Information Preservation is implemented as
generation and evaluation helpers rather than as a metric function. HOPE Aggregate must currently
be calculated by the caller.

## Features

- Six metric functions for size, cohesion, context, boundaries, Concept Unity, and Semantic
  Independence.
- Eleven preparation functions for embeddings, perplexity, retrieval, local or API-based
  statement/question/answer generation, and API-based Information Preservation.
- Ten public prompt constants, including system and user prompts for all generation workflows.
- Local Hugging Face inference and provider-neutral OpenAI-compatible Chat Completions helpers.

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
from chunking_metrics.metrics import size_compliance

chunk_lengths = [180, 240, 510, 320]
score = size_compliance(chunk_lengths, min_size=200, max_size=500)
print(score)  # 0.5
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
score = intrachunk_cohesion(embeddings_by_chunk)
print(score)
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

| Function | Input | Result |
| --- | --- | --- |
| `size_compliance(lengths, min_size, max_size)` | Chunk lengths and an inclusive range | Fraction within the range |
| `intrachunk_cohesion(embs)` | One sentence-embedding matrix per chunk | Mean sentence-to-centroid similarity |
| `contextual_coherence(chunk_embs, context_embs)` | One chunk vector and a context matrix | Chunk-to-mean-context similarity |
| `boundary_clarity(uncond_ppls, cond_ppls)` | `K` unconditional and `K - 1` conditioned perplexities | Mean conditioned/unconditioned ratio |
| `concept_unity(statements_embs)` | Statement-embedding matrix for one chunk | Mean clipped pairwise similarity |
| `semantic_independence(standalone_answer_embs, contextual_answer_embs)` | Corresponding answer matrices | Mean clipped pairwise answer similarity |

The module also exposes the unimplemented placeholders `block_integrity`,
`coreference_integrity`, and `chunk_score`. Calling them raises `NotImplementedError`.

### Input preparation (`chunking_metrics.preparations`)

| Function | Purpose |
| --- | --- |
| `local.calculate_embeddings` | Encode one text or a sequence as normalized embeddings |
| `local.calculate_perplexity` | Calculate target perplexity, optionally conditioned on preceding text |
| `local.retrieve_relevant_chunks` | Rank candidate chunks independently for each query |
| `local.generate_statements` | Generate statements with a local causal chat model |
| `api.generate_statements` | Generate statements through an OpenAI-compatible API |
| `local.generate_questions` | Generate questions with a local causal chat model |
| `api.generate_questions` | Generate questions through an OpenAI-compatible API |
| `local.generate_answers` | Answer questions with a local causal chat model |
| `api.generate_answers` | Answer questions through an OpenAI-compatible API |
| `api.generate_information_preservation_statements` | Generate one true and three false statements |
| `api.evaluate_information_preservation` | Score one retrieved multiple-choice test as `0` or `1` |

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
- API helpers use OpenAI-compatible Chat Completions. Supply an explicit provider model, API key,
  and base URL; callers own provider errors, retries, rate limits, and costs.
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
