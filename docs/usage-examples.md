# Practical usage examples

This guide covers every implemented public callable in `chunking_metrics` version `0.1.0`, from
small standalone calls to complete document-level evaluation pipelines. Import functions from
`chunking_metrics.metrics`, `chunking_metrics.preparations`, and prompt constants from
`chunking_metrics.prompts`.

The examples deliberately separate calculation from input preparation. Local model examples may
download Hugging Face models on first use. API examples use `OPENAI_API_KEY` and always set both a
provider-specific model identifier and an OpenAI-compatible `base_url`; replace the example values
with those supplied by your provider.

## Metric functions

### Size Compliance

`size_compliance` returns one score per chunk: `1.0` when its length is in the inclusive range and
`0.0` otherwise. Invalid input produces an empty list.

```python
import numpy as np

from chunking_metrics.metrics import size_compliance

lengths = [180, 240, 510, 320]
scores_by_chunk = size_compliance(lengths, min_size=200, max_size=500)
print(scores_by_chunk)  # [0.0, 1.0, 0.0, 1.0]

# Aggregate explicitly when a document-level score is needed.
document_score = float(np.mean(scores_by_chunk))
print(document_score)  # 0.5
```

### Intrachunk Cohesion

Provide one two-dimensional sentence-embedding matrix for each chunk.

```python
import numpy as np

from chunking_metrics.metrics import intrachunk_cohesion

embeddings_by_chunk = [
    np.array([[1.0, 0.0], [0.8, 0.2]]),
    np.array([[0.0, 1.0], [0.1, 0.9]]),
]
scores_by_chunk = intrachunk_cohesion(embeddings_by_chunk)
print(scores_by_chunk)

# Aggregate explicitly when a document-level score is needed.
document_score = float(np.mean(scores_by_chunk))
print(document_score)
```

### Contextual Coherence

The first argument is one chunk vector. The second is a matrix containing the vectors selected as
that chunk's context.

```python
import numpy as np

from chunking_metrics.metrics import contextual_coherence

chunk_embedding = np.array([0.9, 0.1])
context_embeddings = np.array([[1.0, 0.0], [0.8, 0.2]])
score = contextual_coherence(chunk_embedding, context_embeddings)
print(score)
```

### Coreference Integrity

Pass entity-pronoun pairs already extracted by an external coreference resolver. Each pair stores
the entity start and pronoun end as character offsets in the source document; chunk boundaries use
the same coordinate system.

```python
from chunking_metrics.metrics import coreference_integrity

entity_pronoun_spans = [(2, 8), (12, 18), (22, 28)]
internal_chunk_boundaries = [10, 15]
score = coreference_integrity(entity_pronoun_spans, internal_chunk_boundaries)
print(score)  # 0.6666666666666667
```

### Boundary Clarity

For `K` chunks, pass `K` unconditional perplexities and `K - 1` perplexities in which each chunk
after the first is conditioned on its predecessor.

```python
import numpy as np

from chunking_metrics.metrics import boundary_clarity

unconditional_perplexities = np.array([22.0, 30.0, 25.0])
conditional_perplexities = np.array([18.0, 20.0])
scores_by_boundary = boundary_clarity(unconditional_perplexities, conditional_perplexities)
print(scores_by_boundary)  # [0.6, 0.8]

# Aggregate explicitly when a document-level score is needed.
document_score = float(np.mean(scores_by_boundary))
print(document_score)  # 0.7
```

### Semantic Dispersion and ChunkScore

`semantic_dispersion` accepts one embedding per chunk. It centers each row across its features
and returns the regularized log-determinant score. The value is not clipped and may be negative.
`chunk_score` combines already calculated scalar LI and SD components.

```python
import numpy as np

from chunking_metrics.metrics import chunk_score, semantic_dispersion

chunk_embeddings = np.array(
    [
        [1.0, -1.0, 0.0],
        [1.0, 1.0, -2.0],
    ]
)
sd = semantic_dispersion(chunk_embeddings, alpha=1e-3)
score = chunk_score(logical_independence=0.8, semantic_dispersion=sd)
print({"SD": sd, "ChunkScore": score})
```

### HOPE Concept Unity

Concept Unity expects the embeddings of factual statements generated from one chunk. The diagonal
self-similarities are included by the current implementation.

```python
import numpy as np

from chunking_metrics.metrics import concept_unity

statement_embeddings = np.array(
    [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
    ]
)
score = concept_unity(statement_embeddings)
print(score)
```

### HOPE Semantic Independence

Rows must pair standalone and context-augmented answers to the same questions.

```python
import numpy as np

from chunking_metrics.metrics import semantic_independence

standalone_answer_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
contextual_answer_embeddings = np.array([[0.9, 0.1], [0.2, 0.8]])
score = semantic_independence(
    standalone_answer_embeddings,
    contextual_answer_embeddings,
)
print(score)
```

## Embeddings, perplexity, and retrieval

### Embeddings

`local.calculate_embeddings` returns a one-dimensional vector for a string and a matrix for a sequence.
All returned vectors are L2-normalized.

```python
from chunking_metrics.preparations import local

one_embedding = local.calculate_embeddings(
    "Chunking splits a document into retrievable units.",
    model_name="cointegrated/rubert-tiny2",
    device="cpu",
)
many_embeddings = local.calculate_embeddings(
    ["First sentence.", "Second sentence."],
    model_name="cointegrated/rubert-tiny2",
    device="cpu",
    batch_size=2,
)
print(one_embedding.shape, many_embeddings.shape)
```

`api.calculate_embeddings` uses an OpenAI-compatible Embeddings endpoint. Its vectors are returned
as provided by the service, without L2-normalization. Set `dimensions` only when the selected model
and provider support shortening embeddings.

```python
import os

from chunking_metrics.preparations import api

embeddings = api.calculate_embeddings(
    ["First sentence.", "Second sentence."],
    model_name="provider/embedding-model",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.provider.example/v1",
    dimensions=1024,
    batch_size=128,
)
print(embeddings.shape)
```

### Perplexity

Only target tokens contribute to the loss when `context` is supplied.

```python
from chunking_metrics.preparations import local

unconditional = local.calculate_perplexity(
    "The second chunk starts here.",
    model_name="ai-forever/rugpt3small_based_on_gpt2",
    device="cpu",
)
conditional = local.calculate_perplexity(
    "The second chunk starts here.",
    model_name="ai-forever/rugpt3small_based_on_gpt2",
    context="The first chunk provides preceding context.",
    device="cpu",
)
print(unconditional, conditional)
```

The API equivalent targets a causal LM served through vLLM's
[OpenAI-compatible `/v1/completions` endpoint](https://docs.vllm.ai/en/v0.12.0/serving/openai_compatible_server/).
The server must return echoed prompt `text_offset` and `token_logprobs` values; provider errors
and context-window overflow are left to the caller.

```python
import os

from chunking_metrics.preparations import api

settings = {
    "model_name": "Qwen/Qwen2.5-1.5B",
    "api_key": os.environ["OPENAI_API_KEY"],
    "base_url": "http://localhost:8000/v1",
}
unconditional = api.calculate_perplexity(
    "The second chunk starts here.",
    **settings,
)
conditional = api.calculate_perplexity(
    "The second chunk starts here.",
    context="The first chunk provides preceding context.",
    **settings,
)
print(unconditional, conditional)
```

### Retrieval

Each query gets its own relevance-ranked list. If fewer than `top_k` candidates exist, all are
returned. Retrieval currently uses local embeddings; an API retrieval helper is not implemented.

```python
from chunking_metrics.preparations import local

queries = ["How is chunk quality measured?", "Which model computes perplexity?"]
candidates = [
    "Cohesion measures whether sentences in a chunk are related.",
    "A causal language model supplies perplexity values.",
    "Retrieval ranks chunks against a query embedding.",
]
matches = local.retrieve_relevant_chunks(
    queries,
    candidates,
    model_name="cointegrated/rubert-tiny2",
    top_k=2,
    device="cpu",
)
print(matches)
```

## Statement, question, and answer generation

Local generation uses a Hugging Face causal chat model. API generation uses OpenAI-compatible Chat
Completions. Results are validated strictly with Pydantic. `max_regenerations` controls additional
attempts after an invalid result and defaults to `0`; provider and model-pipeline errors are always
propagated immediately, so callers should handle those retry policies outside the library.

### Statements with a local model

```python
from chunking_metrics.preparations import local

statements = local.generate_statements(
    "The archive opened in 2019. It digitized 4,000 maps in its first year.",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    statement_count=3,
    temperature=0.7,
    max_new_tokens=192,
    max_regenerations=2,
    device="cpu",
)
print(statements)
```

### Statements through an API

```python
import os

from chunking_metrics.preparations import api

statements = api.generate_statements(
    "The archive opened in 2019. It digitized 4,000 maps in its first year.",
    model_name="provider/model-name",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.provider.example/v1",
    statement_count=3,
    temperature=0.7,
    max_new_tokens=192,
    max_regenerations=2,
)
print(statements)
```

### Questions with a local model

```python
from chunking_metrics.preparations import local

questions = local.generate_questions(
    "The archive opened in 2019. It digitized 4,000 maps in its first year.",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    question_count=3,
    temperature=0.7,
    max_new_tokens=192,
    device="cpu",
)
print(questions)
```

### Questions through an API

```python
import os

from chunking_metrics.preparations import api

questions = api.generate_questions(
    "The archive opened in 2019. It digitized 4,000 maps in its first year.",
    model_name="provider/model-name",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.provider.example/v1",
    question_count=3,
    temperature=0.7,
    max_new_tokens=192,
)
print(questions)
```

### Answers with a local model

The optional outer `additional_chunks_by_question` sequence must match the number and order of
questions.

```python
from chunking_metrics.preparations import local

questions = ["When did the archive open?", "How many maps did it digitize?"]
answers = local.generate_answers(
    questions,
    "The archive opened in 2019.",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    additional_chunks_by_question=[
        [],
        ["The archive digitized 4,000 maps in its first year."],
    ],
    temperature=0.0,
    max_new_tokens=64,
    max_regenerations=2,
    device="cpu",
)
print(answers)
```

### Answers through an API

```python
import os

from chunking_metrics.preparations import api

questions = ["When did the archive open?", "How many maps did it digitize?"]
answers = api.generate_answers(
    questions,
    "The archive opened in 2019.",
    model_name="provider/model-name",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.provider.example/v1",
    additional_chunks_by_question=[
        [],
        ["The archive digitized 4,000 maps in its first year."],
    ],
    temperature=0.0,
    max_new_tokens=64,
)
print(answers)
```

## Information Preservation generation and evaluation

The generation helper returns one true statement and exactly three distinct false alternatives.
The evaluation helper shuffles those four choices, asks the API model to select the statement
supported by the retrieved chunks, and returns `1` for a correct choice or `0` otherwise.

### Generate a test

```python
import os

from chunking_metrics.preparations import api

true_statement, false_statements = api.generate_information_preservation_statements(
    "In 2024, the library opened a reading room on the second floor.",
    model_name="provider/model-name",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.provider.example/v1",
    temperature=0.7,
    max_new_tokens=192,
    max_regenerations=2,
)
print(true_statement, false_statements)
```

### Evaluate a test

```python
import os

from chunking_metrics.preparations import api

score = api.evaluate_information_preservation(
    "The library opened a reading room in 2024.",
    [
        "The library closed its reading room in 2024.",
        "The library opened a laboratory in 2024.",
        "The library opened a reading room in 2020.",
    ],
    ["In 2024, the library opened a reading room on the second floor."],
    model_name="provider/model-name",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.provider.example/v1",
    temperature=0.0,
    max_new_tokens=32,
    seed=0,
    max_regenerations=2,
)
print(score)
```

`seed` controls only option shuffling. It does not make the API model itself deterministic.

## Custom prompts

The public constants are grouped by workflow:

```python
from chunking_metrics.prompts import (
    DEFAULT_ANSWER_PROMPT,
    DEFAULT_ANSWER_SYSTEM_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_EVALUATION_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_EVALUATION_SYSTEM_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT,
    DEFAULT_QUESTION_PROMPT,
    DEFAULT_QUESTION_SYSTEM_PROMPT,
    DEFAULT_STATEMENT_PROMPT,
    DEFAULT_STATEMENT_SYSTEM_PROMPT,
)
```

The helpers accept only a replacement user prompt; their system prompts are fixed. Custom user
templates must contain every placeholder required by that workflow:

| Workflow | Required placeholders |
| --- | --- |
| Statements | `{chunk}`, `{statement_count}` |
| Questions | `{chunk}`, `{question_count}` |
| Answers | `{question}`, `{chunk}`, `{additional_chunks}` |
| Information Preservation generation | `{segment}` |
| Information Preservation evaluation | `{statements}`, `{relevant_chunks}` |

For example:

```python
from chunking_metrics.preparations import local

custom_prompt = "Return exactly {statement_count} claims from {chunk} as a JSON array."
statements = local.generate_statements(
    "The observatory began operating in 1965.",
    prompt=custom_prompt,
    statement_count=2,
)
print(statements)
```

Substituted values are JSON-serialized. Double literal braces (`{{` and `}}`) in Python format
strings. A custom prompt does not change the response shape enforced by the corresponding parser.

## Complete document-level pipelines

The following examples show how to aggregate chunk-level inputs into document scores. Adapt the
sentence splitting, context window, generation counts, retrieval depth, and models to the document
and evaluation protocol. Keep those choices fixed when comparing chunking strategies.

### SC, ICC, DCC, and BC

This pipeline uses immediate neighboring chunks as the DCC context and the preceding chunk as the
BC context.

```python
import numpy as np

from chunking_metrics.metrics import (
    boundary_clarity,
    contextual_coherence,
    intrachunk_cohesion,
    size_compliance,
)
from chunking_metrics.preparations import local

sentences_by_chunk = [
    ["The archive opened in 2019.", "It stores historical maps."],
    ["A digitization project began in 2020.", "The first phase covered 4,000 maps."],
    ["Researchers can search the catalog online.", "Original maps remain in controlled storage."],
]
chunks = [" ".join(sentences) for sentences in sentences_by_chunk]

sc_by_chunk = size_compliance([len(chunk) for chunk in chunks], min_size=80, max_size=240)
sc = float(np.mean(sc_by_chunk))

sentence_embeddings_by_chunk = [
    local.calculate_embeddings(sentences, device="cpu") for sentences in sentences_by_chunk
]
icc_by_chunk = intrachunk_cohesion(sentence_embeddings_by_chunk)
icc = float(np.mean(icc_by_chunk))

chunk_embeddings = local.calculate_embeddings(chunks, device="cpu")
dcc_by_chunk = []
for index, chunk_embedding in enumerate(chunk_embeddings):
    neighbor_indices = [
        neighbor for neighbor in (index - 1, index + 1) if 0 <= neighbor < len(chunks)
    ]
    dcc_by_chunk.append(contextual_coherence(chunk_embedding, chunk_embeddings[neighbor_indices]))
dcc = float(np.mean(dcc_by_chunk))

unconditional = np.array([local.calculate_perplexity(chunk, device="cpu") for chunk in chunks])
conditional = np.array(
    [
        local.calculate_perplexity(current, context=previous, device="cpu")
        for previous, current in zip(chunks, chunks[1:])
    ]
)
bc_by_boundary = boundary_clarity(unconditional, conditional)
bc = float(np.mean(bc_by_boundary))

print({"SC": sc, "ICC": icc, "DCC": dcc, "BC": bc})
```

### QChunker ChunkScore

This pipeline embeds complete chunks for SD, averages Boundary Clarity into the LI scalar, and
then combines the two precomputed components with the recommended LI weight of `0.3`.

```python
import numpy as np

from chunking_metrics.metrics import boundary_clarity, chunk_score, semantic_dispersion
from chunking_metrics.preparations import local

chunks = [
    "The archive opened in 2019 and stores historical maps.",
    "A digitization project began in 2020 and covered 4,000 maps.",
    "Researchers can search the catalog online.",
]

# embeddings -> Semantic Dispersion
chunk_embeddings = local.calculate_embeddings(chunks, device="cpu")
sd = semantic_dispersion(chunk_embeddings)

# perplexities -> Boundary Clarity -> mean Logical Independence
unconditional = np.array([local.calculate_perplexity(chunk, device="cpu") for chunk in chunks])
conditional = np.array(
    [
        local.calculate_perplexity(current, context=previous, device="cpu")
        for previous, current in zip(chunks, chunks[1:])
    ]
)
li_by_boundary = boundary_clarity(unconditional, conditional)
li = float(np.mean(li_by_boundary))

# LI + SD -> ChunkScore
score = chunk_score(li, sd)
print({"LI": li, "SD": sd, "ChunkScore": score})
```

### HOPE Concept Unity across a document

`concept_unity` evaluates one statement matrix. Generate statements per chunk and average the
resulting chunk scores for a document-level value.

```python
from statistics import fmean

from chunking_metrics.metrics import concept_unity
from chunking_metrics.preparations import local

chunks = [
    "The archive opened in 2019. It stores historical maps.",
    "A digitization project began in 2020. Its first phase covered 4,000 maps.",
]

chunk_scores = []
for chunk in chunks:
    statements = local.generate_statements(
        chunk,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        statement_count=4,
        device="cpu",
    )
    statement_embeddings = local.calculate_embeddings(statements, device="cpu")
    chunk_scores.append(concept_unity(statement_embeddings))

document_concept_unity = fmean(chunk_scores)
print(document_concept_unity)
```

### HOPE Semantic Independence with local generation

For every evaluated chunk, retrieve only from the *other* chunks in the same document. Including
the evaluated chunk among retrieval candidates would invalidate the independence comparison.

```python
from statistics import fmean

from chunking_metrics.metrics import semantic_independence
from chunking_metrics.preparations import local

chunks = [
    "The archive opened in 2019. It stores historical maps.",
    "A digitization project began in 2020. Its first phase covered 4,000 maps.",
    "Researchers can search the catalog online.",
]

chunk_scores = []
for chunk_index, chunk in enumerate(chunks):
    other_chunks = [candidate for index, candidate in enumerate(chunks) if index != chunk_index]
    questions = local.generate_questions(
        chunk,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        question_count=4,
        device="cpu",
    )
    retrieved = local.retrieve_relevant_chunks(questions, other_chunks, top_k=2, device="cpu")
    standalone_answers = local.generate_answers(
        questions,
        chunk,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cpu",
    )
    contextual_answers = local.generate_answers(
        questions,
        chunk,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        additional_chunks_by_question=retrieved,
        device="cpu",
    )
    chunk_scores.append(
        semantic_independence(
            local.calculate_embeddings(standalone_answers, device="cpu"),
            local.calculate_embeddings(contextual_answers, device="cpu"),
        )
    )

document_semantic_independence = fmean(chunk_scores)
print(document_semantic_independence)
```

This pipeline requires at least two chunks because retrieval candidates cannot be empty.

### HOPE Semantic Independence through an API

Embeddings and retrieval remain local here; question and answer generation use the same explicit
API model and endpoint.

```python
import os
from statistics import fmean

from chunking_metrics.metrics import semantic_independence
from chunking_metrics.preparations import api, local

api_key = os.environ["OPENAI_API_KEY"]
model_name = "provider/model-name"
base_url = "https://api.provider.example/v1"
chunks = [
    "The archive opened in 2019. It stores historical maps.",
    "A digitization project began in 2020. Its first phase covered 4,000 maps.",
    "Researchers can search the catalog online.",
]

chunk_scores = []
for chunk_index, chunk in enumerate(chunks):
    other_chunks = [candidate for index, candidate in enumerate(chunks) if index != chunk_index]
    questions = api.generate_questions(
        chunk,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        question_count=4,
    )
    retrieved = local.retrieve_relevant_chunks(questions, other_chunks, top_k=2, device="cpu")
    standalone_answers = api.generate_answers(
        questions,
        chunk,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    contextual_answers = api.generate_answers(
        questions,
        chunk,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        additional_chunks_by_question=retrieved,
    )
    chunk_scores.append(
        semantic_independence(
            local.calculate_embeddings(standalone_answers, device="cpu"),
            local.calculate_embeddings(contextual_answers, device="cpu"),
        )
    )

document_semantic_independence = fmean(chunk_scores)
print(document_semantic_independence)
```

### HOPE Information Preservation

Generate a multiple-choice test from each source segment, retrieve evidence using its true
statement, evaluate the choice against those chunks, and average the binary results.

```python
from statistics import fmean

from chunking_metrics.preparations import local

chunks = [
    "The archive opened in 2019. It stores historical maps.",
    "A digitization project began in 2020. Its first phase covered 4,000 maps.",
    "Researchers can search the catalog online.",
]

test_scores = []
for test_index, segment in enumerate(chunks):
    true_statement, false_statements = local.generate_information_preservation_statements(
        segment,
        device="cpu",
    )
    relevant_chunks = local.retrieve_relevant_chunks(
        [true_statement],
        chunks,
        top_k=2,
        device="cpu",
    )[0]
    test_scores.append(
        local.evaluate_information_preservation(
            true_statement,
            false_statements,
            relevant_chunks,
            seed=test_index,
            device="cpu",
        )
    )

document_information_preservation = fmean(test_scores)
print(document_information_preservation)
```

Use multiple independently generated tests per document in a serious evaluation. Generation
helpers do not return a partial aggregate; optional regeneration applies only to an invalid result
for the current operation.

### Manual HOPE Aggregate

Once the three document-level components have been computed, combine them manually with an
unweighted arithmetic mean:

```python
from statistics import fmean

document_concept_unity = 0.84
document_semantic_independence = 0.79
document_information_preservation = 0.88

hope_aggregate = fmean(
    [
        document_concept_unity,
        document_semantic_independence,
        document_information_preservation,
    ]
)
print(hope_aggregate)
```

Version `0.1.0` has no library function for HOPE Aggregate. The manual mean above is an explicit
caller-side convention, not a call to a public aggregate API.

## Unimplemented metrics

Block Integrity is documented in the [metric reference](chunking_metrics.md), but its current
function is a placeholder that raises `NotImplementedError`. Coreference Integrity calculation is
implemented, but input preparation through a coreference resolver is not. HOPE Aggregate has no
function at all. Do not include the unimplemented pieces in an automated pipeline until their
implementations are added.
