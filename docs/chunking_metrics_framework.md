# Метрики оценки чанкирования документов

## 1. Назначение

Этот документ фиксирует набор метрик для фреймворка автоматической оценки качества чанкирования документов.

Ключевое ограничение фреймворка: **метрики не должны требовать заранее размеченных эталонных данных**. В частности, в основной набор не входят метрики, которым необходимы:

- gold boundaries;
- эталонные чанки;
- вручную размеченные вопросы и ответы;
- evidence spans;
- human relevance labels.

Допускается использование информации, автоматически извлечённой из исходного документа: структуры Markdown/HTML/PDF, предложений, токенов, embeddings, coreference chains, а также синтетических вопросов и утверждений, генерируемых LLM.

Цель фреймворка — оценивать чанкинг на нескольких уровнях:

1. **Формальные ограничения** — размер и техническая корректность чанков.
2. **Структурная целостность** — сохранение естественных блоков документа.
3. **Внутренняя семантическая связность** — насколько содержимое чанка относится к общей теме.
4. **Независимость чанков** — насколько отдельный чанк можно корректно интерпретировать без других чанков.
5. **Сохранение информации** — не теряется ли содержимое исходного документа после преобразования в набор чанков.

Метрики разделены по вычислительной сложности на три уровня.

---

## 2. Рекомендуемый стек метрик

| Уровень | Метрика | Основной аспект | Нужна генеративная LLM |
|---|---|---|---|
| 1 | Size Compliance | размер чанков | нет |
| 1 | Block Integrity | структурная целостность | нет |
| 1 | Intrachunk Cohesion | внутренняя семантическая связность | нет |
| 1 | Contextual Coherence | согласованность с локальным контекстом | нет |
| 2 | Coreference Integrity | разрывы ссылок и coreference chains | нет |
| 2 | Boundary Clarity | независимость соседних чанков | нет, нужна causal LM |
| 2 | ChunkScore | independence + semantic dispersion | нет, нужны causal LM и embeddings |
| 3 | HOPE Semantic Independence | самодостаточность чанков | да |
| 3 | HOPE Information Preservation | сохранение исходной информации | да |
| 3 | HOPE Concept Unity | единство концепта внутри чанка | да |
| 3 | HOPE aggregate | агрегат HOPE | да |

Для первой версии фреймворка приоритет реализации:

**Size Compliance → Block Integrity → Intrachunk Cohesion → Boundary Clarity → HOPE Semantic Independence → HOPE Information Preservation.**

Остальные показатели рекомендуется реализовать как дополнительные или диагностические компоненты.

---

# Level 1 — лёгкие intrinsic-метрики

## 3. Size Compliance (SC)

### Назначение

Size Compliance измеряет, какая доля чанков удовлетворяет заданным ограничениям на размер.

Метрика не оценивает семантику напрямую, но необходима как ограничивающий показатель. Без неё методы, создающие слишком крупные или слишком мелкие чанки, могут искусственно получать хорошие результаты по cohesion-метрикам.

### Формула

Пусть имеется набор чанков \(P = \{p_1, \dots, p_K\}\), а допустимая длина чанка находится в диапазоне \([L_{min}, L_{max}]\):

\[
SC = \frac{1}{K}\sum_{i=1}^{K}
\mathbf{1}\left[L_{min} \le |p_i| \le L_{max}\right].
\]

### Диапазон

\[
SC \in [0,1].
\]

- `1.0` — все чанки удовлетворяют ограничениям;
- `0.0` — ни один чанк им не удовлетворяет.

### Вход

- список чанков;
- tokenizer;
- `min_tokens`;
- `max_tokens`.

### Вычислительная сложность

После токенизации:

\[
O(K).
\]

### Рекомендации

SC следует хранить отдельно и не использовать как самостоятельный показатель качества чанкинга. Это скорее constraint metric.

Желательно дополнительно возвращать:

- `mean_chunk_size`;
- `median_chunk_size`;
- `min_chunk_size`;
- `max_chunk_size`;
- `std_chunk_size`;
- долю слишком маленьких чанков;
- долю слишком больших чанков.

---

## 4. Block Integrity (BI)

### Назначение

Block Integrity измеряет, насколько chunker сохраняет естественные структурные элементы документа и не разрезает их границами чанков.

К структурным блокам могут относиться:

- абзацы;
- заголовки и секции;
- списки;
- таблицы;
- code blocks;
- quotations;
- figure captions;
- другие элементы, полученные parser-ом.

### Формула

\[
BI = 1 - \frac{N_{cut}}{\max(1, N_{blocks})},
\]

где:

- \(N_{blocks}\) — число структурных блоков;
- \(N_{cut}\) — число блоков, пересечённых хотя бы одной границей чанка.

### Диапазон

\[
BI \in [0,1].
\]

- `1.0` — ни один структурный блок не разрезан;
- `0.0` — все учитываемые блоки нарушены.

### Вход

- исходный документ;
- позиции чанков в исходном документе;
- структурные spans, автоматически полученные parser-ом.

### Важное ограничение

BI не требует ручной разметки, если структура документа автоматически извлекается из Markdown, HTML, DOCX, PDF parser или другого ingestion pipeline.

### Расширение

Разные типы блоков желательно оценивать отдельно:

```text
block_integrity:
    paragraph
    heading
    list
    table
    code
    aggregate
```

Для таблиц и code blocks можно назначить больший вес, поскольку их разрыв часто значительно сильнее влияет на retrieval и генерацию.

---

## 5. Intrachunk Cohesion (ICC)

### Назначение

Intrachunk Cohesion оценивает семантическую однородность содержимого внутри каждого чанка.

Основная идея: предложения хорошего чанка должны быть семантически близки к общему представлению этого чанка.

### Алгоритм

Для каждого предложения \(s_i\) вычисляется embedding \(v_i\).

Для чанка \(c\) определяется centroid:

\[
\bar v_c = \frac{1}{|c|}\sum_{s_i \in c} v_i.
\]

После нормализации вычисляется среднее cosine similarity предложений с centroid:

\[
ICC(c) =
\frac{1}{|c|}
\sum_{s_i \in c}
\cos(v_i, \bar v_c).
\]

Итоговая метрика документа:

\[
ICC = \frac{1}{K}\sum_{c \in P} ICC(c).
\]

### Вход

- chunks;
- sentence splitter;
- embedding model.

### Рекомендуемая реализация

Embeddings следует вычислять **один раз на уровне предложений документа**, а затем переиспользовать для разных вариантов чанкинга. Это существенно снижает стоимость benchmark при сравнении многих chunkers.

### Ограничения

Высокий ICC сам по себе не означает хороший чанкинг.

Например, весь документ одной тематики можно объединить в один огромный чанк и получить высокий cohesion. Поэтому ICC необходимо использовать совместно как минимум с:

- Size Compliance;
- метрикой независимости чанков;
- желательно Block Integrity.

### Выход

Рекомендуется возвращать не только средний score:

```text
intrachunk_cohesion:
    mean
    median
    min
    p10
    per_chunk[]
```

Низкие значения отдельных чанков полезны для диагностики проблемных границ.

---

## 6. Contextual Coherence (DCC)

### Назначение

Contextual Coherence оценивает, насколько содержание чанка семантически согласуется с локальным контекстом вокруг него в исходном документе.

Метрика полезна для обнаружения фрагментов, которые оказались вырваны из локальной структуры повествования.

### Возможная реализация

Для каждого чанка \(c_i\) строится embedding самого чанка и embedding окружающего контекстного окна \(w_i\):

\[
DCC(c_i) = \cos(v_{c_i}, v_{w_i}).
\]

Контекстное окно может включать:

- соседние предложения исходного документа;
- предыдущий и следующий абзац;
- фиксированное число токенов вокруг чанка.

### Интерпретация

DCC следует использовать осторожно.

Высокая локальная coherence может быть полезной, но слишком сильная зависимость от соседнего контекста может означать, что чанк плохо работает самостоятельно. Поэтому DCC лучше считать **diagnostic metric**, а не основной optimization target.

### Роль во фреймворке

- ICC отвечает на вопрос: «связан ли чанк сам с собой?»;
- DCC: «естественно ли он расположен в локальном контексте?»;
- independence-метрики: «можно ли его понять без соседей?».

Эти показатели не следует смешивать в один score без предварительной эмпирической калибровки.

---

# Level 2 — модели без генеративной оценки

## 7. Coreference Integrity (RC)

### Назначение

Coreference Integrity измеряет, насколько часто границы чанков разрывают зависимости между упоминаниями сущностей.

Пример:

```text
Chunk 1:
Иван передал договор Петру.

Chunk 2:
Он подписал его на следующий день.
```

Второй чанк содержит местоимения, смысл которых зависит от первого чанка.

### Общая идея

Coreference resolver строит цепочки связанных mentions:

\[
E = \{e_1, e_2, \dots\}.
\]

Для каждой связи проверяется, находятся ли зависимые mentions в одном чанке.

Можно определить:

\[
RC = 1 - \frac{N_{broken}}{\max(1,N_{relations})}.
\]

### Вход

- исходный документ;
- chunks/spans;
- coreference resolution model.

### Ограничения

Метрика сильно зависит от:

- языка;
- качества coreference resolver;
- жанра документа.

Для русского языка поддержку необходимо проверять отдельно.

### Статус во фреймворке

**Optional metric.**

Не рекомендуется делать её обязательной зависимостью core-пакета.

---

## 8. Boundary Clarity (BC)

### Назначение

Boundary Clarity оценивает, насколько хорошо граница разделяет два соседних чанка с точки зрения языковой модели.

Если предыдущий чанк значительно облегчает предсказание следующего, между ними присутствует сильная зависимость. Это может означать, что граница проведена неудачно.

### Основная формула

Для соседних чанков \(d\) и \(q\):

\[
BC(q,d) =
\frac{PPL(q\mid d)}{PPL(q)}.
\]

где:

- \(PPL(q)\) — perplexity следующего чанка без предыдущего контекста;
- \(PPL(q\mid d)\) — perplexity того же чанка при наличии предыдущего чанка.

### Интерпретация

Если предыдущий чанк содержит критически важный контекст, обычно:

\[
PPL(q\mid d) \ll PPL(q).
\]

Следовательно, чанки сильно связаны.

При сравнительно независимой границе влияние предыдущего чанка меньше.

### Вход

- последовательность чанков;
- causal language model с доступом к token log-probabilities.

### Преимущества

- не нужна ручная разметка;
- не нужны QA;
- не нужна генерация текста;
- может полностью работать локально;
- хорошо подходит для массового offline evaluation.

### Ограничения

Perplexity зависит от:

- конкретной LM;
- языка;
- домена;
- длины контекста;
- токенизации.

Поэтому нельзя напрямую сравнивать BC, рассчитанный разными language models.

### Рекомендация

Во фреймворке необходимо фиксировать идентификатор и версию модели вместе с результатом:

```text
boundary_clarity:
    model: ...
    mean: ...
    median: ...
    per_boundary: ...
```

---

## 9. ChunkScore

### Назначение

ChunkScore — более комплексная reference-free метрика, объединяющая два свойства:

1. **Logical Independence (LI)** — логическая независимость чанков;
2. **Semantic Dispersion (SD)** — отсутствие чрезмерной семантической избыточности между чанками.

Итоговая форма:

\[
ChunkScore = \lambda LI + (1-\lambda)SD.
\]

### Logical Independence

LI использует conditional/unconditional perplexity и концептуально близок к Boundary Clarity.

Он оценивает, насколько сильно соседние чанки зависят друг от друга с позиции causal LM.

### Semantic Dispersion

Для каждого чанка вычисляется embedding. Затем оценивается геометрическая дисперсия набора embeddings, в частности через Gram matrix и log-determinant-based показатель.

Идея: если несколько чанков почти дублируют одну и ту же информацию, их embeddings будут близки и semantic dispersion снизится.

### Вход

- chunks;
- causal LM;
- embedding model;
- параметры \(\lambda\) и регуляризации.

### Преимущества

- не нужна генеративная LLM;
- не нужны labels;
- оценивает сразу independence и redundancy;
- подходит для автоматического ранжирования большого числа chunking-конфигураций.

### Ограничения

Semantic Dispersion не гарантирует сохранение исходной информации. Более того, чрезмерная фрагментация потенциально может искусственно увеличить разнообразие embeddings.

Поэтому ChunkScore не заменяет Information Preservation.

### Статус

**Advanced composite metric.**

Рекомендуется хранить отдельно:

```text
chunk_score:
    logical_independence
    semantic_dispersion
    aggregate
```

---

# Level 3 — HOPE и генеративная оценка

## 10. Общая идея HOPE

HOPE (Holistic Passage Evaluation) предлагает автоматическую domain-agnostic оценку чанкинга без human annotations.

Метод основывается на трёх принципах:

1. **Concept Unity** — passage должен содержать связанный концепт.
2. **Semantic Independence** — passage должен быть интерпретируем независимо от других passages.
3. **Information Preservation** — набор passages должен сохранять информацию исходного документа.

Итоговый показатель:

\[
HOPE = \frac{1}{3}
\left(
\zeta_{con}+
\zeta_{sem}+
\zeta_{inf}
\right).
\]

В нашем фреймворке рекомендуется считать все компоненты отдельно и **не использовать aggregate HOPE как единственный результат**.

---

## 11. HOPE Concept Unity

### Назначение

Concept Unity проверяет, насколько информация внутри одного чанка относится к единой семантической области.

### Алгоритм

Для чанка \(p\) LLM генерирует набор утверждений:

\[
LLM(p) \rightarrow S = \{s_1,\dots,s_m\}.
\]

Для утверждений строятся embeddings, после чего вычисляется среднее pairwise cosine similarity:

\[
\bar\zeta_{con}(p) =
\frac{1}{|S|^2}
\sum_{s_i\in S}
\sum_{s_j\in S}
\cos(v_i,v_j).
\]

Затем результат усредняется по чанкам.

### Интерпретация

Высокая similarity между утверждениями должна означать, что чанк посвящён согласованному набору близких концептов.

### Ограничение

В статье HOPE Concept Unity показала слабую практическую ценность: её корреляции с рассмотренными RAG performance indicators были отрицательными. Авторы отмечают, что традиционное требование «один passage — один concept» может быть неполным или неверным как универсальный принцип.

### Статус во фреймворке

**Experimental.**

Не следует использовать как обязательный optimization target.

---

## 12. HOPE Semantic Independence

### Назначение

Semantic Independence оценивает, насколько смысл чанка и ответы на вопросы по нему остаются стабильными при добавлении других чанков документа.

Это одна из наиболее важных метрик выбранного набора.

### Алгоритм

Для каждого focus chunk \(p^*\):

1. LLM генерирует вопросы \(Q\) по содержанию \(p^*\).
2. Каждый вопрос задаётся LLM только с \(p^*\).
3. Тот же вопрос задаётся LLM с \(p^*\) и дополнительными релевантными chunks.
4. Ответы сравниваются по embedding cosine similarity.

Формально:

\[
LLM(q,p^*) \rightarrow a^*,
\]

\[
LLM(q,p^*,P_q) \rightarrow a.
\]

Для набора вопросов:

\[
\bar\zeta_{sem}(p^*) =
\frac{1}{|Q|}
\sum_{q\in Q}
\cos(v_{a^*},v_a).
\]

Итог:

\[
\zeta_{sem} =
\frac{1}{K}
\sum_{p\in P}
\bar\zeta_{sem}(p).
\]

### Интерпретация

Если дополнительный контекст существенно меняет ответ, focus chunk не является семантически самостоятельным.

Если ответ остаётся практически тем же, independence высокая.

### Retrieval

Для дополнительного контекста рекомендуется использовать top-k chunks, наиболее похожих на вопрос. В оригинальном HOPE используется `k = 3`.

### Практическая значимость

В экспериментах HOPE Semantic Independence была наиболее убедительной компонентой: рост independence ассоциировался с улучшением RAG-показателей, включая Answer Correctness и Factual Correctness.

### Статус

**Core expensive metric.**

Если из HOPE реализуется только одна компонентная метрика, приоритет следует отдать именно Semantic Independence.

---

## 13. HOPE Information Preservation

### Назначение

Information Preservation проверяет, сохранилась ли информация исходного документа после преобразования в набор чанков.

Это принципиально отличается от cohesion и independence: хороший набор независимых чанков бесполезен, если часть фактов была потеряна или искажена.

### Алгоритм HOPE

1. Из исходного документа случайно выбирается сегмент из нескольких последовательных предложений.
2. LLM генерирует набор утверждений, содержащий один истинный вариант и несколько правдоподобных ложных.
3. По истинному утверждению retrieval находит релевантные чанки.
4. Вторая LLM получает retrieved context и должна определить истинное утверждение.
5. Результат оценивается бинарно.

Для набора тестов:

\[
\zeta_{inf} =
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\hat a_i = a_i].
\]

### Интерпретация

Высокий результат означает, что фактическая информация исходного документа может быть восстановлена по chunked representation.

### Вход

- исходный документ;
- chunks;
- generative LLM;
- embedding model;
- vector retrieval.

### Ограничения

Метрика зависит от качества синтетических заданий. Ошибки могут возникать из-за:

- hallucinations при генерации утверждений;
- недостаточного разнообразия samples;
- ошибок retriever;
- ошибок LLM-judge.

Поэтому рекомендуется хранить seed, prompts, model version и параметры sampling.

### Статус

**Core expensive metric.**

В паре с Semantic Independence даёт наиболее содержательную дорогостоящую оценку чанкинга.

---

## 14. HOPE Aggregate

Итоговый HOPE определяется как:

\[
HOPE =
\frac{
\zeta_{con} +
\zeta_{sem} +
\zeta_{inf}
}{3}.
\]

### Рекомендация для фреймворка

Aggregate score должен быть **вторичным выходом**.

Основной API должен возвращать компоненты отдельно:

```yaml
hope:
  concept_unity: 0.82
  semantic_independence: 0.74
  information_preservation: 0.91
  aggregate: 0.823
```

Причины:

- компоненты измеряют разные свойства;
- Concept Unity имеет спорную downstream-значимость;
- одинаковый aggregate может скрывать принципиально разные failure modes.

---

# 15. Метрики, не включаемые в основной framework

Следующие популярные показатели не подходят под базовое требование проекта.

## Boundary F1, Pk, WindowDiff, Boundary Similarity

Не включаются, поскольку требуют эталонных границ или эталонной сегментации.

Их можно поддержать в отдельном `reference_based` модуле, если в будущем появятся размеченные benchmarks.

## Chroma token Precision / Recall / IoU

В стандартной постановке требуют evaluation queries и evidence spans. Они полезны для retrieval-oriented benchmark, но не являются полностью reference-free без дополнительной генерации evaluation dataset.

## HiCBench metrics

Требуют benchmark annotations, hierarchical boundaries, QA/evidence data.

## BLEU / ROUGE

Не требуют gold boundaries, однако плохо измеряют качество чанкинга. Если chunks просто конкатенировать обратно, lexical overlap может быть почти идеальным даже при крайне неудачных границах.

Допустимо использовать только как sanity check на потерю или изменение исходного текста.

---

# 16. Архитектура вычисления

Рекомендуемый pipeline:

```text
Document
   │
   ├── Parser ───────────────► structural spans
   │
   ├── Sentence splitter ────► sentences
   │                              │
   │                              └── Embedding model
   │                                      │
   ▼                                      ▼
Chunker ───────────────────────────────► Chunks
                                          │
                ┌─────────────────────────┼─────────────────────────┐
                │                         │                         │
                ▼                         ▼                         ▼
             Level 1                   Level 2                   Level 3
          SC / BI / ICC            BC / ChunkScore          HOPE components
                │                         │                         │
                └─────────────────────────┴─────────────────────────┘
                                          │
                                          ▼
                                  Evaluation Report
```

---

# 17. Общий интерфейс метрики

Рекомендуемый программный интерфейс:

```python
class ChunkingMetric:
    name: str
    level: int

    def compute(
        self,
        document: Document,
        chunks: list[Chunk],
        context: EvaluationContext,
    ) -> MetricResult:
        ...
```

`EvaluationContext` должен позволять переиспользовать дорогие промежуточные вычисления:

```python
@dataclass
class EvaluationContext:
    sentences: list[str] | None = None
    sentence_embeddings: np.ndarray | None = None
    chunk_embeddings: np.ndarray | None = None
    structural_blocks: list[Span] | None = None
    tokenized_chunks: list[list[int]] | None = None
    vector_index: object | None = None
```

Это особенно важно при сравнении нескольких chunking strategies на одном и том же документе.

---

# 18. Формат результата

Каждая метрика должна возвращать стандартизированный объект:

```python
@dataclass
class MetricResult:
    name: str
    score: float | None
    details: dict
    metadata: dict
```

Пример полного отчёта:

```yaml
document_id: example-001
chunker:
  name: recursive
  params:
    chunk_size: 800
    overlap: 100

metrics:
  size_compliance:
    score: 0.96

  block_integrity:
    score: 0.91
    details:
      paragraph: 0.97
      table: 0.75

  intrachunk_cohesion:
    score: 0.84
    details:
      min: 0.61
      median: 0.86

  boundary_clarity:
    score: 0.77
    metadata:
      model: ...

  hope:
    semantic_independence: 0.81
    information_preservation: 0.92
    concept_unity: 0.79
    aggregate: 0.84
```

---

# 19. Воспроизводимость

Для всех model-based и LLM-based метрик необходимо сохранять:

- model name;
- model revision/hash;
- tokenizer version;
- prompts;
- temperature;
- top-p;
- seed;
- число samples/questions/statements;
- retrieval `k`;
- embedding model;
- версии библиотек, влияющих на inference.

Для генеративных метрик рекомендуется поддерживать несколько режимов:

```text
fast:
    минимальное число samples

standard:
    достаточная оценка для сравнения chunkers

robust:
    несколько повторов с агрегацией mean/std
```

При фиксированных локальных моделях предпочтительно использовать deterministic generation (`temperature=0`) там, где разнообразие не является частью определения метрики. Если по методологии требуется ненулевая temperature, необходимо фиксировать seed и число повторов.

---

# 20. Агрегация метрик

На первой стадии разработки **не рекомендуется создавать единый общий Chunking Quality Score** из всех метрик.

Причины:

- показатели имеют разные семантические значения;
- часть из них конфликтует между собой;
- оптимальные веса неизвестны;
- компоненты имеют разную доказанную связь с downstream RAG quality.

Основным выходом должен быть вектор:

\[
M =
(SC, BI, ICC, DCC, RC, BC, LI, SD,
\zeta_{con}, \zeta_{sem}, \zeta_{inf}).
\]

Для автоматического выбора конфигураций рекомендуется использовать:

1. hard constraints по SC и BI;
2. Pareto ranking по нескольким semantic metrics;
3. дорогие HOPE-метрики только для финального поднабора кандидатов.

После накопления экспериментальных данных можно обучить или откалибровать собственную функцию агрегации относительно downstream RAG performance.

---

# 21. Рекомендуемые этапы реализации

## Этап 1 — минимальный framework

Реализовать:

1. Size Compliance;
2. Block Integrity;
3. Intrachunk Cohesion.

Общие инфраструктурные компоненты:

- tokenizer abstraction;
- sentence splitter;
- embedding provider;
- caching;
- стандартизированный MetricResult.

## Этап 2 — independence

Добавить:

4. Boundary Clarity;
5. ChunkScore;
6. optional Coreference Integrity.

Потребуются:

- causal LM abstraction;
- log-probability/perplexity calculator.

## Этап 3 — HOPE

Добавить:

7. Semantic Independence;
8. Information Preservation;
9. Concept Unity;
10. aggregate HOPE.

Потребуются:

- generative LLM abstraction;
- prompt templates;
- deterministic/seeding policy;
- vector retrieval;
- synthetic sample cache.

---

# 22. Итоговая позиция

Для framework основной набор должен покрывать разные failure modes, а не пытаться свести качество чанкинга только к тематической cohesion.

Минимальный содержательно сильный стек:

```text
Size Compliance
      ↓
Block Integrity
      ↓
Intrachunk Cohesion
      ↓
Boundary / Logical Independence
      ↓
HOPE Semantic Independence
      ↓
HOPE Information Preservation
```

Он последовательно проверяет:

**размер → структуру → внутреннюю семантику → независимость → сохранение исходной информации.**

Concept Unity, DCC, Coreference Integrity и Semantic Dispersion полезно сохранять как отдельные диагностические показатели, но они не должны определять итоговое качество в одиночку.

---

# 23. Источники

Основные определения HOPE, включая Concept Unity, Semantic Independence, Information Preservation и итоговую формулу HOPE, взяты из работы:

> Henrik Brådland, Morten Goodwin, Per-Arne Andersen, Alexander S. Nossum, Aditya Gupta. **A New HOPE: Domain-agnostic Automatic Evaluation of Text Chunking.** SIGIR 2025.

Лёгкие intrinsic-метрики и дополнительные reference-free подходы (Size Compliance, Intrachunk Cohesion, Contextual Coherence, Block Integrity, coreference-related integrity, Boundary Clarity, ChunkScore) выбраны на основе проведённого в рамках проекта обзора существующих автоматических методов оценки чанкинга.
