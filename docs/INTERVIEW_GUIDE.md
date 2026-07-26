# Oviora AI — Interview Guide

Everything needed to present and defend this project. Read the
[architecture doc](ARCHITECTURE.md) alongside it for the diagrams.

**A note on how to use this.** The strongest thing about this project in an
interview is not the feature list — it is that most decisions were *measured*,
several were *wrong first*, and the fixes are documented. Lead with that.

---

## Contents

1. [The 5-minute walkthrough](#the-5-minute-walkthrough)
2. [Business problem](#business-problem)
3. [Why each technology](#why-each-technology)
4. [How the ML pipeline works](#how-the-ml-pipeline-works)
5. [How RAG works](#how-rag-works)
6. [How the multi-agent system works](#how-the-multi-agent-system-works)
7. [How memory works](#how-memory-works)
8. [How explainable AI works](#how-explainable-ai-works)
9. [How OCR works](#how-ocr-works)
10. [How image AI works](#how-image-ai-works)
11. [Bugs found by testing](#bugs-found-by-testing)
12. [20 interview questions](#20-interview-questions)
13. [Questions to ask them](#questions-to-ask-them)

---

## The 5-minute walkthrough

> Rehearse this until it is natural. Time each section.

### 0:00–0:45 — The problem

"PCOS affects roughly one in eight women of reproductive age, and most wait
over two years and see three or more doctors before being diagnosed. The reason
is structural: every individual symptom has a boring explanation. Irregular
periods get blamed on stress, acne on being young, weight gain on lifestyle.
It's the *combination* that's diagnostic — and nobody is tracking the
combination.

So I built Oviora: an AI copilot that remembers your history, explains what's
happening in plain language, and turns 'my periods are a bit irregular' into
three months of structured data a clinician can actually act on. It explicitly
does **not** diagnose — that boundary is enforced in code, not just in the
prompt."

### 0:45–1:45 — Architecture

"FastAPI backend, Next.js frontend, Postgres, Redis, ChromaDB. Clean layering:
routes handle HTTP, services hold business logic, repositories own data access.
Services raise domain exceptions and never import `HTTPException`, so exactly
one handler maps meaning to status codes.

Three things I'd point at specifically.

First, the repository layer enforces ownership *in SQL* — `get_for_user`
filters by owner in the query. That closes IDOR by construction rather than
relying on every handler remembering to check.

Second, everything degrades gracefully. Redis, ChromaDB and the LLM all have
working fallbacks, so the whole app runs with no API key at all. A missing ML
artifact disables prediction endpoints and nothing else.

Third, 61 endpoints, all documented in Swagger, and 182 tests."

### 1:45–3:00 — The AI layer

"Four parts.

**Multi-agent.** Eight specialists behind a router. The routing is two-tier: a
weighted lexical classifier resolves the large majority of queries in under a
millisecond for free, and the LLM is only consulted when the wording is
genuinely ambiguous. Asking a model 'which agent?' on every message would
double latency and cost to answer a question that's usually trivial. It scores
10 out of 10 on my routing test set.

**RAG.** Hybrid retrieval — dense vectors fused with BM25 by Reciprocal Rank
Fusion. Dense handles paraphrase, BM25 handles rare exact terms like 'HOMA-IR'.
Fusing on rank rather than score matters because cosine similarity and BM25
live on incomparable scales. If nothing relevant is found, the assistant says
so instead of inventing an answer.

**Memory.** Three tiers: a verbatim window in Redis, a rolling summary in
Postgres, and structured long-term facts on the profile. Say 'I'm 22' and
Friday's meal suggestion accounts for it. Fact extraction is deterministic, not
LLM-based — memory is a correctness feature, and a hallucinated age would
poison every future answer for that account.

**Safety.** Crisis language never reaches the model; a signposted response with
real helpline numbers is returned instead. The medical disclaimer is verified
after generation and appended if missing, including mid-stream."

### 3:00–4:00 — The ML

"A nine-stage pipeline: load, engineer, stratified split, build three
pipelines, cross-validate, select, calibrate, evaluate, persist.

I compare Logistic Regression, Random Forest and XGBoost by cross-validated
ROC-AUC — AUC rather than accuracy because the classes are imbalanced and
accuracy would reward predicting the majority. Logistic regression wins at
0.930; on holdout it's 0.897 accuracy, 0.833 F1, 0.922 AUC.

Then I calibrate with isotonic regression, because the score is shown to a user
as a percentage — '70% risk' has to actually mean something, not just rank
correctly. Brier score 0.082.

Two details I'd highlight. I clamp the output to 0.02–0.97, because isotonic
calibration saturates at exactly 1.0 and telling someone a questionnaire proves
100% certainty is indefensible. And SHAP: the naive configuration took 7.5
seconds per explanation, which I only found by measuring. Compressing the
background to 20 k-means centroids gives the same ranked factors in 36
milliseconds."

### 4:00–4:40 — What went wrong

"The part I'd actually want to talk about is the bugs the tests caught.

My feature engineering has two implementations — vectorised for training,
scalar for inference. A parity test on 200 random rows caught them disagreeing
in the fourth decimal on about 0.5% of cases, because Python floats and numpy
float64 round half-way values differently. That's textbook train/serve skew:
silent, and it would have meant the model scored well offline and behaved
differently in production.

Another one: my cascade-deletion test was passing vacuously. SQLite ships with
foreign keys *disabled*, so every `ondelete=CASCADE` in the schema was
unverified — including the one implementing right-to-erasure. The test asserted
a compliance guarantee and proved nothing."

### 4:40–5:00 — Honest limitations

"The most important caveat: the model trains on a **synthetic** cohort. Real
PCOS datasets have licences that forbid redistribution in a repo, so I simulate
one from a documented causal model. The metrics show the pipeline recovers the
generative structure — they are **not** clinical validation, and I'd never
present them as such. Swapping in a licensed dataset is a one-file change.

The knowledge base is nine curated documents, not a medical reference, and
none of it has had clinician review. For a real product in this space, that
review is table stakes."

---

## Business problem

| Dimension | Detail |
|---|---|
| **Condition** | PCOS — 8–13% prevalence in reproductive-age women; a large share undiagnosed |
| **Pain** | Multi-year diagnostic delay; fragmented advice; symptoms dismissed individually |
| **Existing tools** | Period trackers do not understand PCOS; general health apps do not track cycles; forums are unreliable |
| **Wedge** | The *combination* of symptoms is the signal, and only a longitudinal tracker sees it |
| **Value** | Turn scattered symptoms into structured data a clinician can act on, with education that makes the appointment productive |
| **Why now** | LLMs make patient-grade explanation possible at zero marginal cost; RAG makes it groundable |

**Deliberately not built:** diagnosis, prescription, or anything that would
make this a regulated medical device.

---

## Why each technology

| Choice | Reason | What I rejected, and why |
|---|---|---|
| **FastAPI** | Async matters when most request time is spent waiting on an LLM; Pydantic gives validation and OpenAPI from one type definition | Django REST — heavier, sync-first, ORM less suited to async |
| **PostgreSQL** | JSONB for flexible AI memory *plus* relational integrity for health records; transactional DDL | MongoDB — health data is highly relational, and cascade deletes matter for erasure |
| **SQLAlchemy 2.0 async** | Mature, typed, with real migration tooling | Raw asyncpg — no migrations, hand-written mapping |
| **Redis** | Sub-ms reads for the memory window read on every message; atomic INCR for rate limiting | In-process only — breaks across workers |
| **ChromaDB** | Embedded, persistent, no separate server at this scale | pgvector is the better production answer at scale; Pinecone adds a hosted dependency for a corpus of 46 chunks |
| **scikit-learn + XGBoost** | Pipelines that bundle preprocessing with the model, which is what prevents train/serve skew | Deep learning — 6k rows, 19 tabular features; a neural net would overfit and lose interpretability |
| **SHAP** | The only widely-used method with an additivity guarantee: contributions sum exactly to the gap from baseline, so nothing hides in a remainder | LIME — local surrogate, less stable; feature importance — global, cannot answer "why *me*?" |
| **Next.js 15** | SSR for the landing page, App Router, standalone Docker output | Plain React SPA — worse first paint and SEO on the page most likely to be someone's first impression |
| **TanStack Query** | Caching, retry policy and request dedup without hand-rolling it | Redux — heavy state machinery for what is mostly server cache |
| **Chart.js** | Small, canvas-based, fine for these chart types | D3 — far more power than six line charts need |

---

## How the ML pipeline works

**1. Data.** `ml/dataset.py` prefers a real CSV; absent one, it simulates a
cohort from an explicit causal model: sample age → sample latent PCOS status at
an age-modulated prevalence → sample BMI conditional on status → sample each
symptom from literature-informed conditional probabilities → sample lifestyle
variables with BMI feeding back so features correlate *with each other* → flip
4% of labels for diagnostic uncertainty.

That last step is deliberate. A model hitting 99% on this data would indicate
leakage, not skill.

**2. Features.** Fifteen raw inputs plus four engineered:

- `androgenic_symptom_count` — hyperandrogenism is diagnosed on a *pattern*, so
  the sum carries more information than the flags individually
- `metabolic_load` — BMI, diet and inactivity compressed into one index; a
  hand-built interaction term
- `cycle_deviation` — absolute distance from 28 days, because both 19-day and
  45-day cycles are abnormal and raw length cannot express that monotonically
- `lifestyle_score` — the modifiable slice, which is what the coaching agents
  can actually act on

**3. Pipeline.** Preprocessing lives *inside* the sklearn `Pipeline` — median
imputation and scaling for continuous columns, most-frequent for binary.
Fitting a scaler outside is the classic train/serve skew bug.

**4. Selection.** 5-fold stratified CV on the training split only, ranked by
mean ROC-AUC.

**5. Calibration.** Isotonic regression via `CalibratedClassifierCV`, because
the number is shown to a user as a percentage.

**6. Evaluation.** One pass on the holdout: accuracy, precision, recall, F1,
ROC-AUC, PR-AUC and Brier score.

**7. Artifact.** The fitted pipeline plus metrics, feature list, ROC points,
confusion matrix, permutation importance, training medians and a
class-stratified SHAP background — all in one joblib file, so the served model
and its explanation baseline can never drift apart.

---

## How RAG works

**Ingestion.** Nine Markdown documents → heading-aware chunking (a `##` is a
semantic boundary the author already provided) → 46 chunks, each prefixed with
its document title and section so a retrieved fragment is self-describing →
embedded → stored in ChromaDB with a parallel in-memory BM25 index.

**Retrieval.**
1. Embed the query; fetch `k × 3` dense candidates.
2. Score the same corpus with BM25.
3. **Quality gate, before truncation.** A candidate survives if *either*
   retriever considers it real evidence.
4. If nothing qualifies, return empty — and the prompt then instructs the model
   to say it does not know.
5. Fuse survivors with weighted RRF: `Σ weight / (60 + rank)`.
6. Dedupe by section, return top-k with relevance normalised against the best
   hit.

**Why RRF.** Cosine similarity and BM25 scores are not comparable. Any attempt
to normalise and weight them directly is fragile. RRF merges on *rank*, needs
no tuning, and is robust to one retriever producing wild scores.

**Why each provider declares its own floor.** Measured, not guessed: a semantic
encoder puts genuine matches at 0.3–0.6; the lexical hashing embedder puts the
*same* matches near 0.08, because a five-word query activates a handful of the
768 buckets a 1,400-character chunk spreads across. One global threshold
silently starves one of them.

---

## How the multi-agent system works

**Router.** Weighted lexical scoring: keywords worth 1.0, multi-word phrases
3.5 (far more discriminative), a 1.6× continuity boost for the agent that
answered last — applied *only* to an agent that already has signal, so
continuity smooths follow-ups without trapping the conversation. Attachments
short-circuit entirely: a photo goes to the Food Analyzer, a report to the
Blood Report Analyzer.

If the top score is below the confidence bar, or the top two are within 10%,
the LLM is asked to classify. That is the only path that costs a call.

**Specialists.** A template-method base class handles retrieve → prompt →
generate → enforce. Three agents genuinely need different behaviour and get
subclasses: the Blood Report Analyzer runs at temperature 0 (there is one
correct explanation of what SHBG measures), the Food Analyzer works from vision
output rather than retrieval, and the Cycle Assistant appends a contraception
caveat in code.

**Prompt assembly is centralised** so the safety rules are impossible to omit.
Adding a ninth agent cannot ship one without them.

---

## How memory works

The problem: an LLM has no memory between calls. Everything it "knows" must be
reconstructed and re-sent every turn. Replaying the whole thread hits the
context limit and grows cost quadratically.

| Tier | Store | Solves |
|---|---|---|
| Working window | Redis, last 12 turns verbatim | Immediate follow-ups ("and for dinner?") |
| Rolling summary | Postgres, regenerated incrementally | Unbounded thread length at bounded prompt cost |
| Long-term facts | Postgres JSONB, structured | Persistence across conversations and sessions |

**Why structured, not a text blob.** Structured facts can be validated,
displayed to the user, edited and deleted individually — which matters when the
data is health information and the user has a right to correct it. The UI shows
"Noted: age = 22" for exactly that reason: memory the user cannot see is memory
they cannot correct.

**Why deterministic extraction.** Regex handles the high-value fields — age,
height, weight, diet, diagnosis status, allergies, goals — with negation
handling ("I'm *not* vegetarian") and third-person rejection ("my *sister* is
30"). An LLM supplements it on a schedule. Memory is a correctness feature; a
model that occasionally hallucinates an age into permanent storage produces
confidently wrong answers forever.

---

## How explainable AI works

**Why SHAP over feature importance.** Global importance answers "what does this
model care about generally?" A user asking "why is *my* risk elevated?" needs a
*local* explanation. SHAP's additivity guarantee means the contributions sum
exactly to the gap between this person's score and the population baseline —
nothing hides in a remainder.

**Which explainer.** The served model is a `CalibratedClassifierCV` wrapping a
`Pipeline` containing imputation and scaling. TreeSHAP cannot see through that,
and the selected model is not always a tree. So: the model-agnostic Permutation
explainer over `predict_proba`, with a class-stratified background sampled at
training time and shipped inside the artifact.

**The performance work.** Naive configuration — 120 background rows, generous
eval budget — measured **7.5 seconds**. Unusable. Compressing to 20 k-means
centroids (weighted by cluster size, so the baseline still reflects the real
distribution) returns the same ranked factors in **~36 ms**.

**Safety by construction.** The explanation *text* is composed from a curated
table in `ml/features.py`, never generated by an LLM. The app therefore cannot
hallucinate a clinical claim about a user's own data.

**Recommendations filter on `modifiable`.** Telling someone whose top driver is
family history to "lower your BMI" is useless.

---

## How OCR works

**Cheapest path first.**
1. Digital PDF → read the embedded text layer with `pypdf`. Most lab reports
   emailed to patients are digitally generated and already contain perfect
   text. Running OCR on them would be slower *and* less accurate.
2. Scanned PDF → rasterise, then Tesseract.
3. Image → Tesseract directly, with per-word confidence surfaced honestly.

**Parsing.** An alias index collapses "Testosterone, Total", "TOTAL
TESTOSTERONE" and "T. Testosterone" to one canonical key. Regexes handle the
layouts real reports use: value and range on one line, range on the next,
`12 - 45`, `< 150`, `Ref: 0.4-4.0`.

**The range rule.** The range printed on *your* report always wins; built-in
ranges are a labelled fallback. Ranges are assay- and laboratory-specific, so
preferring a hard-coded one produces confidently wrong flags.

Verified at 9/9 markers on a representative report with correct flags.

---

## How image AI works

Two providers behind one interface. With a vision model configured, the image
is analysed directly. Without one, **the response says so explicitly** and the
breakdown is built from a user-supplied description.

That honesty is the design point. A system with no vision capability claiming
"I can see grilled chicken" would be lying to the user about their own health
data. A fallback that says "tell me what's on the plate and I'll do the
nutrition maths" is truthful and still useful — the food database, portion
inference, glycaemic-load scoring and swap suggestions are identical on both
paths, and that is where most of the value is.

**Scoring** is a gram-weighted mean of per-food PCOS scores, adjusted for two
whole-meal properties no ingredient captures: glycaemic load, and whether there
is enough protein to blunt the glucose response.

---

## Bugs found by testing

Worth memorising — these are the most credible thing in an interview, because
they show the tests do real work.

### 1. Train/serve skew in feature engineering
Vectorised (training) and scalar (inference) implementations disagreed in the
fourth decimal on ~0.5% of rows, because Python floats and numpy float64 round
half-way values differently. **Silent** — the model would score well offline
and behave differently in production. Fixed by removing rounding from both;
precision is free, and rounding is a display concern. A parity test on 200
randomised rows now pins them together.

### 2. Cascade-deletion test passing vacuously
SQLite ships with foreign keys **disabled**. Every `ondelete="CASCADE"` in the
schema was unverified — including the one implementing right-to-erasure. The
test asserted a compliance guarantee and proved nothing. Fixed by enabling the
pragma per connection.

### 3. RAG returning nothing for answerable questions
The score filter ran *after* truncating to top-k, so good hits pushed down by
the weaker retriever were discarded. Compounded by a single global cosine
threshold across providers with different score scales. Fixed by gating before
truncation and having each provider declare a measured floor.

### 4. Every PATCH/PUT raising `MissingGreenlet`
Columns with a server-side `onupdate` are expired after UPDATE; reading one
triggers implicit IO, which asyncio forbids. Would have broken every endpoint
returning a timestamped row. Fixed with `eager_defaults` so those values come
back via RETURNING.

### 5. OCR mis-flagging converted values
The value's units were converted but its reference range was not, so a
testosterone result in nmol/L was compared against an unconverted range.

### 6. SHAP at 7.5 seconds per explanation
Found by measuring rather than assuming. Fixed to ~36 ms.

---

## 20 interview questions

**1. Why multiple agents instead of one good system prompt?**
Specialisation lets each agent carry a different persona, temperature and
retrieval scope — the Blood Report Analyzer runs at temperature 0, the Mental
Wellness Coach at 0.6. It also constrains scope, which reduces drift, and it
makes the system legible to the user: they can see which specialist answered.
The cost is routing, which I made nearly free by making it lexical-first.

**2. Why not let the LLM route every message?**
Cost and latency, for a decision that's usually trivial. "What should I eat for
breakfast?" doesn't need a language model to classify. My lexical classifier
gets 10/10 on the test set in under a millisecond; the LLM is a tie-breaker for
genuine ambiguity. Same "cheap path first" pattern I used throughout.

**3. How do you prevent the AI giving dangerous medical advice?**
Four layers. Retrieval grounding, so factual claims come from a curated corpus.
A pre-generation safety screen that blocks crisis and emergency language before
the model is called. A post-generation check that guarantees the disclaimer.
And explanation text for predictions composed from a curated table rather than
generated. The key principle: for the things that must never fail, enforcement
is in code, not in the prompt.

**4. Why hybrid retrieval?**
Dense embeddings handle paraphrase but miss rare exact terms — a query for
"HOMA-IR" barely moves a document-level embedding. BM25 is the mirror image.
Running both covers both failure modes. I fuse with RRF because cosine and BM25
scores aren't comparable, and RRF merges on rank so it needs no tuning.

**5. Your model is logistic regression. Isn't that too simple?**
It won on cross-validated ROC-AUC against Random Forest and XGBoost — and I'd
argue that's the expected result, not a disappointment. My generative process
is largely additive, so a linear model recovers it while the ensembles overfit
slightly. It's also faster, more interpretable, and better calibrated. Picking
the ensemble because it sounds more impressive would be choosing the model that
lost.

**6. How do you know your model isn't just memorising?**
Stratified train/test split with evaluation once on holdout. Cross-validation
for selection on the training split only. I inject 4% label noise deliberately,
so a 99% score would be a red flag for leakage rather than a good result. And
the reported metrics are honest about being on synthetic data.

**7. Why calibrate?**
Because I show the number as a percentage. An uncalibrated tree ensemble ranks
well but its probabilities are meaningless — "70%" doesn't happen 70% of the
time. Isotonic calibration fixes that; Brier score went to 0.082. Related: I
clamp to 0.02–0.97 because isotonic saturates at exactly 1.0, and a
questionnaire reporting 100% certainty would be indefensible.

**8. Walk me through the memory implementation.**
Three tiers. Redis holds the last 12 turns verbatim for immediate follow-ups.
Postgres holds a rolling summary, regenerated incrementally past a threshold so
long threads stay affordable. And structured long-term facts on the profile.
Extraction is deterministic for the critical fields, because a hallucinated age
would poison every future answer for that account permanently.

**9. How does the app work without an OpenAI key?**
Every external dependency has a fallback: an in-process cache for Redis,
brute-force cosine for ChromaDB, and an extractive composer for the LLM. The
composer ranks sentences from retrieved chunks with BM25 and assembles a
grounded answer. It's deliberately extractive — a template engine that invented
fluent clinical prose would be actively dangerous. If retrieval finds nothing,
it says so.

**10. Why deterministic plan generation instead of the LLM?**
A meal plan has to add up, and an allergy exclusion has to hold with certainty.
An LLM asked for 1,800 kcal will confidently produce 2,340 and the user can't
tell. Macros come from the same food database the tracker uses, so the plan's
numbers are the numbers that get logged. The LLM's job here is conversation;
where arithmetic and hard constraints are the value, code wins.

**11. How do you prevent one user reading another's data?**
Ownership is enforced in SQL. `get_for_user(id, user_id)` filters by owner in
the query, so a handler that forgets to check still can't leak. That closes
IDOR by construction rather than by code review, and there's an API test that
asserts a second user gets a 404 on someone else's conversation.

**12. Walk me through your authentication.**
bcrypt at cost 12 with per-password salts. Short access tokens (30 min),
longer refresh tokens (14 days) that rotate on use, so a stolen refresh token
is single-use. Every token carries a `jti`, and logout adds it to a Redis
denylist with a TTL matching its expiry — that gives stateless JWTs a real
logout. Lockout after five failures. And login is timing-equalised against a
dummy hash, so a wrong email and a wrong password are indistinguishable.

**13. What breaks first at 100,000 users?**
Chat throughput, bounded by the LLM. Then the dashboard aggregation — it's
several queries per load and would need a materialised summary table. Then
ChromaDB, which I'd move to pgvector or a managed store. The API scales
horizontally; the database is the bottleneck that needs read replicas.

**14. How do you test AI features that are non-deterministic?**
Separate what's deterministic from what isn't. Routing, safety screening,
memory extraction and retrieval ranking are all deterministic and directly
testable. For generation I test *invariants* rather than exact output: that the
disclaimer is present, that crisis input never reaches the model, that
citations are attached. And the local provider is fully deterministic, so the
whole pipeline runs reproducibly in CI.

**15. Biggest technical mistake, and what you learned?**
The train/serve skew. I wrote two implementations of feature engineering for
performance and assumed they agreed. A parity test on 200 random rows found
them differing in the fourth decimal on ~0.5% of cases — Python float versus
numpy float64 rounding. It's the textbook silent ML bug. What I learned is
that in ML the dangerous failures are the ones that don't raise: I now treat
any duplicated computation as something that needs a pinning test.

**16. Why Postgres over MongoDB for health data?**
The data is highly relational — a user has cycles, symptoms, predictions,
conversations — and referential integrity matters. Cascade deletes implement
right-to-erasure at the database level. Postgres also gives me JSONB where I
genuinely want schemaless storage, like AI memory. I get both without giving up
transactions.

**17. How would you handle a user in crisis?**
It's implemented, not hypothetical. Language suggesting self-harm is detected
before generation; the model is never called. The user gets a signposted
response with real helpline numbers for several countries. It's in code rather
than the prompt because "the model will usually handle this well" is the wrong
reliability target when the failure mode is someone in danger receiving a
paragraph about glycaemic load.

**18. What are the ethical risks?**
Four I'd name. False reassurance — mitigated by never diagnosing and always
directing to a clinician. Anxiety amplification — mitigated by calibration,
clamping, and leading with positive insights. Disordered eating, which is
elevated in this population — the app detects signals, never prescribes
restriction, and surfaces support resources. And data sensitivity — hence
export, deletion, and cascade-enforced erasure.

**19. What would you do differently starting over?**
Three things. Move tokens to httpOnly cookies behind a same-origin BFF from the
start. Write the parity test before the second implementation of feature
engineering rather than after. And build the evaluation harness for retrieval
quality earlier — I tuned the retriever by inspecting results, which worked but
doesn't scale past a few dozen queries.

**20. Why should we hire you based on this?**
Because most of what's interesting here isn't the feature list — it's that the
decisions are measured and the mistakes are documented. I found the SHAP
latency by profiling, not guessing. I found four real bugs with tests, and one
of those tests was itself proving nothing until I noticed SQLite had foreign
keys off. And I'm upfront that the dataset is synthetic and the metrics aren't
clinical validation, because a project that hides that is harder to trust than
one that says it.

---

## Questions to ask them

- How do you handle model versioning and rollback in production?
- What's the process when a model degrades — who notices, and how?
- How do engineering and clinical/domain expertise interact here?
- What does the on-call rotation look like for ML services?
- How much of the ML work is modelling versus data and infrastructure?
