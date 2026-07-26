# Oviora AI — Build Progress

> Handoff note. This file records what exists, what is next, and the decisions
> that would otherwise have to be rediscovered. Delete it before the project is
> presented — it is scaffolding, not documentation.

**Branch:** `claude/oviora-ai-pcos-copilot-x244re`
**Status:** backend substantially complete; frontend not started.
**Roughly 55% of the planned scope.**

---

## Done

| Area | Location | State |
|---|---|---|
| Core (config, async DB, Redis, JWT, middleware, exceptions) | `backend/app/core/` | Complete |
| Postgres schema — 17 tables | `backend/app/models/` | Complete |
| Pydantic request/response contracts | `backend/app/schemas/` | Complete |
| Repository layer | `backend/app/repositories/` | Complete |
| ML pipeline + SHAP | `backend/ml/`, `backend/app/ml/` | Complete, trained |
| LLM providers (OpenAI + local fallback) | `backend/app/ai/llm/` | Complete |
| RAG — 9 documents, 46 chunks, hybrid retrieval | `backend/app/ai/rag/`, `backend/ml/knowledge/` | Complete |
| 8 agents + router + safety guardrails | `backend/app/ai/agents/` | Complete |
| 3-tier conversation memory | `backend/app/ai/memory/` | Complete |
| OCR, food vision, voice | `backend/app/ai/multimodal/` | Complete |
| Service layer (12 services) | `backend/app/services/` | Complete |

### Verified by running it, not by assumption

- **Model training** (`python -m ml.train`): accuracy 0.897, precision 0.866,
  recall 0.803, F1 0.833, ROC-AUC 0.922, Brier 0.082. Logistic regression won
  on cross-validated ROC-AUC.
- **SHAP**: ~36 ms per explanation.
- **Agent routing**: 10/10 on a hand-written test set.
- **Biomarker OCR**: 9/9 markers parsed correctly from a representative report,
  with correct HIGH/LOW/NORMAL flags.
- **RAG**: top-1 result correct across six probe queries.
- **Memory extraction**: handles negation ("I'm not vegetarian") and third
  person ("my sister is 30") correctly.

---

## Next, in order

1. **API routes + app assembly** — `backend/app/api/v1/routes/` and
   `backend/app/main.py`. The directories exist but are empty. Every service
   method a route needs is already written and tested; this is wiring plus
   dependency injection, error handlers, CORS, lifespan hooks, and the SSE
   endpoint for streaming chat.
2. **Frontend** — `frontend/`. Nothing exists yet. Largest remaining piece.
3. **Tests** — `backend/tests/{unit,integration,api}/`. Directories exist,
   empty. `aiosqlite` is already in `requirements-dev.txt` for the integration
   suite; `configure_engine()` in `core/database.py` exists specifically so
   tests can point the app at SQLite.
4. **Docker, GitHub Actions, Alembic migrations.**
5. **README, interview guide, architecture diagrams.**

---

## Decisions worth not re-litigating

**The app must run with no OpenAI key.** `LLM_PROVIDER=local` uses an
*extractive* composer that can only quote the curated corpus and says "I don't
have material on this" when retrieval returns nothing. This is deliberate: a
template engine that invented fluent clinical prose would be dangerous in a
health product. Same pattern for Redis and ChromaDB — both have in-process
fallbacks so a fresh clone runs.

**Safety is enforced in code, never in prompts.** Crisis and emergency language
blocks generation entirely (`app/ai/agents/safety.py`); the medical disclaimer
is appended if the model omits it, including mid-stream. Prompt instructions are
guidance; these two need guarantees.

**Plan generation is deterministic, not LLM-driven** (`plan_service.py`). A meal
plan has to add up, and allergy exclusions have to hold with certainty rather
than high probability. The LLM's job here is conversation, not arithmetic.

**Risk scores are clamped to [0.02, 0.97]** (`app/ml/predictor.py`). Isotonic
calibration saturates at exactly 1.0, and a screening questionnaire reporting
"100% risk" would be indefensible.

**The dataset is synthetic** (`ml/dataset.py`), generated from a documented
causal model, because real PCOS datasets cannot be redistributed in a repo. The
metrics show the pipeline recovers the generative structure — they are **not**
clinical validation, and must never be presented as such. Dropping a licensed
CSV at `ml/data/pcos_dataset.csv` switches it over with no code change.

---

## Bugs found by testing, already fixed

Recorded because each is a plausible regression.

1. **RAG returned nothing for answerable questions.** The score filter ran
   *after* truncating to top-k, so good hits pushed down by a weak retriever
   were discarded. Fixed by gating before truncation and letting each embedding
   provider declare its own measured score floor and fusion weight.
2. **OCR mis-flagged converted lab values.** The value's units were converted
   but its reference range was not, so a testosterone result in nmol/L was
   compared against an unconverted range.
3. **SHAP took 7.5 s per explanation** in the naive configuration — unusable in
   a request path. Measured, then fixed with a k-means-compressed background.
4. **The SQLAlchemy engine was created at import time**, making the app
   un-importable without `asyncpg` installed. Now lazy.

---

## Running what exists

```bash
cd backend
pip install -r requirements-dev.txt
python -m ml.train          # writes ml/artifacts/pcos_risk_model.joblib
```

The trained artifact is gitignored, so it must be regenerated after a fresh
clone. Training takes about 30 seconds.
