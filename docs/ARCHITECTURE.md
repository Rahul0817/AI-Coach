# Oviora AI — Architecture

Diagrams and the reasoning behind each design. Companion to the
[README](../README.md) and the [interview guide](INTERVIEW_GUIDE.md).

---

## 1. System overview

```mermaid
graph TB
    subgraph Browser
        NEXT["Next.js 15 App Router<br/>React 19 · TypeScript strict"]
        RQ["TanStack Query<br/><i>cache + retry policy</i>"]
        SSE["SSE reader<br/><i>fetch + ReadableStream</i>"]
        SPEECH["Web Speech API<br/><i>on-device</i>"]
    end

    subgraph "FastAPI process"
        direction TB
        M1["RequestContext<br/>correlation id"]
        M2["SecurityHeaders"]
        M3["RateLimit<br/>per identity"]
        ROUTES["61 REST endpoints"]
        DI["Dependencies<br/><i>auth choke point</i>"]
        SVC["12 services"]
        REPO["Repositories<br/><i>ownership in SQL</i>"]
    end

    subgraph "AI subsystem"
        ORCH["Orchestrator"]
        SAFE["Safety screen"]
        ROUTE["Router agent"]
        SPEC["8 specialists"]
        MEM["Memory manager"]
        RET["Hybrid retriever"]
        PROV["LLM provider<br/><i>interface</i>"]
    end

    subgraph "ML subsystem"
        LOAD["Predictor<br/><i>loaded once</i>"]
        EXP["SHAP explainer<br/><i>~36ms</i>"]
    end

    subgraph Storage
        PG[("PostgreSQL 16")]
        RD[("Redis 7")]
        CH[("ChromaDB")]
        OBJ[("Object storage")]
    end

    NEXT --> RQ --> M1
    SSE --> M1
    SPEECH -.-> NEXT
    M1 --> M2 --> M3 --> ROUTES --> DI --> SVC --> REPO --> PG
    SVC --> ORCH
    ORCH --> SAFE --> ROUTE --> SPEC
    SPEC --> RET --> CH
    SPEC --> PROV
    ORCH --> MEM --> RD
    MEM --> PG
    SVC --> LOAD --> EXP
    M3 --> RD
    SVC --> OBJ
```

**Why the layers are separated this way.** Services raise domain exceptions and
never import `HTTPException`, so exactly one place maps meaning to status
codes. That is what would let the same services be driven by a CLI, a worker or
a gRPC surface without change.

---

## 2. Database schema

```mermaid
erDiagram
    USERS ||--o| PROFILES : has
    USERS ||--o{ CYCLE_LOGS : logs
    USERS ||--o{ SYMPTOM_LOGS : logs
    USERS ||--o{ PREDICTIONS : receives
    USERS ||--o{ HABITS : creates
    USERS ||--o{ MEAL_LOGS : logs
    USERS ||--o{ WORKOUT_LOGS : logs
    USERS ||--o{ WATER_LOGS : logs
    USERS ||--o{ SLEEP_LOGS : logs
    USERS ||--o{ WEIGHT_LOGS : logs
    USERS ||--o{ MOOD_LOGS : logs
    USERS ||--o{ CONVERSATIONS : owns
    USERS ||--o{ BLOOD_REPORTS : uploads
    USERS ||--o{ NOTIFICATIONS : receives
    HABITS ||--o{ HABIT_ENTRIES : "ticked on"
    CONVERSATIONS ||--o{ MESSAGES : contains
    BLOOD_REPORTS ||--o{ BIOMARKERS : "parsed into"

    USERS {
        uuid id PK
        string email UK
        string hashed_password
        int failed_login_count
        timestamp locked_until
    }
    PROFILES {
        uuid user_id FK "unique"
        date date_of_birth
        float height_cm
        float weight_kg
        jsonb allergies
        jsonb ai_memory "long-term facts"
    }
    PREDICTIONS {
        uuid id PK
        string model_version "auditability"
        jsonb features "exact input vector"
        float risk_score
        jsonb explanation "SHAP factors"
    }
    MESSAGES {
        uuid id PK
        int sequence "monotonic, MAX-derived"
        string agent "which specialist"
        jsonb sources "citations"
        int tokens_used "cost attribution"
    }
```

Eighteen tables. Design points:

- **UUID primary keys.** Generatable client-side, and they do not leak business
  volume the way `/users/1043` does.
- **`ondelete="CASCADE"` on every user-owned table.** Right-to-erasure is
  implemented at the database level, so no application bug can leave orphaned
  PHI behind.
- **Split `users` / `profiles`.** The login path reads a small hot table, and
  sensitive health attributes stay out of the row loaded on every request.
- **`predictions` stores the feature vector and model version.** Months later
  you can answer "why did the model say that?" and reproduce it exactly.
- **`messages.sequence` derived from `MAX`, not `COUNT`.** Deleting a message
  cannot then cause a collision.
- **Dialect-aware types.** `JSONB` and native `uuid` on Postgres, portable
  equivalents on SQLite, via `with_variant` — so the test suite runs against
  the same schema definition.

---

## 3. Chat request flow

```mermaid
flowchart TD
    A["POST /api/v1/chat"] --> B{"Safety screen"}
    B -->|self-harm / emergency| C["Signposted response<br/>model never called"]
    B -->|disordered eating| D["Flag: prepend support,<br/>force Mental Wellness"]
    B -->|clean| E["Router"]
    D --> E

    E --> F{"Lexical score<br/>confident?"}
    F -->|yes ~majority| G["Selected specialist"]
    F -->|ambiguous| H["LLM tie-breaker"] --> G

    G --> I["Memory: learn facts<br/><i>before</i> building context"]
    I --> J["Memory: load profile<br/>+ recent window + summary"]
    J --> K["Retrieve<br/>agent category first"]
    K --> L{"Anything<br/>relevant?"}
    L -->|no| M["Prompt says: do not invent"]
    L -->|yes| N["Prompt with cited chunks"]
    M --> O["LLM"]
    N --> O
    O --> P["enforce_disclaimer()"]
    P --> Q["Persist turn + provenance"]
    Q --> R["Response:<br/>answer · agent · citations · memory"]
```

Facts are learned **before** context is assembled, so "I'm 22, what should I
eat?" personalises *that* answer rather than only the next one.

---

## 4. Retrieval — why hybrid

```mermaid
graph LR
    Q["Query"] --> D["Dense<br/>vector search"]
    Q --> B["BM25<br/>lexical"]

    D --> DG{"score ≥<br/>provider floor?"}
    B --> BG{"score ≥<br/>BM25 floor?"}

    DG --> QUAL["Qualified set<br/><i>union</i>"]
    BG --> QUAL

    QUAL --> E{"Empty?"}
    E -->|yes| NONE["Return nothing<br/><i>assistant says 'I don't know'</i>"]
    E -->|no| RRF["Weighted RRF<br/>fuse on rank"]
    RRF --> DEDUP["Dedupe by section"]
    DEDUP --> TOPK["Top k with<br/>normalised relevance"]
```

| Failure mode | Dense | BM25 | Hybrid |
|---|---|---|---|
| "I keep skipping periods" → *oligomenorrhoea* | ✅ | ❌ | ✅ |
| "What is HOMA-IR?" (rare exact term) | ❌ | ✅ | ✅ |

**The bug this design fixed.** The first version filtered by an absolute cosine
threshold *after* truncating to top-k, and returned nothing for obviously
answerable questions. Two changes: gate before truncation, and let each
embedding provider declare its own measured floor and fusion weight — a
semantic encoder scores genuine matches around 0.3–0.6, a lexical one around
0.08, and a single global threshold silently starves one of them.

---

## 5. Memory tiers

```mermaid
graph TB
    subgraph T1["Tier 1 — Working memory · Redis"]
        W["Last 12 turns, verbatim"]
        WN["Read every message.<br/>Postgres rehydrates on a cache miss,<br/>so a flush costs one query."]
    end
    subgraph T2["Tier 2 — Rolling summary · Postgres"]
        S["Compressed older turns"]
        SN["Regenerated incrementally past a<br/>threshold, so prompt cost stays bounded<br/>however long the thread runs."]
    end
    subgraph T3["Tier 3 — Long-term facts · Postgres JSONB"]
        L["age · height · diet · goals"]
        LN["Structured, not a text blob:<br/>validatable, displayable, individually<br/>editable and deletable."]
    end

    MSG["New message"] --> EX["Deterministic extraction<br/><i>negation + third-person aware</i>"]
    EX --> T3
    MSG --> T1
    T1 -->|"past threshold"| T2
    T1 --> P["Prompt"]
    T2 --> P
    T3 --> P
```

Extraction is deterministic rather than model-based for the critical fields.
Memory is a *correctness* feature: a hallucinated age poisons every future
answer for that account, permanently.

---

## 6. ML pipeline

```mermaid
flowchart LR
    A["Load dataset<br/><i>real CSV if present,<br/>else documented synthetic</i>"] --> B["Engineer 19 features<br/><i>vectorised</i>"]
    B --> C["Stratified 80/20 split"]
    C --> D["3 pipelines<br/><i>preprocessing inside</i>"]
    D --> E["5-fold CV<br/>select on ROC-AUC"]
    E --> F["Isotonic calibration"]
    F --> G["Holdout evaluation"]
    G --> H["Persist artifact +<br/>metrics + background sample"]

    H --> I["Load once at startup"]
    I --> J["predict()"]
    J --> K["clamp to [0.02, 0.97]"]
    K --> L["SHAP explain ~36ms"]
    L --> M["Curated text + recommendations"]
```

**Preprocessing lives inside the pipeline.** Fitting a scaler outside it is the
classic source of train/serve skew — the scaler would be re-fit, or not fit at
all, at inference time.

**Four engineered features** encode clinical structure a tree would otherwise
have to rediscover from limited data: androgenic symptom *count* (the pattern
matters more than any single marker), metabolic load (a hand-built interaction
term), cycle deviation (absolute, since both 19-day and 45-day cycles are
abnormal), and lifestyle score (the modifiable slice, which is what the
coaching agents can act on).

---

## 7. Deployment topology

```mermaid
graph TB
    subgraph Internet
        U["Users"]
    end
    subgraph Edge
        CDN["CDN / TLS"]
        LB["Load balancer<br/><i>routes on /health/ready</i>"]
    end
    subgraph Compute
        W1["web · Next.js standalone"]
        A1["api · replica 1"]
        A2["api · replica 2"]
        TR["trainer<br/><i>one-shot job</i>"]
    end
    subgraph Managed
        PG[("PostgreSQL<br/>+ automated backups")]
        RD[("Redis")]
        OBJ[("Object storage")]
    end

    U --> CDN --> LB
    LB --> W1
    LB --> A1
    LB --> A2
    A1 --> PG
    A2 --> PG
    A1 --> RD
    A2 --> RD
    A1 --> OBJ
    TR -.->|writes artifact| A1
```

**One uvicorn worker per container.** The ML model and the embedded knowledge
base are loaded per process, so scaling by adding containers keeps each one's
memory footprint predictable — multiple workers would multiply that footprint
inside a single memory limit.

**Liveness and readiness are separate.** Liveness touches no dependency, so a
transient database blip cannot cause the orchestrator to kill an otherwise
healthy pod; readiness checks dependencies so the load balancer stops routing
to a pod that cannot serve.

---

## 8. Security

```mermaid
graph TB
    subgraph Transport
        T1["HTTPS + HSTS in production"]
        T2["Explicit CORS allowlist, never *"]
        T3["nosniff · DENY · Referrer-Policy"]
    end
    subgraph Identity
        I1["bcrypt, cost 12, per-password salt"]
        I2["Access 30min · refresh 14d, rotating"]
        I3["jti + Redis denylist = real logout"]
        I4["Lockout after 5 failures"]
        I5["Timing-equalised login<br/><i>no user enumeration</i>"]
    end
    subgraph Application
        A1["Pydantic validation on every input"]
        A2["Ownership filtered in SQL<br/><i>IDOR closed by construction</i>"]
        A3["Parameterised queries only"]
        A4["Rate limit, tighter on AI routes"]
        A5["Magic-byte verification on uploads"]
        A6["Generated filenames"]
    end
    subgraph AI
        S1["Crisis screen before generation"]
        S2["Disclaimer enforced after"]
        S3["Prompt sentinels neutralised"]
    end
```

| Threat | Mitigation |
|---|---|
| Credential stuffing | Lockout with backoff; strong password policy |
| User enumeration | Identical error and comparable timing on both failure paths |
| Token theft | Short access tokens; rotation makes a stolen refresh token single-use |
| IDOR | Ownership in the query, not in a handler check |
| SQL injection | ORM parameterisation throughout |
| Malicious upload | Declared MIME verified against magic bytes; generated names |
| Prompt injection | Sentinels stripped; safety rules restated last in the prompt |
| Data leakage in SSR | QueryClient per request, never at module scope |
