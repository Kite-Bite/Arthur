# Architecture

Arthur is a single Python package with four hard boundaries: **interfaces** (CLI,
API), an **agent** that turns natural language into typed tool calls, an
**execution** layer that decides whether a call is allowed, and **data** stores
(SQLite for structured state, a vector index for retrieval).

```mermaid
flowchart TD
    subgraph interfaces["Interfaces (thin)"]
        CLI["cli/ — Typer + Rich"]
        API["api/ — FastAPI"]
    end

    subgraph core["Core"]
        SVC["services.py — composition root"]
        ORCH["agent/ — orchestrator + parser"]
        EX["execution/ — ToolExecutor"]
        POL["security/ — policy, paths, confirmation"]
    end

    subgraph stores["Data"]
        DB[("database/ — SQLite via SQLAlchemy")]
        VS[("retrieval/ — vector index")]
        MEM[("memory/ — long-term memory")]
        AUD[("logging/ — audit sink")]
    end

    subgraph model["Local model"]
        LLM["llm/ — Ollama client + prompts"]
        EM["llm/ — embedders"]
    end

    CLI --> SVC
    API --> SVC
    SVC --> ORCH
    SVC --> EX
    ORCH --> LLM
    ORCH --> EX
    EX --> POL
    EX --> AUD
    ORCH --> MEM
    ORCH --> VS
    MEM --> DB
    AUD --> DB
    SVC --> EM
```

**Rule of thumb:** `cli/` and `api/` contain no business logic. They format
input, call `Services`, and render the result. If a behaviour exists in both
interfaces, it belongs one layer down.

---

## Composition root

`src/arthur/services.py` is the only place that constructs objects. `Services.create()`
loads configuration, opens the database, builds the embedder and vector store,
registers tools, and wires the executor's audit sink — in that order. Both
`arthur` (CLI) and `create_app()` (API) call it, so a tool behaves identically
regardless of which surface invoked it.

```mermaid
flowchart LR
    ENV["env + ~/.config/arthur/config.toml"] --> CFG["Config (Pydantic)"]
    CFG --> S["Services.create()"]
    S --> REG["default_registry()"]
    S --> POL["SecurityPolicy"]
    S --> EMB["build_embedder()"]
    EMB --> VS["build_vector_store()"]
    S --> EX["ToolExecutor"]
    EX --> AUD["AuditLogger → SQLite"]
    S --> AG["Agent"]
    AG --> CLI2["CLI"]
    AG --> API2["API"]
```

Tests build a real `Services` graph with an in-memory database and a scripted
LLM, which is why integration tests exercise the actual policy and audit trail
rather than mocks.

---

## The agent loop

Arthur uses a **model-agnostic JSON decision protocol** instead of provider
tool-calling APIs. Any instruct model served by Ollama works, and every
decision is parseable, testable and auditable.

```mermaid
flowchart TD
    REQ["user request"] --> PRE["recall memories + auto-retrieve passages"]
    PRE --> SYS["system prompt: protocol · tool menu · examples"]
    SYS --> DECIDE{"decide<br/>(greedy, JSON)"}

    DECIDE -->|"tool + args"| VALID{"structurally<br/>valid?"}
    VALID -->|"no"| REPAIR["feed INVALID ACTION back"]
    REPAIR --> DECIDE
    VALID -->|"yes"| ISSUE{"decision_issue()<br/>meaningful?"}
    ISSUE -->|"plan with no tool, etc."| REPAIR
    ISSUE -->|"ok"| EXEC["executor.execute()"]

    EXEC -->|"confirmation_required"| PENDING["stop cleanly:<br/>result.pending_confirmation"]
    EXEC -->|"success / denied / error"| OBS["append OBSERVATION"]
    OBS -->|"more steps needed"| DECIDE
    OBS --> ANSWER

    DECIDE -->|"tool: null"| ANSWER["answer phase<br/>(streamed prose)"]
    ANSWER --> CITE["sanitize_citations()<br/>strip unverified paths"]
    CITE --> OUT["AgentResult"]
```

### Why decisions are a separate phase

1. **Portability** — no dependency on OpenAI-style tool schemas.
2. **Repairability** — a malformed reply is fed back as a bounded correction
   (`max_repairs`, default 2) instead of crashing the run.
3. **Streaming** — only the final answer streams to the user; the JSON
   scaffolding never reaches the terminal.

### Reliability measures

| Problem | Countermeasure |
| --- | --- |
| Model wraps JSON in prose or fences | `_extract_json_object` finds the first balanced object |
| Truncated JSON (small `num_ctx`) | `num_ctx=8192`, compact tool menu, greedy decisions |
| Structurally valid but meaningless output (`plan` with no tool) | `decision_issue()` routes it through the same repair path |
| Repeated failures | bounded by `max_repairs`, then degrade to `parse_failure` with a best-effort answer |
| Model skips retrieval and invents facts | `agent.auto_retrieve` injects passages before the first decision |
| Model echoes its own JSON in the answer | decision transcripts are dropped from the answer context |
| Invented document paths | `sanitize_citations()` removes anything not actually retrieved |

### Bounded work

`max_steps` (default 6) caps the decide→act cycle. Hitting it appends an
explicit "best final answer now" instruction rather than looping forever. Every
tool additionally runs under a per-tool `timeout` enforced with a worker thread.

---

## Execution pipeline

`ToolExecutor.execute()` runs six ordered stages. Each returns a structured
`ExecutionOutcome` — **the executor never raises for policy outcomes** — and
every invocation, including denials, writes an `ExecutionRecord`.

```mermaid
flowchart LR
    A["1 lookup<br/>unknown_tool"] --> B["2 validate<br/>Pydantic args<br/>reject unknown fields"]
    B --> C["3 policy<br/>permission + paths<br/>errors fail closed"]
    C --> D["4 confirm<br/>callback · approve ·<br/>deny · pending"]
    D --> E["5 run<br/>thread + timeout"]
    E --> F["6 audit<br/>append-only record"]
```

Statuses: `success`, `error`, `denied`, `declined`, `confirmation_required`,
`invalid_args`, `timeout`, `unknown_tool`.

### Confirmation semantics

`confirm` accepts four shapes, which is what lets one executor serve an
interactive CLI, a one-shot `--yes`, and a stateless HTTP request:

| Value | Meaning | Outcome |
| --- | --- | --- |
| callable | Ask the human (CLI prompt) | proceeds or `declined` |
| `True` | Approved (`--yes`, API `confirm: true`) | runs, audited `confirmed=True` |
| `False` | Declined | `declined` without prompting |
| `None` | Nobody to ask | `confirmation_required` + `request`; agent stops cleanly and the API re-posts with `confirm: true` |

---

## Data model

```mermaid
erDiagram
    CONVERSATION ||--o{ MESSAGE : "chat history"
    DOCUMENT ||--o{ DOCUMENT_CHUNK : "indexed file"
    MEMORY_RECORD ||--o| MEMORY_ITEM : "long-term memory"
    TOOL_EXECUTION ||--o| AUDIT_LOG : "append-only trail"

    CONVERSATION {
        text id PK
        text title
        datetime created_at
    }
    MESSAGE {
        int id PK
        text conversation_id FK
        text role
        text content
    }
    DOCUMENT {
        text path PK
        text title
        text format
        int chunk_count
    }
    DOCUMENT_CHUNK {
        int id PK
        text source FK
        int chunk_index
        blob embedding
        text text
    }
    MEMORY_RECORD {
        int id PK
        text content
        text category
        float importance
        datetime expires_at
    }
    TOOL_EXECUTION {
        text request_id
        text tool_name
        text permission
        bool confirmed
        text status
        float duration_ms
    }
```

Chat history, documents, memory and audit records share one SQLite file
(`~/.local/share/arthur/arthur.db`). Conversation replay and memory recall are
plain SQL; only document retrieval goes through the vector store.

### Vector store abstraction

```mermaid
flowchart LR
    R["Retriever"] --> I["VectorStore protocol<br/>add · query · delete_source · count"]
    I --> S1["SqliteVectorStore<br/>BLOBs + NumPy cosine — default, zero deps"]
    I --> S2["ChromaVectorStore<br/>optional extra (--extra chroma)"]
```

The backend is a config switch (`retrieval.backend`), so swapping in Chroma or
FAISS is a local change: implement `add`, `query`, `delete_source`, `count`.

Embeddings follow the same pattern — `OllamaEmbedder` (`nomic-embed-text`) by
default, `HashEmbedder` (blake2b) for offline runs and tests, selected by
`retrieval.embedder`.

---

## Security layer

The policy sits *between* argument validation and execution, so no tool can run
without being classified. Details in [`security.md`](security.md).

```mermaid
flowchart TD
    ARGS["validated args"] --> LEV{"tool level<br/>(config override wins)"}
    LEV -->|"DENIED"| X1["never runs"]
    LEV -->|"RESTRICTED"| OPT{"allow_restricted?"}
    OPT -->|no| X1
    OPT -->|yes| PATH
    LEV -->|"CONFIRM / SAFE"| PATH{"PathPolicy<br/>resolve → contain → deny rules"}
    PATH -->|violation| X1
    PATH -->|ok| DEL{"destructive?"}
    DEL -->|yes| STRICT["check_deletable:<br/>a root can never be deleted"]
    DEL -->|no| CONF
    STRICT -->|violation| X1
    STRICT --> CONF{"confirmation required?"}
    CONF -->|yes| GATE{"a human decides"}
    CONF -->|no| RUN["execute"]
    GATE -->|approved| RUN
```

---

## Extension points

| To add… | Do this |
| --- | --- |
| A tool | Subclass `Tool[ArgsModel]` in `src/arthur/tools/`, set the `ClassVar`s (`name`, `permission`, `action`, `risk`, `args_model`), implement `run()`, register it in `default_registry()` |
| An LLM backend | Implement `LLMClient` (`complete`, `stream`, `health`) and construct it in `Services` — the agent only sees the protocol |
| A vector backend | Implement the `VectorStore` protocol and add it to `build_vector_store()` |
| An embedding backend | Implement `Embedder` and add it to `build_embedder()` |
| A config field | Add it to the relevant Pydantic section in `config/schema.py`; `ARTHUR_<SECTION>_<FIELD>` env support follows automatically |
| An API endpoint | Add a thin handler in `api/app.py` that calls `Services` |

Adding a tool is the common case, so the contract is deliberately minimal:

```python
class ReadFileTool(Tool[ReadFileArgs]):
    name = "read_file"
    description = "Read a text file (with line numbers)."
    action = "Read a file"
    permission = PermissionLevel.SAFE
    args_model = ReadFileArgs

    def run(self, args: ReadFileArgs, ctx: ToolContext) -> ToolResult:
        text = ctx.paths.resolve(args.path).read_text(encoding="utf-8")
        return ToolResult(summary=f"{len(text)} bytes", data={"text": text})
```

`permission` is the only line that decides whether a human is asked; the path
policy applies automatically because `ToolExecutor` routes every path argument
through it.

---

## Module map

| Package | Responsibility | Key module |
| --- | --- | --- |
| `agent/` | The loop and the decision protocol | `orchestrator.py`, `parser.py` |
| `execution/` | Validate → permission → confirm → run → audit | `executor.py`, `types.py` |
| `security/` | Permission levels, path/command policy, confirmation | `policy.py`, `paths.py` |
| `tools/` | 24 registered tools and the registry | `base.py`, `registry.py` |
| `retrieval/` | Chunking, embeddings, vector store, citations | `service.py`, `store.py` |
| `memory/` | Long-term memory and keyword recall | `store.py` |
| `llm/` | Ollama client, prompts, test double | `ollama.py`, `prompts.py` |
| `database/` | Models, repositories, sessions | `models.py`, `repos.py` |
| `config/` | Pydantic schema, TOML loader, env overrides | `schema.py`, `loader.py` |
| `logging/` | Structured logging and the audit sink | `setup.py`, `audit.py` |
| `cli/`, `api/` | Interfaces over `Services` | `chat.py`, `app.py` |
