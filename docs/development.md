# Development

Arthur targets Python 3.12+ and uses [`uv`](https://docs.astral.sh/uv) for
environments and dependency resolution. The whole project is one installable
package under `src/`.

---

## Setup

```bash
git clone git@github.com:Kite-Bite/Arthur.git arthur && cd arthur
uv sync                    # core deps + dev group (pytest, ruff, mypy)
uv run arthur doctor       # verifies Ollama, model, tools, database
```

Optional extras:

```bash
uv sync --extra chroma     # ChromaDB vector backend
uv sync --extra pdf        # PDF extraction for RAG ingestion
uv sync --all-extras
```

You need a running Ollama for the *real* experience, but **not for the test
suite** — tests substitute a scripted LLM and hash embeddings, so `uv run
pytest` works offline in a couple of seconds.

```bash
ollama serve &
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### Editor

```bash
uv run ruff format .       # format on save via your LSP
uv run ruff check . --fix  # auto-fix imports and unsafe-upgrade hints
```

---

## Quality gates

All four must pass before a commit:

```bash
uv run pytest              # 241 tests
uv run ruff check .        # lint (E, F, I, UP, B)
uv run ruff format --check .
uv run mypy src            # strict: disallow_untyped_defs
```

One-liner:

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

| Tool | Scope | Config |
| --- | --- | --- |
| `pytest` | `tests/` (unit, integration, security) | `[tool.pytest.ini_options]` |
| `ruff` | whole repo, line length 100 | `[tool.ruff]` |
| `mypy` | `src/` only, tests relaxed | `[tool.mypy]` |

Notes:

* `src/arthur/cli/*` carries a `B008` per-file ignore — Typer declares options
  as function-call defaults *by design*.
* `mypy` runs with `warn_unused_ignores`, so a stale `# type: ignore` is an
  error. Two `prop-decorator` ignores remain for `@computed_field @property`,
  which pydantic's own docs say to annotate (mypy issue #1362).
* Type annotations use PEP 695 syntax (`class Tool[ToolArgs: BaseModel]`).

---

## Layout

```
src/arthur/
├── agent/        orchestrator, decision parser, agent result types
├── api/          FastAPI app + response schemas
├── cli/          Typer commands, chat REPL
├── config/       Pydantic schema, TOML loader, env overrides
├── database/     SQLAlchemy models, repositories, session helpers
├── execution/    executor (validate → policy → confirm → run → audit)
├── llm/          Ollama client, prompts, scripted test double
├── logging/      structured logging + audit sink
├── memory/       long-term memory store and recall
├── retrieval/    chunking, embeddings, vector store, citations
├── security/     permission levels, path/command policy, confirmation
├── tools/        the registered tools + registry
└── services.py   composition root shared by CLI and API
```

Dependency direction is enforced by review, not tooling: `tools/` never
imports `agent/`, `cli/` and `api/` never contain business logic, and
`services.py` is the only module that constructs objects.

---

## Testing conventions

### The three suites

| Directory | What it proves | Typical fakes |
| --- | --- | --- |
| `tests/unit` | Pure logic in isolation | none — real parser, real policy, real chunker |
| `tests/integration` | The full `Services` graph behaves | only the LLM (`ScriptedLLM`); real SQLite, real tools, real git |
| `tests/security` | Hostile input cannot escape | a deliberately dangerous `sentinel` tool |

Integration tests construct a **real** `Services` object: real security policy,
real audit trail, in-memory SQLite, hash embeddings. Only the model is swapped.
That is why the security suite can assert "the tool never executed", not "the
tool returned an error".

### Fixtures (`tests/conftest.py`)

| Fixture | Provides |
| --- | --- |
| `base_config` | Sandboxed `Config`: `:memory:` DB, hash embedder, `allowed_roots=[tmp_path]`, file logging off |
| `services` | Built `Services` graph, closed after the test |
| `scripted([...])` | Swaps in a `ScriptedLLM`; returns the double so you can inspect `llm.calls` |
| `decision(tool, args)` | Factory for a well-formed decision JSON string |

`ScriptedLLM` is strict on purpose: it **raises** if the agent makes more calls
than you scripted, so a test that silently stops exercising the loop fails
instead of passing vacuously. It records `calls` and `temperatures`.

Typical agent test:

```python
def test_multi_step_run(services, scripted, decision, tmp_path):
    (tmp_path / "alpha.txt").write_text("alpha")
    scripted(
        [
            decision("list_files", {"directory": str(tmp_path)}),
            decision(None),
            "There is one file: alpha.txt",
        ]
    )

    result = services.agent.run("What files are there?")

    assert result.stopped_reason == "answered"
    assert result.used_tools == ["list_files"]
    assert result.answer == "There is one file: alpha.txt"
```

A tool run needs three responses (decide → tool, decide → hand off, answer).
Confirmation tests stop after the first, because the loop halts cleanly.

### What to test for a new tool

1. **Unit**: argument validation, the happy path, and at least one refusal
   (missing file, bad path).
2. **Policy**: if it touches paths, add a denial case to
   `tests/security/test_denials.py` — escape, symlink, denied directory.
3. **Permission**: if it writes, assert the level *and* that a confirmation
   is requested (`tests/security/test_confirmation.py`).

Do not mock the policy to make a test pass; the policy is the thing being
tested.

### Logging hygiene

An autouse fixture removes handlers that CLI/API code installs, so a test that
starts the server can't leak handlers into the next capture. If you add a
module that configures logging at import time, make it lazy.

---

## Working on the agent loop

The prompt, the parser and the orchestrator are the parts most affected by
model behaviour. Two habits keep this honest:

* **Never evaluate a prompt change on a single run.** Small models are
  stochastic; run the same request several times (and at
  `llm.decide_temperature = 0`) before believing a change helped.
* **Measure the prompt.** Prompt evaluation on a CPU runs at tens of tokens
  per second, so size is latency:

```bash
uv run python -c "
from arthur.services import Services
from arthur.llm.prompts import build_system_prompt
s = Services.create()
p = build_system_prompt(s.config, s.registry.schemas())
print(len(p), 'chars ~', len(p) // 4, 'tokens')
"
```

Structural problems get fixed in code, not prose: `decision_issue()` catches
"valid JSON that says nothing", `sanitize_citations()` catches invented paths,
`auto_retrieve` grounds answers that would otherwise come from the model's
memory. See [`architecture.md`](architecture.md#the-agent-loop).

---

## End-to-end checklist

Run this before a release, against a real Ollama:

```bash
uv run arthur doctor
uv run arthur ask "Show me my system information."        # tool is called
uv run arthur documents index ~/notes                     # real embeddings
uv run arthur ask "When does X happen? Cite the document."# grounded + cited
uv run arthur ask "Delete ~/notes/scratch.txt"            # prompts, then refuses
uv run arthur ask --yes "Delete ~/notes/scratch.txt"      # audited confirmed=True
uv run arthur logs --limit 5                              # trail matches
uv run arthur serve & curl -s localhost:8420/health
```

Things to watch for:

* a step trace in the output (a missing trace means the model skipped the tool);
* a real hostname/kernel in the answer (a generic answer means hallucination);
* a `[source: ...]` marker pointing at a file that was actually indexed.

---

## Committing

Small, narrated commits. The history is meant to be read as a design log —
each message explains *why*, not just *what*:

```
Pass lint, format and strict mypy across src
Add integration and security test suites
Make the agent actually usable with a small local model
```

Before pushing, run the four gates above. CI runs exactly the same commands
(see `.github/workflows/ci.yml`), so a green local run means a green build.
It then builds the wheel and the Docker image, and has a dedicated
*Optional extras* job that re-runs the gates and the full suite under
`uv sync --locked --all-extras`. That last job matters because `chromadb`
and `pypdf` are absent from a default install: without it, the Chroma
backend and PDF ingestion would never be executed — and mypy would only
ever see `chromadb` as an untyped `Any`. Run `uv sync --all-extras` locally
to execute `tests/extras/`; on a default install those tests skip with a
message naming the extra they need.
