"""Typed configuration schema.

Every configurable aspect of Arthur is represented here as a Pydantic model so
that configuration is validated once at load time and every module can rely on
well-typed settings. Nothing in this module reads the environment directly;
see :mod:`arthur.config.loader` for the loading precedence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# --- Defaults shared with the security module ---------------------------------

DEFAULT_DENIED_PATH_PATTERNS: list[str] = [
    "**/.ssh/**",
    "**/.gnupg/**",
    "**/.aws/**",
    "**/.docker/config.json",
    "**/.git-credentials",
    "**/.env",
    "**/.env.*",
    "**/*_rsa",
    "**/*_ed25519",
    "**/*_dsa",
    "**/id_ed25519",
    "**/shadow",
    "**/gshadow",
]

DEFAULT_DENIED_DIR_NAMES: list[str] = [".ssh", ".gnupg", ".aws", ".password-store"]

# Allow-listed commands for the controlled `run_command` tool. A value of
# `None` means "any positional arguments"; a list restricts positional verbs
# (used to keep e.g. ``systemctl`` limited to read-only subcommands).
DEFAULT_ALLOWED_COMMANDS: dict[str, list[str] | None] = {
    "date": None,
    "df": None,
    "du": None,
    "free": None,
    "hostname": None,
    "id": None,
    "ip": ["addr", "address", "link", "route", "neigh", "rule", "help"],
    "lscpu": None,
    "lsblk": None,
    "nproc": None,
    "ps": None,
    "ss": None,
    "systemctl": [
        "status",
        "is-active",
        "is-enabled",
        "list-units",
        "list-timers",
        "show",
        "cat",
        "help",
    ],
    "uname": None,
    "uptime": None,
    "whoami": None,
}

#: Verbs that mutate state; never allowed as positional arguments to
#: allow-listed commands (defence in depth beyond the sub-command allow-list).
DENIED_COMMAND_TOKENS: frozenset[str] = frozenset(
    {
        "set",
        "add",
        "del",
        "delete",
        "replace",
        "change",
        "remove",
        "rm",
        "restart",
        "stop",
        "start",
        "disable",
        "enable",
        "kill",
        "killall",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "format",
        "mkfs",
        "wipe",
        "truncate",
        "chmod",
        "chown",
        "chgrp",
        "mount",
        "umount",
        "sync",
    }
)

# Commands that are read-only and run without confirmation by default.
DEFAULT_SAFE_COMMANDS: list[str] = list(DEFAULT_ALLOWED_COMMANDS)


# --- Config sections -----------------------------------------------------------


class LLMConfig(BaseModel):
    """Local LLM connection and sampling settings."""

    provider: Literal["ollama"] = "ollama"
    host: str = "http://localhost:11434"
    model: str = "llama3.2:1b"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.2
    #: Sampling temperature for the decision step. Decisions are structured
    #: selections, so they run greedy by default - at non-zero temperature a
    #: small model sometimes answers instead of calling an obvious tool.
    decide_temperature: float = 0.0
    timeout_seconds: float = 180.0
    keep_alive: str = "5m"
    #: Prompt context window handed to Ollama. Must comfortably exceed the
    #: system prompt (tool schemas + memories) plus the answer budget, or
    #: generation is truncated with ``done_reason: "length"``.
    num_ctx: int = 8192
    # Token budgets: the decision step must stay small; answers get more room.
    decide_max_tokens: int = 320
    answer_max_tokens: int = 900
    history_messages: int = 20


class AgentConfig(BaseModel):
    """Agent loop behaviour."""

    max_steps: int = 6
    max_repairs: int = 2
    memory_injection: bool = True
    memory_injection_limit: int = 3
    # Retrieve relevant indexed passages before deciding. Small local models
    # routinely skip the search tool and answer from memory, so grounding is
    # done up-front; the search tool stays available for follow-up queries.
    auto_retrieve: bool = True
    auto_retrieve_limit: int = 4


class SecurityConfig(BaseModel):
    """Security policy. Safe defaults: no shell, confirmation for risky tools."""

    require_confirmation: bool = True
    # Gates the controlled `run_command` tool entirely. Arbitrary shell stays
    # denied even when this is true; only allow-listed commands become usable.
    default_shell_access: bool = False
    # RESTRICTED-level tools stay denied unless this is explicitly enabled.
    allow_restricted: bool = False
    # Empty means "the user's home directory" (resolved by PathPolicy).
    allowed_roots: list[str] = Field(default_factory=list)
    denied_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_DENIED_PATH_PATTERNS))
    denied_dir_names: list[str] = Field(default_factory=lambda: list(DEFAULT_DENIED_DIR_NAMES))
    allowed_commands: dict[str, list[str] | None] = Field(
        default_factory=lambda: dict(DEFAULT_ALLOWED_COMMANDS)
    )
    safe_commands: list[str] = Field(default_factory=lambda: list(DEFAULT_SAFE_COMMANDS))
    # Per-tool permission changes, e.g. {"delete_file": "RESTRICTED"}.
    tool_overrides: dict[str, str] = Field(default_factory=dict)

    @field_validator("tool_overrides")
    @classmethod
    def _validate_tool_overrides(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject unknown levels at load time.

        A typo here would otherwise be discovered only when that tool is first
        called, in the middle of a run.

        The import is local: ``arthur.security`` imports this module, so a
        module-level import would form a cycle.
        """
        from arthur.security.permissions import PermissionLevel

        for tool, level in value.items():
            try:
                PermissionLevel.parse(level)
            except ValueError as exc:
                raise ValueError(f"security.tool_overrides[{tool!r}]: {exc}") from exc
        return value


class MemoryConfig(BaseModel):
    """Persistent memory storage."""

    database: str = "~/.local/share/arthur/arthur.db"
    auto_recall: bool = True
    recall_limit: int = 3
    default_importance: float = 0.5


class RetrievalConfig(BaseModel):
    """RAG ingestion, chunking, embeddings and vector storage."""

    backend: Literal["sqlite", "chroma"] = "sqlite"
    embedder: Literal["ollama", "hash"] = "ollama"
    chroma_path: str = "~/.local/share/arthur/chroma"
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_passages: int = 5
    #: Cosine floor for passages injected up-front by the agent. Below it a
    #: passage is likely irrelevant; injecting it wastes context and invites a
    #: citation that has nothing to do with the question. Measured with
    #: nomic-embed-text: ~0.7 for on-topic, ~0.3-0.5 for unrelated queries.
    auto_retrieve_min_score: float = 0.6
    max_file_bytes: int = 2_000_000
    formats: list[str] = Field(
        default_factory=lambda: [
            "md",
            "markdown",
            "txt",
            "rst",
            "log",
            "py",
            "toml",
            "json",
            "yaml",
            "yml",
            "sh",
            "cfg",
            "ini",
            "csv",
        ]
    )


class LoggingConfig(BaseModel):
    """Logging behaviour. File logging defaults into the Arthur data dir."""

    level: str = "INFO"
    format: Literal["text", "json"] = "text"
    file: str | None = "~/.local/share/arthur/arthur.log"


class ApiConfig(BaseModel):
    """FastAPI server settings. Loopback-only unless explicitly changed."""

    host: str = "127.0.0.1"
    port: int = 8420
    token: str | None = None


class Config(BaseModel):
    """Root configuration object."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
