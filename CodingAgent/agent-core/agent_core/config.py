"""Configuration loading and validation for agent-core."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Configuration for a single model alias."""

    alias: str
    model: str
    api_base: str = "http://127.0.0.1:11434"
    context_window: int = 32768
    usable_budget: int = 8192
    tool_mode: Optional[str] = None

    @field_validator("alias", "model", "api_base")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v.strip()


class ModelsFileConfig(BaseModel):
    """Container for models.yaml structure."""

    models: Dict[str, ModelConfig]

    @model_validator(mode="before")
    @classmethod
    def populate_alias_from_key(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Models configuration root must be a mapping")

        models = data.get("models")
        if not isinstance(models, dict) or not models:
            raise ValueError("Configuration must contain a non-empty 'models' dictionary")

        for key, entry in models.items():
            if isinstance(entry, dict) and "alias" not in entry:
                entry["alias"] = key
        return data


DEFAULT_AGENT_STORE_DIR = Path(r"D:\agent-store")


def find_default_store_dir() -> Path:
    """Resolve the default external store directory outside workspace."""
    env_dir = os.environ.get("AGENT_CORE_STORE_DIR")
    if env_dir and env_dir.strip():
        return Path(env_dir.strip()).resolve()
    return DEFAULT_AGENT_STORE_DIR


def find_default_audit_log_path() -> Path:
    """Resolve the production audit log path outside workspace under D:\agent-store\audit\audit.jsonl."""
    env_path = os.environ.get("AGENT_CORE_AUDIT_LOG_PATH")
    if env_path and env_path.strip():
        p = Path(env_path.strip()).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    audit_dir = find_default_store_dir() / "audit"
    try:
        audit_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return audit_dir / "audit.jsonl"


class ServerConfig(BaseModel):
    """Runtime configuration for the WebSocket server."""

    host: str = "127.0.0.1"
    port: int
    auth_token: str
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    log_file: Path
    audit_log_path: Path = Field(default_factory=find_default_audit_log_path)

    @field_validator("host")
    @classmethod
    def validate_local_host(cls, v: str) -> str:
        # Strictly enforce local-only binding
        if v != "127.0.0.1":
            raise ValueError(f"Server host MUST be '127.0.0.1' for security; got '{v}'")
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1024 <= v <= 65535):
            raise ValueError(f"Port must be between 1024 and 65535; got {v}")
        return v

    @field_validator("auth_token")
    @classmethod
    def validate_auth_token(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("AGENT_CORE_AUTH_TOKEN must be non-empty")
        return v.strip()


def find_default_config_path() -> Path:
    """Resolve the default path to config/models.yaml."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "config" / "models.yaml"


def find_default_log_path() -> Path:
    """Resolve the default log file path under logs/agent-core.log."""
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "agent-core.log"


def load_models_config(path: Path | None = None) -> Dict[str, ModelConfig]:
    """Load and validate models.yaml using Pydantic v2."""
    config_path = path or find_default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Models configuration file not found at: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"Failed to read/parse YAML from {config_path}: {exc}") from exc

    try:
        parsed = ModelsFileConfig.model_validate(raw_data)
        return parsed.models
    except Exception as exc:
        raise ValueError(f"Invalid models configuration in {config_path}: {exc}") from exc


def load_server_config(
    port: int | None = None,
    host: str = "127.0.0.1",
    models_path: Path | None = None,
    log_path: Path | None = None,
    audit_log_path: Path | None = None,
) -> ServerConfig:
    """Build and validate ServerConfig from environment and arguments."""
    # Auth token must be provided via environment variable
    token = os.environ.get("AGENT_CORE_AUTH_TOKEN")
    if not token or not token.strip():
        raise ValueError(
            "AGENT_CORE_AUTH_TOKEN environment variable is required and must not be empty"
        )

    # Determine port: explicit argument > AGENT_CORE_PORT env var
    resolved_port: int
    if port is not None:
        resolved_port = port
    else:
        env_port = os.environ.get("AGENT_CORE_PORT")
        if not env_port or not env_port.strip():
            raise ValueError(
                "Port must be supplied via CLI argument (--port) or AGENT_CORE_PORT environment variable"
            )
        try:
            resolved_port = int(env_port.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid integer in AGENT_CORE_PORT: {env_port}") from exc

    models = load_models_config(models_path)
    log_file = log_path or find_default_log_path()
    resolved_audit = audit_log_path or find_default_audit_log_path()

    return ServerConfig(
        host=host,
        port=resolved_port,
        auth_token=token,
        models=models,
        log_file=log_file,
        audit_log_path=resolved_audit,
    )
