# CodingAgent

A local-first personal Windows coding agent built with:
- **agent-shell**: Tauri 2 + React + TypeScript + Xterm.js frontend.
- **supervisor**: Rust sidecar supervisor managing Python lifecycle and token auth.
- **agent-core**: Python 3.12 async WebSocket engine with LiteLLM and Ollama models.
- **tools**: Safe, read-only workspace containment tools (list_directory, read_file).
