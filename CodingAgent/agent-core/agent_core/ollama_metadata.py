"""Ollama metadata fetcher and TTL cache for agent-core."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

from .config import ModelConfig
from .protocol import ModelMetadataItem, ModelsListPayload

logger = logging.getLogger(__name__)

DEFAULT_METADATA_TTL_SECONDS = 20.0
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 2.0


class OllamaMetadataService:
    """Fetches and caches Ollama model metadata with a short TTL."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_METADATA_TTL_SECONDS,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.timeout = timeout
        self._cached_payload: Optional[ModelsListPayload] = None
        self._last_fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    def invalidate_cache(self) -> None:
        """Clear cached metadata."""
        self._cached_payload = None
        self._last_fetched_at = 0.0

    async def get_models_metadata(
        self,
        configured_models: Dict[str, ModelConfig],
        force_refresh: bool = False,
    ) -> ModelsListPayload:
        """Retrieve model metadata, reusing cache if within TTL."""
        async with self._lock:
            now = time.monotonic()
            if (
                not force_refresh
                and self._cached_payload is not None
                and (now - self._last_fetched_at) < self.ttl_seconds
            ):
                return self._cached_payload

            payload = await asyncio.to_thread(self._fetch_metadata_sync, configured_models)
            self._cached_payload = payload
            self._last_fetched_at = now
            return payload

    def _fetch_metadata_sync(
        self,
        configured_models: Dict[str, ModelConfig],
    ) -> ModelsListPayload:
        """Synchronously query Ollama API endpoints for installed models and details."""
        api_base = "http://127.0.0.1:11434"
        for m in configured_models.values():
            if m.api_base:
                api_base = m.api_base.rstrip("/")
                break

        # 1. Fetch tags
        tags_url = f"{api_base}/api/tags"
        installed_models_data: List[Dict[str, Any]] = []
        ollama_available = False

        try:
            req = urllib.request.Request(tags_url, headers={"User-Agent": "agent-core/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    raw = json.loads(resp.read().decode("utf-8"))
                    installed_models_data = raw.get("models", [])
                    ollama_available = True
        except Exception as exc:
            logger.debug("Ollama /api/tags unreachable at %s: %s", tags_url, exc)
            ollama_available = False

        if not ollama_available:
            items: List[ModelMetadataItem] = []
            for alias, cfg in configured_models.items():
                items.append(
                    ModelMetadataItem(
                        alias=alias,
                        configured_model=cfg.model,
                        installed=False,
                        resolved_model=None,
                        parameter_size=None,
                        quantization=None,
                        family=None,
                        tool_mode=cfg.tool_mode,
                        context_length=cfg.context_window,
                    )
                )
            return ModelsListPayload(ollama_available=False, models=items)

        # Build lookup table of installed models
        installed_map: Dict[str, Dict[str, Any]] = {}
        for im in installed_models_data:
            name = im.get("name", "")
            if name:
                installed_map[name] = im
                if name.endswith(":latest"):
                    installed_map[name[:-7]] = im

        # 2. For each configured model, resolve against installed tags and query /api/show
        items = []
        for alias, cfg in configured_models.items():
            raw_model_name = cfg.model
            cleaned_name = raw_model_name.split("/", 1)[-1].strip() if "/" in raw_model_name else raw_model_name.strip()

            matched_info = (
                installed_map.get(cleaned_name)
                or installed_map.get(f"{cleaned_name}:latest")
            )
            if not matched_info:
                for k, v in installed_map.items():
                    if k == cleaned_name or k.startswith(f"{cleaned_name}:"):
                        matched_info = v
                        break

            if not matched_info:
                items.append(
                    ModelMetadataItem(
                        alias=alias,
                        configured_model=cfg.model,
                        installed=False,
                        resolved_model=None,
                        parameter_size=None,
                        quantization=None,
                        family=None,
                        tool_mode=cfg.tool_mode,
                        context_length=cfg.context_window,
                    )
                )
                continue

            resolved_tag = matched_info.get("name", cleaned_name)
            details = matched_info.get("details", {})
            param_size = details.get("parameter_size")
            quantization = details.get("quantization_level")
            family = details.get("family")

            # Query /api/show
            show_url = f"{api_base}/api/show"
            tool_mode = "text_json"
            context_len = cfg.context_window

            try:
                post_data = json.dumps({"model": resolved_tag}).encode("utf-8")
                show_req = urllib.request.Request(
                    show_url,
                    data=post_data,
                    headers={"Content-Type": "application/json", "User-Agent": "agent-core/1.0"},
                )
                with urllib.request.urlopen(show_req, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        show_data = json.loads(resp.read().decode("utf-8"))
                        model_info = show_data.get("model_info", {})
                        for k, v in model_info.items():
                            if k.endswith(".context_length") or k == "context_length":
                                try:
                                    context_len = int(v)
                                    break
                                except (ValueError, TypeError):
                                    pass

                        capabilities = show_data.get("capabilities", [])
                        template = show_data.get("template", "")
                        has_tool_cap = "tools" in capabilities
                        has_tool_tmpl = ".Tools" in template or "tool_call" in template
                        # Infer native only if Ollama model explicitly exposes tools capability or tool template
                        inferred_tool_mode = "native" if (has_tool_cap or has_tool_tmpl) else "text_json"
            except Exception as exc:
                logger.debug("Failed to query /api/show for %s: %s", resolved_tag, exc)
                inferred_tool_mode = "text_json"

            # Explicit configuration always wins; inference only serves as fallback/suggestion
            final_tool_mode = cfg.tool_mode or inferred_tool_mode

            items.append(
                ModelMetadataItem(
                    alias=alias,
                    configured_model=cfg.model,
                    installed=True,
                    resolved_model=resolved_tag,
                    parameter_size=param_size,
                    quantization=quantization,
                    family=family,
                    tool_mode=final_tool_mode,
                    context_length=context_len,
                )
            )

        return ModelsListPayload(ollama_available=True, models=items)


_default_service: Optional[OllamaMetadataService] = None


def get_ollama_metadata_service() -> OllamaMetadataService:
    global _default_service
    if _default_service is None:
        _default_service = OllamaMetadataService()
    return _default_service
