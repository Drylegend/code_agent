"""Unit tests for Ollama model metadata fetching, TTL caching, and protocol exposure (Phase 7.7 B1-B6)."""

from __future__ import annotations

import asyncio
import io
import json
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from agent_core.config import ModelConfig
from agent_core.ollama_metadata import OllamaMetadataService
from agent_core.protocol import (
    ClientModelsListMessage,
    ModelMetadataItem,
    ModelsListPayload,
    ServerModelsListMessage,
    create_models_list,
    parse_client_message,
)


def make_mock_response(status: int, data: Dict[str, Any]) -> MagicMock:
    """Helper to mock urllib.request.urlopen response."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(data).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    return resp


@pytest.fixture
def sample_configured_models() -> Dict[str, ModelConfig]:
    return {
        "local-fast": ModelConfig(
            alias="local-fast",
            model="ollama/qwen2.5:3b",
            api_base="http://127.0.0.1:11434",
            context_window=32768,
        ),
        "local-coder": ModelConfig(
            alias="local-coder",
            model="ollama/qwen2.5-coder:7b",
            api_base="http://127.0.0.1:11434",
            context_window=32768,
        ),
        "local-custom": ModelConfig(
            alias="local-custom",
            model="ollama/not-installed-model:latest",
            api_base="http://127.0.0.1:11434",
            context_window=8192,
        ),
    }


# ── B1 & B6.1, B6.2: Metadata Mapping from /api/tags and /api/show ───────────


@pytest.mark.anyio
async def test_b6_sample_tags_and_show_metadata_mapping(
    sample_configured_models: Dict[str, ModelConfig],
) -> None:
    """Verify correct extraction of parameter size, quantization, family, context length, and tool mode."""
    tags_data = {
        "models": [
            {
                "name": "qwen2.5:3b",
                "model": "qwen2.5:3b",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "qwen2",
                    "parameter_size": "3.1B",
                    "quantization_level": "Q4_K_M",
                },
            },
            {
                "name": "qwen2.5-coder:7b",
                "model": "qwen2.5-coder:7b",
                "details": {
                    "family": "qwen2",
                    "parameter_size": "7.6B",
                    "quantization_level": "Q4_K_M",
                },
            },
        ]
    }

    show_data_3b = {
        "model_info": {
            "general.architecture": "qwen2",
            "qwen2.context_length": 131072,
        },
        "capabilities": ["tools", "completion"],
        "template": "{{- if .Tools }}...{{- end }}",
    }

    show_data_7b = {
        "model_info": {
            "qwen2.context_length": 32768,
        },
        "capabilities": ["tools"],
    }

    def mocked_urlopen(req: Any, timeout: float = 2.0) -> Any:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/api/tags" in url:
            return make_mock_response(200, tags_data)
        elif "/api/show" in url:
            body = json.loads(req.data.decode("utf-8")) if hasattr(req, "data") else {}
            if "3b" in body.get("model", ""):
                return make_mock_response(200, show_data_3b)
            return make_mock_response(200, show_data_7b)
        raise ValueError(f"Unexpected URL: {url}")

    service = OllamaMetadataService(ttl_seconds=10.0)

    with patch("urllib.request.urlopen", side_effect=mocked_urlopen):
        payload = await service.get_models_metadata(sample_configured_models)

    assert payload.ollama_available is True
    assert len(payload.models) == 3

    # Check local-fast mapping
    m_fast = next(m for m in payload.models if m.alias == "local-fast")
    assert m_fast.installed is True
    assert m_fast.resolved_model == "qwen2.5:3b"
    assert m_fast.parameter_size == "3.1B"
    assert m_fast.quantization == "Q4_K_M"
    assert m_fast.family == "qwen2"
    assert m_fast.context_length == 131072
    assert m_fast.tool_mode == "native"

    # Check local-coder mapping
    m_coder = next(m for m in payload.models if m.alias == "local-coder")
    assert m_coder.installed is True
    assert m_coder.resolved_model == "qwen2.5-coder:7b"
    assert m_coder.parameter_size == "7.6B"
    assert m_coder.quantization == "Q4_K_M"
    assert m_coder.tool_mode == "native"


# ── B6.3: Ollama Unavailable -> Offline State ────────────────────────────────


@pytest.mark.anyio
async def test_b6_ollama_unavailable_offline_state(
    sample_configured_models: Dict[str, ModelConfig],
) -> None:
    """When Ollama is unreachable, returns ollama_available=False without crashing."""
    service = OllamaMetadataService(ttl_seconds=10.0)

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        payload = await service.get_models_metadata(sample_configured_models)

    assert payload.ollama_available is False
    assert len(payload.models) == 3
    for m in payload.models:
        assert m.installed is False
        assert m.resolved_model is None


# ── B6.4: Configured Model Not Installed -> Missing State ────────────────────


@pytest.mark.anyio
async def test_b6_configured_model_not_installed(
    sample_configured_models: Dict[str, ModelConfig],
) -> None:
    """Configured model missing from /api/tags has installed=False."""
    tags_data = {
        "models": [
            {
                "name": "qwen2.5:3b",
                "details": {"parameter_size": "3.1B", "quantization_level": "Q4_K_M"},
            }
        ]
    }

    def mocked_urlopen(req: Any, timeout: float = 2.0) -> Any:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/api/tags" in url:
            return make_mock_response(200, tags_data)
        elif "/api/show" in url:
            return make_mock_response(200, {"model_info": {}})
        raise ValueError(url)

    service = OllamaMetadataService(ttl_seconds=10.0)

    with patch("urllib.request.urlopen", side_effect=mocked_urlopen):
        payload = await service.get_models_metadata(sample_configured_models)

    assert payload.ollama_available is True
    # local-fast is installed
    fast = next(m for m in payload.models if m.alias == "local-fast")
    assert fast.installed is True

    # local-custom is NOT installed
    custom = next(m for m in payload.models if m.alias == "local-custom")
    assert custom.installed is False
    assert custom.resolved_model is None


# ── B6.5: TTL Cache Avoids Repeated API Calls ────────────────────────────────


@pytest.mark.anyio
async def test_b6_ttl_cache_avoids_repeated_calls(
    sample_configured_models: Dict[str, ModelConfig],
) -> None:
    """Calls within TTL reuse cached payload and do not make additional HTTP requests."""
    tags_data = {"models": [{"name": "qwen2.5:3b"}]}
    service = OllamaMetadataService(ttl_seconds=5.0)

    call_count = 0

    def counting_urlopen(req: Any, timeout: float = 2.0) -> Any:
        nonlocal call_count
        call_count += 1
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/api/tags" in url:
            return make_mock_response(200, tags_data)
        return make_mock_response(200, {})

    with patch("urllib.request.urlopen", side_effect=counting_urlopen):
        payload1 = await service.get_models_metadata(sample_configured_models)
        initial_count = call_count
        assert initial_count > 0

        # Second call within TTL
        payload2 = await service.get_models_metadata(sample_configured_models)
        assert call_count == initial_count  # No additional calls
        assert payload1 is payload2

        # Invalidate cache -> next call should fetch
        service.invalidate_cache()
        await service.get_models_metadata(sample_configured_models)
        assert call_count > initial_count


# ── B6.6: Protocol models.list Message Parsing ───────────────────────────────


def test_b6_protocol_models_list_message() -> None:
    """Client and Server models.list envelopes are validated correctly."""
    client_raw = json.dumps({"type": "models.list", "payload": {}})
    parsed = parse_client_message(client_raw)
    assert isinstance(parsed, ClientModelsListMessage)
    assert parsed.type == "models.list"

    payload = ModelsListPayload(
        ollama_available=True,
        models=[
            ModelMetadataItem(
                alias="local-fast",
                configured_model="ollama/qwen2.5:3b",
                installed=True,
                resolved_model="qwen2.5:3b",
                parameter_size="3.1B",
                quantization="Q4_K_M",
                tool_mode="native",
            )
        ],
    )
    server_msg = create_models_list(payload)
    assert isinstance(server_msg, ServerModelsListMessage)
    assert server_msg.type == "models.list"
    assert server_msg.payload.models[0].resolved_model == "qwen2.5:3b"


# ── B6.7: Alias Remains Stable While Resolved Metadata Changes ───────────────


@pytest.mark.anyio
async def test_b6_alias_remains_stable_across_resolutions(
    sample_configured_models: Dict[str, ModelConfig],
) -> None:
    """Configured alias 'local-fast' remains identical regardless of underlying model tags."""
    service = OllamaMetadataService(ttl_seconds=0.1)

    # Resolution 1: qwen2.5:3b installed
    tags1 = {"models": [{"name": "qwen2.5:3b", "details": {"parameter_size": "3B"}}]}
    with patch("urllib.request.urlopen", return_value=make_mock_response(200, tags1)):
        p1 = await service.get_models_metadata(sample_configured_models, force_refresh=True)

    fast1 = next(m for m in p1.models if m.alias == "local-fast")
    assert fast1.alias == "local-fast"
    assert fast1.installed is True

    # Resolution 2: Ollama goes offline
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        p2 = await service.get_models_metadata(sample_configured_models, force_refresh=True)

    fast2 = next(m for m in p2.models if m.alias == "local-fast")
    assert fast2.alias == "local-fast"  # Stable alias
    assert fast2.installed is False
