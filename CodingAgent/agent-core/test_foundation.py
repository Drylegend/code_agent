"""Foundation and Phase 2 test suite verifying all Step 4 requirements."""

import asyncio
import http
import json
import os
from pathlib import Path
import secrets
import sys
import urllib.request

import websockets
from websockets.asyncio.client import connect

from agent_core.config import load_server_config, load_models_config
from agent_core.llm import resolve_model_config, LLMServiceError
from agent_core.protocol import create_pong, parse_client_message
from agent_core.server import AgentCoreServer
from agent_core.sessions import SessionManager


async def run_tests():
    print("[1/16] Checking Python 3.12...")
    assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12, got {sys.version}"
    print(f"       OK! Python version: {sys.version.split()[0]}")

    print("[2/16] Checking virtual environment & dependencies...")
    import litellm
    import pydantic
    import yaml
    assert pydantic.__version__.startswith("2."), f"Expected Pydantic 2.x, got {pydantic.__version__}"
    import importlib.metadata
    litellm_ver = importlib.metadata.version("litellm")
    print(f"       OK! litellm={litellm_ver}, websockets={websockets.__version__}, pydantic={pydantic.__version__}")

    print("[3/16] Checking configuration & models.yaml loading...")
    models = load_models_config()
    assert "local-fast" in models, "Missing local-fast model alias"
    assert "local-coder" in models, "Missing local-coder model alias"
    print(f"       OK! Loaded models: {list(models.keys())}")

    # Set up test environment
    test_port = 18765
    test_token = secrets.token_hex(16)
    os.environ["AGENT_CORE_PORT"] = str(test_port)
    os.environ["AGENT_CORE_AUTH_TOKEN"] = test_token

    config = load_server_config()
    assert config.host == "127.0.0.1", "Host must be 127.0.0.1"
    assert config.port == test_port
    assert config.auth_token == test_token

    print("[4/16] Starting AgentCoreServer on 127.0.0.1...")
    server = AgentCoreServer(config)
    server_task = asyncio.create_task(server.run())

    # Wait briefly for server to bind
    for _ in range(50):
        if server.server and server.server.sockets:
            break
        await asyncio.sleep(0.05)

    assert server.server is not None, "Server failed to initialize"
    actual_sock = server.server.sockets[0].getsockname()
    print(f"       OK! Server bound to {actual_sock[0]}:{actual_sock[1]}")
    assert actual_sock[0] == "127.0.0.1", f"Expected bind to 127.0.0.1, got {actual_sock[0]}"

    ws_url = f"ws://127.0.0.1:{test_port}"

    print("[5/16] Testing unauthenticated connection rejection...")
    rejected = False
    try:
        async with connect(ws_url):
            pass
    except Exception as exc:
        rejected = True
        print(f"       OK! Unauthenticated connection rejected as expected: {type(exc).__name__}")
    assert rejected, "Security failure: Unauthenticated connection was NOT rejected!"

    print("[6/16] Testing wrong token rejection...")
    rejected_wrong = False
    try:
        async with connect(f"{ws_url}?token=bad_token_12345"):
            pass
    except Exception as exc:
        rejected_wrong = True
        print(f"       OK! Wrong token rejected: {type(exc).__name__}")
    assert rejected_wrong, "Security failure: Connection with invalid token was NOT rejected!"

    print("[7/16] Testing authenticated connection & ping -> pong...")
    # Test auth via query param
    async with connect(f"{ws_url}?token={test_token}") as ws:
        ping_msg = {"id": "test-ping-1", "type": "ping", "payload": {}}
        await ws.send(json.dumps(ping_msg))
        raw_resp = await asyncio.wait_for(ws.recv(), timeout=2.0)
        resp = json.loads(raw_resp)
        assert resp.get("type") == "pong", f"Expected pong, got {resp}"
        assert resp.get("id") == "test-ping-1", f"Expected matching id, got {resp}"
        print("       OK! Authenticated via query param and ping -> pong succeeded")

    # Test auth via Authorization header
    async with connect(ws_url, additional_headers={"Authorization": f"Bearer {test_token}"}) as ws:
        ping_msg = {"id": "test-ping-2", "type": "ping", "payload": {}}
        await ws.send(json.dumps(ping_msg))
        raw_resp = await asyncio.wait_for(ws.recv(), timeout=2.0)
        resp = json.loads(raw_resp)
        assert resp.get("type") == "pong"
        assert resp.get("id") == "test-ping-2"
        print("       OK! Authenticated via Authorization: Bearer header and ping -> pong succeeded")

        print("[8/16] Testing malformed messages & structured error handling...")
        # A. Malformed JSON
        await ws.send("{ bad json: ")
        err1 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err1.get("type") == "chat.error"
        assert err1["payload"]["code"] == "JSON_DECODE_ERROR"
        print("       OK! Handled JSON_DECODE_ERROR")

        # B. Missing required payload fields
        await ws.send(json.dumps({"id": "err-test-2", "type": "chat.send", "payload": {}}))
        err2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err2.get("type") == "chat.error"
        assert err2["payload"]["code"] == "VALIDATION_ERROR"
        print("       OK! Handled VALIDATION_ERROR")

        # C. Unknown model alias
        await ws.send(json.dumps({
            "id": "err-test-3",
            "type": "chat.send",
            "payload": {
                "session_id": "sess-1",
                "text": "Hello",
                "model_alias": "non-existent-model"
            }
        }))
        err3 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err3.get("type") == "chat.error"
        assert err3["payload"]["code"] == "MODEL_NOT_FOUND"
        print("       OK! Handled MODEL_NOT_FOUND via WebSocket")

    print("[9/16] Testing model alias resolution in LLM module...")
    fast_cfg = resolve_model_config("local-fast", models)
    assert fast_cfg.alias == "local-fast"
    coder_cfg = resolve_model_config("local-coder", models)
    assert coder_cfg.alias == "local-coder"
    try:
        resolve_model_config("bogus-alias", models)
        assert False, "Should have raised LLMServiceError for unknown alias"
    except LLMServiceError as exc:
        assert exc.code == "MODEL_NOT_FOUND"
        print("       OK! Model alias resolution and unknown alias rejection verified")

    print("[10/16] Testing session history isolation...")
    sm = SessionManager()
    s_a = sm.get_or_create("session-alpha")
    s_b = sm.get_or_create("session-beta")
    s_a.add_user_message("Message in Alpha")
    s_a.add_assistant_message("Response in Alpha")
    assert len(s_b.get_messages()) == 0, "Session Beta leaked history from Alpha!"
    s_b.add_user_message("Message in Beta")
    assert len(s_a.get_messages()) == 2
    assert len(s_b.get_messages()) == 1
    assert s_a.get_messages()[0]["content"] == "Message in Alpha"
    assert s_b.get_messages()[0]["content"] == "Message in Beta"
    print("       OK! In-memory session histories are strictly isolated")

    print("[11/16] Testing multiple message turns in one session...")
    s_multi = sm.get_or_create("session-multi")
    s_multi.add_user_message("Turn 1 Question")
    s_multi.add_assistant_message("Turn 1 Answer")
    s_multi.add_user_message("Turn 2 Question")
    s_multi.add_assistant_message("Turn 2 Answer")
    msgs = s_multi.get_messages()
    assert len(msgs) == 4
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[2]["content"] == "Turn 2 Question"
    assert msgs[3]["content"] == "Turn 2 Answer"
    print("       OK! Multi-turn session history accumulates in standard format")

    # Connect client for live/interactive testing
    async with connect(f"{ws_url}?token={test_token}") as ws:
        print("[12/16] Testing concurrent ping responsiveness during active generation...")
        # Send a long generation request
        await ws.send(json.dumps({
            "id": "gen-turn-1",
            "type": "chat.send",
            "payload": {
                "session_id": "sess-concurrent",
                "text": "Write a lengthy essay discussing the geology of Mars.",
                "model_alias": "local-fast"
            }
        }))

        # Concurrently send ping immediately without waiting for generation to finish
        await ws.send(json.dumps({
            "id": "ping-concurrent-1",
            "type": "ping",
            "payload": {}
        }))

        # Verify pong arrives promptly even while generation is underway
        got_pong = False
        deltas_before_pong = 0
        while not got_pong:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            msg = json.loads(raw)
            if msg["type"] == "pong" and msg["id"] == "ping-concurrent-1":
                got_pong = True
            elif msg["type"] == "chat.delta":
                deltas_before_pong += 1

        assert got_pong, "Failed to receive pong during generation"
        print(f"       OK! Server responded to ping while generation active ({deltas_before_pong} deltas received)")

        print("[13/16] Testing chat cancellation & history cleanup...")
        # Cancel the active generation on sess-concurrent
        await ws.send(json.dumps({
            "id": "cancel-turn-1",
            "type": "chat.cancel",
            "payload": {"session_id": "sess-concurrent"}
        }))

        # Wait briefly for cancellation to take effect
        await asyncio.sleep(0.5)

        # Verify session state: user message must be present, incomplete assistant message must NOT be in history
        sess_c = server.session_manager.get("sess-concurrent")
        assert sess_c is not None
        c_msgs = sess_c.get_messages()
        assert len(c_msgs) == 1, f"Expected 1 user message, found {len(c_msgs)}: {c_msgs}"
        assert c_msgs[0]["role"] == "user"
        print("       OK! Active generation cancelled cleanly; incomplete response was NOT added to history")

        print("[14/16] Testing successful streamed response & usage reporting...")
        # Send a fast prompt that completes
        await ws.send(json.dumps({
            "id": "stream-req-1",
            "type": "chat.send",
            "payload": {
                "session_id": "sess-stream-1",
                "text": "Reply with exactly: hello",
                "model_alias": "local-fast"
            }
        }))

        streamed_deltas = []
        done_payload = None
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            if msg["type"] == "chat.delta":
                assert msg["payload"]["session_id"] == "sess-stream-1"
                assert "text" in msg["payload"]
                streamed_deltas.append(msg["payload"]["text"])
            elif msg["type"] == "chat.done":
                assert msg["id"] == "stream-req-1", "Done message must correlate with request ID"
                assert msg["payload"]["session_id"] == "sess-stream-1"
                done_payload = msg["payload"]
                break

        assert len(streamed_deltas) > 0, "No chat.delta messages were received!"
        assert done_payload is not None, "chat.done message was not received!"
        assert "usage" in done_payload, "Usage object missing from chat.done"
        print(f"       OK! Streamed {len(streamed_deltas)} deltas; usage={done_payload['usage']}")

        # Verify history after successful completion
        sess_s = server.session_manager.get("sess-stream-1")
        assert sess_s is not None
        s_msgs = sess_s.get_messages()
        assert len(s_msgs) == 2, f"Expected user + assistant messages, got: {s_msgs}"
        assert s_msgs[0]["role"] == "user"
        assert s_msgs[1]["role"] == "assistant"
        assert s_msgs[1]["content"] == "".join(streamed_deltas)
        print("       OK! Assistant response properly committed to session history after stream completion")

    print("[15/16] Checking live Ollama installation & live streaming test...")
    ollama_online = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/", timeout=2.0) as resp:
            if resp.status == 200:
                ollama_online = True
    except Exception:
        ollama_online = False

    if not ollama_online:
        print("       SKIP: Ollama is not running at http://127.0.0.1:11434.")
    else:
        # Check that local-fast model actually functions live
        async with connect(f"{ws_url}?token={test_token}") as ws:
            await ws.send(json.dumps({
                "id": "live-test-req",
                "type": "chat.send",
                "payload": {
                    "session_id": "sess-live-verify",
                    "text": "Reply with exactly: hello",
                    "model_alias": "local-fast"
                }
            }))

            live_text_parts = []
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                msg = json.loads(raw)
                if msg["type"] == "chat.delta":
                    live_text_parts.append(msg["payload"]["text"])
                elif msg["type"] == "chat.done":
                    break

            live_reply = "".join(live_text_parts).strip().lower()
            print(f"       OK! Live response received from {fast_cfg.model}: '{live_reply}'")
            assert "hello" in live_reply, f"Expected 'hello' in response, got '{live_reply}'"

    print("[16/16] Testing graceful shutdown & security audit...")
    server.stop()
    await asyncio.wait_for(server_task, timeout=5.0)
    print("       OK! Server stopped cleanly.")

    # Verify log file
    log_file = config.log_file
    assert log_file.exists(), f"Log file does not exist at {log_file}"
    log_content = log_file.read_text(encoding="utf-8")
    assert "Rejected unauthenticated connection" in log_content, "Missing auth failure in log"
    assert "Client connected" in log_content, "Missing client connection in log"
    assert "Starting agent-core server" in log_content, "Missing startup in log"
    assert "Generation started" in log_content, "Missing generation start log"
    assert "Generation cancelled" in log_content, "Missing generation cancelled log"
    assert test_token not in log_content, "SECURITY VIOLATION: Auth token leaked into log file!"
    print(f"       OK! Log file verified at {log_file} (no token leaked).")

    print("\n==================================================")
    print("ALL 16 TEST PHASES PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(run_tests())
