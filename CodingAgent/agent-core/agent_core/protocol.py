"""WebSocket protocol definitions and message envelopes using Pydantic v2."""

from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


# ── Payload Models ──────────────────────────────────────────────────────────


class PingPayload(BaseModel):
    """Payload for client ping."""

    model_config = ConfigDict(extra="ignore")


class PongPayload(BaseModel):
    """Payload for server pong."""

    model_config = ConfigDict(extra="ignore")


class ChatSendPayload(BaseModel):
    """Payload to initiate or continue a chat turn."""

    session_id: str = Field(..., description="Target session identifier")
    text: Optional[str] = Field(default=None, description="User message text")
    content: Optional[str] = Field(default=None, description="Alternative field for user message text")
    model_alias: str = Field(default="local-fast", description="Role-agnostic model alias from models.yaml")
    workspace_root: Optional[str] = Field(
        default=None, description="Explicit absolute workspace root directory path"
    )

    @model_validator(mode="after")
    def validate_content_text(self) -> ChatSendPayload:
        if not self.text and not self.content:
            raise ValueError("Either 'text' or 'content' must be provided in chat.send payload")
        if not self.text and self.content:
            self.text = self.content
        return self


class ChatCancelPayload(BaseModel):
    """Payload to cancel an in-flight chat turn."""

    session_id: str = Field(..., description="Session identifier to cancel")


class ChatDeltaPayload(BaseModel):
    """Payload containing a streaming chunk of chat output."""

    session_id: str
    text: str


class TokenUsage(BaseModel):
    """Token consumption statistics."""

    prompt: int = 0
    completion: int = 0
    total: int = 0


class ChatDonePayload(BaseModel):
    """Payload emitted when a chat completion finishes."""

    session_id: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class ChatErrorPayload(BaseModel):
    """Structured error payload for failed requests or protocol violations."""

    session_id: Optional[str] = None
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable explanation")


# ── Client Envelope Models ──────────────────────────────────────────────────


class ClientPingMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["ping"] = "ping"
    payload: PingPayload = Field(default_factory=PingPayload)


class ClientChatSendMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["chat.send"] = "chat.send"
    payload: ChatSendPayload


class ClientChatCancelMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["chat.cancel"] = "chat.cancel"
    payload: ChatCancelPayload


class ModelsListRequestPayload(BaseModel):
    """Payload to request the models list."""
    pass


class ClientModelsListMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["models.list"] = "models.list"
    payload: ModelsListRequestPayload = Field(default_factory=ModelsListRequestPayload)


class EditUndoPayload(BaseModel):
    """Payload to request an undo of a previous file edit by snapshot sequence number."""

    session_id: str
    seq: int
    call_id: Optional[str] = None
    workspace_root: Optional[str] = None


class ClientEditUndoMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["edit.undo"] = "edit.undo"
    payload: EditUndoPayload


class ChangesListPayload(BaseModel):
    workspace_root: Optional[str] = None
    session_id: Optional[str] = None


class ClientChangesListMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["changes.list"] = "changes.list"
    payload: ChangesListPayload = Field(default_factory=ChangesListPayload)


class ChangesUndoPayload(BaseModel):
    session_id: str
    seq: int
    workspace_root: Optional[str] = None
    force: bool = False
    call_id: Optional[str] = None


class ClientChangesUndoMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["changes.undo"] = "changes.undo"
    payload: ChangesUndoPayload


ClientMessage = Annotated[
    Union[
        ClientPingMessage,
        ClientChatSendMessage,
        ClientChatCancelMessage,
        ClientModelsListMessage,
        ClientEditUndoMessage,
        ClientChangesListMessage,
        ClientChangesUndoMessage,
    ],
    Field(discriminator="type"),
]



# ── Server Envelope Models ──────────────────────────────────────────────────


class ServerPongMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["pong"] = "pong"
    payload: PongPayload = Field(default_factory=PongPayload)


class ServerChatDeltaMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["chat.delta"] = "chat.delta"
    payload: ChatDeltaPayload


class ServerChatDoneMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["chat.done"] = "chat.done"
    payload: ChatDonePayload


class ServerChatErrorMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["chat.error"] = "chat.error"
    payload: ChatErrorPayload

class ToolCallPayload(BaseModel):
    session_id: str
    call_id: str
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    is_text_mode: bool = False


class ToolResultPayload(BaseModel):
    session_id: str
    call_id: str
    ok: bool
    summary: str
    preview: Optional[str] = None
    truncated: bool = False
    possible_injection: bool = False
    warning: Optional[bool] = False
    snapshot_seq: Optional[int] = None


class ModelMetadataItem(BaseModel):
    """Metadata for a configured or installed model."""

    alias: str
    configured_model: str
    installed: bool
    resolved_model: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None
    family: Optional[str] = None
    tool_mode: Optional[str] = None
    context_length: Optional[int] = None


class ModelsListPayload(BaseModel):
    """Payload containing Ollama model availability and configured models list."""

    ollama_available: bool
    models: List[ModelMetadataItem] = Field(default_factory=list)


class ServerToolCallMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["tool.call"] = "tool.call"
    session_id: str
    call_id: str
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    is_text_mode: bool = False
    payload: ToolCallPayload


class ServerToolResultMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["tool.result"] = "tool.result"
    session_id: str
    call_id: str
    ok: bool
    summary: str
    preview: Optional[str] = None
    truncated: bool = False
    possible_injection: bool = False
    warning: Optional[bool] = False
    snapshot_seq: Optional[int] = None
    payload: ToolResultPayload


class ServerModelsListMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["models.list"] = "models.list"
    payload: ModelsListPayload


class EditUndoResultPayload(BaseModel):
    """Payload emitted when an undo operation completes."""

    session_id: str
    seq: int
    call_id: Optional[str] = None
    ok: bool
    path: str
    action: str  # 'restored', 'deleted', 'refused', 'failed'
    summary: str
    error: Optional[str] = None
    restored_sha256: Optional[str] = None


class ServerEditUndoResultMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["edit.undo.result"] = "edit.undo.result"
    payload: EditUndoResultPayload


class ChangeItem(BaseModel):
    seq: int
    session_id: str
    path: str
    workspace_root: Optional[str] = None
    tool: Optional[str] = None
    existed: bool = True
    before_sha256: Optional[str] = None
    after_sha256: Optional[str] = None
    timestamp: str
    snapshot_file: Optional[str] = None
    is_legacy: bool = False
    can_undo: bool = True
    status: str = "clean"  # 'clean', 'conflict', 'corrupted', 'undone', 'legacy'
    current_sha256: Optional[str] = None


class ChangesListResultPayload(BaseModel):
    changes: List[ChangeItem] = Field(default_factory=list)


class ServerChangesListMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["changes.list.result"] = "changes.list.result"
    payload: ChangesListResultPayload


class ChangesUndoResultPayload(BaseModel):
    session_id: str
    seq: int
    call_id: Optional[str] = None
    ok: bool
    path: str
    action: str  # 'restored', 'deleted', 'conflict', 'corrupted', 'denied', 'failed'
    summary: str
    error: Optional[str] = None
    restored_sha256: Optional[str] = None


class ServerChangesUndoResultMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["changes.undo.result"] = "changes.undo.result"
    payload: ChangesUndoResultPayload


ServerMessage = Annotated[
    Union[
        ServerPongMessage,
        ServerChatDeltaMessage,
        ServerChatDoneMessage,
        ServerChatErrorMessage,
        ServerToolCallMessage,
        ServerToolResultMessage,
        ServerModelsListMessage,
        ServerEditUndoResultMessage,
        ServerChangesListMessage,
        ServerChangesUndoResultMessage,
    ],
    Field(discriminator="type"),
]



# ── Envelope Discriminator Root for Inbound Parsing ─────────────────────────


class InboundEnvelope(BaseModel):
    """Raw envelope wrapper used to inspect the message type and dispatch."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


# ── Helpers ─────────────────────────────────────────────────────────────────


def parse_client_message(raw_text: str) -> ClientMessage:
    """Parse and validate an incoming client JSON string into a typed ClientMessage.

    Raises:
        json.JSONDecodeError: If raw_text is not valid JSON.
        ValueError: If message type is unknown.
        ValidationError: If the payload does not match the schema for the specified type.
    """
    raw_dict = json.loads(raw_text)
    if not isinstance(raw_dict, dict):
        raise ValueError("Root message must be a JSON object")

    msg_type = raw_dict.get("type")
    if not msg_type:
        raise ValueError("Missing 'type' field in message envelope")

    if msg_type == "ping":
        return ClientPingMessage.model_validate(raw_dict)
    elif msg_type == "chat.send":
        return ClientChatSendMessage.model_validate(raw_dict)
    elif msg_type == "chat.cancel":
        return ClientChatCancelMessage.model_validate(raw_dict)
    elif msg_type == "models.list":
        return ClientModelsListMessage.model_validate(raw_dict)
    elif msg_type == "edit.undo":
        return ClientEditUndoMessage.model_validate(raw_dict)
    elif msg_type == "changes.list":
        return ClientChangesListMessage.model_validate(raw_dict)
    elif msg_type == "changes.undo":
        return ClientChangesUndoMessage.model_validate(raw_dict)
    else:
        raise ValueError(f"Unsupported message type: '{msg_type}'")


def create_pong(reply_to_id: str | None = None) -> ServerPongMessage:
    """Create a typed pong response."""
    return ServerPongMessage(id=reply_to_id or str(uuid4()))


def create_delta(
    session_id: str,
    text: str,
    reply_to_id: str | None = None,
) -> ServerChatDeltaMessage:
    """Create a typed chat.delta response."""
    return ServerChatDeltaMessage(
        id=reply_to_id or str(uuid4()),
        payload=ChatDeltaPayload(session_id=session_id, text=text),
    )


def create_done(
    session_id: str,
    usage: TokenUsage | None = None,
    reply_to_id: str | None = None,
) -> ServerChatDoneMessage:
    """Create a typed chat.done response."""
    return ServerChatDoneMessage(
        id=reply_to_id or str(uuid4()),
        payload=ChatDonePayload(
            session_id=session_id,
            usage=usage or TokenUsage(),
        ),
    )


def create_error(
    code: str,
    message: str,
    reply_to_id: str | None = None,
    session_id: str | None = None,
) -> ServerChatErrorMessage:
    """Create a typed chat.error response."""
    return ServerChatErrorMessage(
        id=reply_to_id or str(uuid4()),
        payload=ChatErrorPayload(session_id=session_id, code=code, message=message),
    )


def create_tool_call(
    session_id: str,
    call_id: str,
    tool: str,
    args: Dict[str, Any],
    is_text_mode: bool = False,
    reply_to_id: str | None = None,
) -> ServerToolCallMessage:
    """Create a typed tool.call event message."""
    payload = ToolCallPayload(
        session_id=session_id,
        call_id=call_id,
        tool=tool,
        args=args,
        is_text_mode=is_text_mode,
    )
    return ServerToolCallMessage(
        id=reply_to_id or str(uuid4()),
        session_id=session_id,
        call_id=call_id,
        tool=tool,
        args=args,
        is_text_mode=is_text_mode,
        payload=payload,
    )


def create_tool_result(
    session_id: str,
    call_id: str,
    ok: bool,
    summary: str,
    preview: Optional[str] = None,
    truncated: bool = False,
    possible_injection: bool = False,
    warning: Optional[bool] = False,
    snapshot_seq: Optional[int] = None,
    reply_to_id: str | None = None,
) -> ServerToolResultMessage:
    """Create a typed tool.result event message with strictly bounded preview (max 4 KB)."""
    bounded_preview = preview[:4200] if preview else None
    payload = ToolResultPayload(
        session_id=session_id,
        call_id=call_id,
        ok=ok,
        summary=summary,
        preview=bounded_preview,
        truncated=truncated,
        possible_injection=possible_injection,
        warning=bool(warning),
        snapshot_seq=snapshot_seq,
    )
    return ServerToolResultMessage(
        id=reply_to_id or str(uuid4()),
        session_id=session_id,
        call_id=call_id,
        ok=ok,
        summary=summary,
        preview=bounded_preview,
        truncated=truncated,
        possible_injection=possible_injection,
        warning=bool(warning),
        snapshot_seq=snapshot_seq,
        payload=payload,
    )


def create_models_list(
    payload: ModelsListPayload,
    reply_to_id: str | None = None,
) -> ServerModelsListMessage:
    """Create a typed models.list event message."""
    return ServerModelsListMessage(
        id=reply_to_id or str(uuid4()),
        payload=payload,
    )


def create_edit_undo(
    session_id: str,
    seq: int,
    workspace_root: Optional[str] = None,
    msg_id: Optional[str] = None,
) -> ClientEditUndoMessage:
    """Create a typed edit.undo request."""
    return ClientEditUndoMessage(
        id=msg_id or str(uuid4()),
        payload=EditUndoPayload(
            session_id=session_id,
            seq=seq,
            workspace_root=workspace_root,
        ),
    )


def create_edit_undo_result(
    session_id: str,
    seq: int,
    ok: bool,
    path: str,
    action: str,
    summary: str,
    call_id: Optional[str] = None,
    error: Optional[str] = None,
    restored_sha256: Optional[str] = None,
    reply_to_id: str | None = None,
) -> ServerEditUndoResultMessage:
    """Create a typed edit.undo.result event message."""
    payload = EditUndoResultPayload(
        session_id=session_id,
        seq=seq,
        call_id=call_id,
        ok=ok,
        path=path,
        action=action,
        summary=summary,
        error=error,
        restored_sha256=restored_sha256,
    )
    return ServerEditUndoResultMessage(
        id=reply_to_id or str(uuid4()),
        payload=payload,
    )



def create_chat_send(
    session_id: str,
    text: str,
    model_alias: str,
    workspace_root: Optional[str] = None,
    msg_id: Optional[str] = None,
) -> ClientChatSendMessage:
    """Create a typed chat.send request."""
    return ClientChatSendMessage(
        id=msg_id or str(uuid4()),
        payload=ChatSendPayload(
            session_id=session_id,
            text=text,
            model_alias=model_alias,
            workspace_root=workspace_root,
        ),
    )


def create_changes_list_result(
    changes: List[Dict[str, Any]] | List[ChangeItem],
    reply_to_id: Optional[str] = None,
) -> ServerChangesListMessage:
    """Create a typed changes.list.result message."""
    items: List[ChangeItem] = []
    for c in changes:
        if isinstance(c, ChangeItem):
            items.append(c)
        elif isinstance(c, dict):
            items.append(ChangeItem(**c))
    return ServerChangesListMessage(
        id=reply_to_id or str(uuid4()),
        payload=ChangesListResultPayload(changes=items),
    )


def create_changes_undo_result(
    session_id: str,
    seq: int,
    ok: bool,
    path: str,
    action: str,
    summary: str,
    call_id: Optional[str] = None,
    error: Optional[str] = None,
    restored_sha256: Optional[str] = None,
    reply_to_id: Optional[str] = None,
) -> ServerChangesUndoResultMessage:
    """Create a typed changes.undo.result event message."""
    payload = ChangesUndoResultPayload(
        session_id=session_id,
        seq=seq,
        call_id=call_id,
        ok=ok,
        path=path,
        action=action,
        summary=summary,
        error=error,
        restored_sha256=restored_sha256,
    )
    return ServerChangesUndoResultMessage(
        id=reply_to_id or str(uuid4()),
        payload=payload,
    )



