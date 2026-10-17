/**
 * Typed WebSocket client for agent-core sidecar communication.
 *
 * Responsibilities:
 * - Querying connection credentials from Tauri (`agent_core_get_connection`)
 * - Establishing authenticated WebSocket connection via URL query token
 * - Automatic reconnection with exponential backoff on disconnect / sidecar crash
 * - Exposing typed envelopes corresponding to Python agent_core.protocol
 * - Periodic ping / pong health check
 * - StrictMode-safe lifecycle management
 */

import { invoke } from "@tauri-apps/api/core";

// ── Protocol Types ────────────────────────────────────────────────────────────

export interface TokenUsage {
  prompt: number;
  completion: number;
  total: number;
  num_ctx?: number;
  usable_budget?: number;
  usage_percent?: number;
  truncated?: boolean;
  warning?: string | null;
}

export interface ContextStatusPayload {
  session_id: string;
  prompt_tokens: number;
  num_ctx: number;
  usable_budget: number;
  usage_percent: number;
  truncated?: boolean;
  warning?: string | null;
}

export interface ChatSendPayload {
  session_id: string;
  text: string;
  model_alias: string;
  workspace_root?: string | null;
}

export interface ChatCancelPayload {
  session_id: string;
}

export interface ChatDeltaPayload {
  session_id: string;
  text: string;
}

export interface ChatDonePayload {
  session_id: string;
  usage: TokenUsage;
}

export interface ChatErrorPayload {
  session_id?: string | null;
  code: string;
  message: string;
}

export interface ToolCallPayload {
  session_id: string;
  call_id: string;
  tool: string;
  args: Record<string, unknown>;
  is_text_mode?: boolean;
}

export interface ToolResultPayload {
  session_id: string;
  call_id: string;
  ok: boolean;
  summary: string;
  preview?: string | null;
  truncated: boolean;
  possible_injection: boolean;
  warning?: boolean;
  snapshot_seq?: number | null;
}

export interface EditUndoResultPayload {
  session_id: string;
  seq: number;
  call_id?: string | null;
  ok: boolean;
  path: string;
  action: string;
  summary: string;
  error?: string | null;
  restored_sha256?: string | null;
}


export interface ModelMetadata {
  alias: string;
  configured_model: string;
  installed: boolean;
  resolved_model?: string | null;
  parameter_size?: string | null;
  quantization?: string | null;
  family?: string | null;
  tool_mode?: string | null;
  context_length?: number | null;
}

export interface ModelsListPayload {
  ollama_available: boolean;
  models: ModelMetadata[];
}

// Client envelopes
export interface ClientPingMessage {
  id: string;
  type: "ping";
  payload: Record<string, never>;
}

export interface ClientChatSendMessage {
  id: string;
  type: "chat.send";
  payload: ChatSendPayload;
}

export interface ClientChatCancelMessage {
  id: string;
  type: "chat.cancel";
  payload: ChatCancelPayload;
}

export interface ClientModelsListMessage {
  id: string;
  type: "models.list";
  payload: Record<string, never>;
}

export interface ChangeItem {
  seq: number;
  session_id: string;
  path: string;
  workspace_root?: string | null;
  tool?: string | null;
  existed: boolean;
  before_sha256?: string | null;
  after_sha256?: string | null;
  timestamp: string;
  snapshot_file?: string | null;
  is_legacy: boolean;
  can_undo: boolean;
  status: "clean" | "conflict" | "corrupted" | "undone" | "legacy";
  current_sha256?: string | null;
}

export interface ChangesListResultPayload {
  changes: ChangeItem[];
}

export interface ChangesUndoResultPayload {
  session_id: string;
  seq: number;
  call_id?: string | null;
  ok: boolean;
  path: string;
  action: string;
  summary: string;
  error?: string | null;
  restored_sha256?: string | null;
}

export interface ClientChangesListMessage {
  id: string;
  type: "changes.list";
  payload: {
    workspace_root?: string | null;
    session_id?: string | null;
  };
}

export interface ClientChangesUndoMessage {
  id: string;
  type: "changes.undo";
  payload: {
    session_id: string;
    seq: number;
    workspace_root?: string | null;
    force?: boolean;
    call_id?: string | null;
  };
}

export interface ClientApprovalResponseMessage {
  id: string;
  type: "approval.response";
  payload: {
    request_id: string;
    decision: "approve" | "reject";
    session_id?: string;
  };
}

export interface ClientSessionSetAutonomyMessage {
  id: string;
  type: "session.set_autonomy";
  payload: {
    session_id: string;
    autonomy_level: number;
  };
}

export type ClientMessage =
  | ClientPingMessage
  | ClientChatSendMessage
  | ClientChatCancelMessage
  | ClientModelsListMessage
  | ClientChangesListMessage
  | ClientChangesUndoMessage
  | ClientApprovalResponseMessage
  | ClientSessionSetAutonomyMessage;

// Server envelopes
export interface ServerPongMessage {
  id: string;
  type: "pong";
  payload: Record<string, unknown>;
}

export interface ServerChatDeltaMessage {
  id: string;
  type: "chat.delta";
  payload: ChatDeltaPayload;
}

export interface ServerChatDoneMessage {
  id: string;
  type: "chat.done";
  payload: ChatDonePayload;
}

export interface ServerChatErrorMessage {
  id: string;
  type: "chat.error";
  payload: ChatErrorPayload;
}

export interface ServerToolCallMessage {
  id: string;
  type: "tool.call";
  session_id?: string;
  call_id?: string;
  tool?: string;
  args?: Record<string, unknown>;
  payload?: ToolCallPayload;
}

export interface ServerToolResultMessage {
  id: string;
  type: "tool.result";
  session_id?: string;
  call_id?: string;
  ok?: boolean;
  summary?: string;
  preview?: string | null;
  truncated?: boolean;
  possible_injection?: boolean;
  warning?: boolean;
  snapshot_seq?: number | null;
  payload?: ToolResultPayload;
}

export interface ServerModelsListMessage {
  id: string;
  type: "models.list";
  payload: ModelsListPayload;
}

export interface ServerChangesListMessage {
  id: string;
  type: "changes.list.result";
  payload: ChangesListResultPayload;
}

export interface ServerChangesUndoResultMessage {
  id: string;
  type: "changes.undo.result";
  payload: ChangesUndoResultPayload;
}

export interface ApprovalRequestPayload {
  request_id: string;
  session_id: string;
  tool: string;
  path: string;
  diff_preview?: string | null;
  bytes?: number | null;
  timeout: number;
}

export interface ServerApprovalRequestMessage {
  id?: string;
  type: "approval_request";
  request_id?: string;
  session_id?: string;
  tool?: string;
  path?: string;
  diff_preview?: string | null;
  bytes?: number | null;
  timeout?: number;
  payload?: ApprovalRequestPayload;
}

export interface SessionAutonomyPayload {
  session_id: string;
  autonomy_level: number;
}

export interface ServerSessionAutonomyMessage {
  id?: string;
  type: "session.autonomy";
  payload: SessionAutonomyPayload;
}

export interface ServerContextStatusMessage {
  id?: string;
  type: "context.status";
  payload: ContextStatusPayload;
}

export type ServerMessage =
  | ServerPongMessage
  | ServerChatDeltaMessage
  | ServerChatDoneMessage
  | ServerChatErrorMessage
  | ServerToolCallMessage
  | ServerToolResultMessage
  | ServerModelsListMessage
  | ServerChangesListMessage
  | ServerChangesUndoResultMessage
  | ServerApprovalRequestMessage
  | ServerSessionAutonomyMessage
  | ServerContextStatusMessage;


// ── Connection Types ──────────────────────────────────────────────────────────

export type ConnectionState =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected";

export interface ConnectionInfo {
  port: number;
  token: string;
  ready: boolean;
}

export type StateChangeCallback = (state: ConnectionState) => void;
export type DeltaCallback = (payload: ChatDeltaPayload) => void;
export type DoneCallback = (payload: ChatDonePayload) => void;
export type ErrorCallback = (payload: ChatErrorPayload) => void;
export type PongCallback = () => void;
export type ToolCallCallback = (payload: ToolCallPayload) => void;
export type ToolResultCallback = (payload: ToolResultPayload) => void;
export type ModelsListCallback = (payload: ModelsListPayload) => void;
export type UndoResultCallback = (payload: EditUndoResultPayload) => void;
export type ChangesListCallback = (changes: ChangeItem[]) => void;
export type ChangesUndoResultCallback = (payload: ChangesUndoResultPayload) => void;
export type ApprovalRequestCallback = (payload: ApprovalRequestPayload) => void;
export type SessionAutonomyCallback = (payload: SessionAutonomyPayload) => void;
export type ContextStatusCallback = (payload: ContextStatusPayload) => void;

// ── Configuration Constants ───────────────────────────────────────────────────

const POLL_INTERVAL_MS = 500;
const INITIAL_RECONNECT_MS = 1000;
const MAX_RECONNECT_MS = 10000;
const RECONNECT_MULTIPLIER = 1.5;
const PING_INTERVAL_MS = 15000;

// ── AgentSocket Class ─────────────────────────────────────────────────────────

export class AgentSocket {
  private ws: WebSocket | null = null;
  private state: ConnectionState = "disconnected";
  private isDisposed = false;

  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;

  private reconnectDelay = INITIAL_RECONNECT_MS;
  private hasConnectedOnce = false;

  // Listeners
  private stateListeners = new Set<StateChangeCallback>();
  private deltaListeners = new Set<DeltaCallback>();
  private doneListeners = new Set<DoneCallback>();
  private errorListeners = new Set<ErrorCallback>();
  private pongListeners = new Set<PongCallback>();
  private toolCallListeners = new Set<ToolCallCallback>();
  private toolResultListeners = new Set<ToolResultCallback>();
  private modelsListListeners = new Set<ModelsListCallback>();
  private undoResultListeners = new Set<UndoResultCallback>();
  private changesListListeners = new Set<ChangesListCallback>();
  private changesUndoResultListeners = new Set<ChangesUndoResultCallback>();
  private approvalRequestListeners = new Set<ApprovalRequestCallback>();
  private sessionAutonomyListeners = new Set<SessionAutonomyCallback>();
  private contextStatusListeners = new Set<ContextStatusCallback>();


  /** Current connection state. */
  public getState(): ConnectionState {
    return this.state;
  }

  // ── Listener Subscription ──────────────────────────────────────────────────

  public onStateChange(cb: StateChangeCallback): () => void {
    this.stateListeners.add(cb);
    cb(this.state);
    return () => this.stateListeners.delete(cb);
  }

  public onDelta(cb: DeltaCallback): () => void {
    this.deltaListeners.add(cb);
    return () => this.deltaListeners.delete(cb);
  }

  public onDone(cb: DoneCallback): () => void {
    this.doneListeners.add(cb);
    return () => this.doneListeners.delete(cb);
  }

  public onError(cb: ErrorCallback): () => void {
    this.errorListeners.add(cb);
    return () => this.errorListeners.delete(cb);
  }

  public onPong(cb: PongCallback): () => void {
    this.pongListeners.add(cb);
    return () => this.pongListeners.delete(cb);
  }

  public onToolCall(cb: ToolCallCallback): () => void {
    this.toolCallListeners.add(cb);
    return () => this.toolCallListeners.delete(cb);
  }

  public onToolResult(cb: ToolResultCallback): () => void {
    this.toolResultListeners.add(cb);
    return () => this.toolResultListeners.delete(cb);
  }

  public onModelsList(cb: ModelsListCallback): () => void {
    this.modelsListListeners.add(cb);
    return () => this.modelsListListeners.delete(cb);
  }

  public onUndoResult(cb: UndoResultCallback): () => void {
    this.undoResultListeners.add(cb);
    return () => this.undoResultListeners.delete(cb);
  }

  public onChangesList(cb: ChangesListCallback): () => void {
    this.changesListListeners.add(cb);
    return () => this.changesListListeners.delete(cb);
  }

  public onChangesUndoResult(cb: ChangesUndoResultCallback): () => void {
    this.changesUndoResultListeners.add(cb);
    return () => this.changesUndoResultListeners.delete(cb);
  }

  public onApprovalRequest(cb: ApprovalRequestCallback): () => void {
    this.approvalRequestListeners.add(cb);
    return () => this.approvalRequestListeners.delete(cb);
  }

  public onSessionAutonomy(cb: SessionAutonomyCallback): () => void {
    this.sessionAutonomyListeners.add(cb);
    return () => this.sessionAutonomyListeners.delete(cb);
  }

  public onContextStatus(cb: ContextStatusCallback): () => void {
    this.contextStatusListeners.add(cb);
    return () => this.contextStatusListeners.delete(cb);
  }


  private setState(newState: ConnectionState): void {
    if (this.state === newState) return;
    this.state = newState;
    this.stateListeners.forEach((listener) => {
      try {
        listener(newState);
      } catch (err) {
        console.error("[AgentSocket] error in state listener:", err);
      }
    });
  }

  // ── Connection Lifecycle ───────────────────────────────────────────────────

  /** Initiate connection to the Python sidecar. */
  public connect(): void {
    if (this.isDisposed) return;
    this.setState(this.hasConnectedOnce ? "reconnecting" : "connecting");
    this.startConnecting();
  }

  /** Terminate connection and clean up all timers. */
  public disconnect(): void {
    this.isDisposed = true;
    this.clearAllTimers();

    if (this.ws) {
      try {
        // Remove handlers before close to prevent spurious reconnect triggers
        this.ws.onopen = null;
        this.ws.onclose = null;
        this.ws.onerror = null;
        this.ws.onmessage = null;
        this.ws.close();
      } catch {
        // Ignore close errors during disconnect
      }
      this.ws = null;
    }

    this.setState("disconnected");
  }

  private clearAllTimers(): void {
    if (this.pollTimer !== null) {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.pingTimer !== null) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  private async startConnecting(): Promise<void> {
    if (this.isDisposed) return;

    try {
      // Query Rust supervisor for current connection credentials
      const info = await invoke<ConnectionInfo>("agent_core_get_connection");

      if (this.isDisposed) return;

      if (!info.ready) {
        // Sidecar not ready yet — retry after short delay
        this.pollTimer = setTimeout(() => {
          this.startConnecting();
        }, POLL_INTERVAL_MS);
        return;
      }

      // Construct authenticated WebSocket URL using query token
      // SECURITY: Do NOT log this URL as it contains the secret token
      const wsUrl = `ws://127.0.0.1:${info.port}/?token=${encodeURIComponent(info.token)}`;
      this.openSocket(wsUrl, info.port);
    } catch (err) {
      if (this.isDisposed) return;
      console.error("[AgentSocket] failed to query connection info:", err);
      this.scheduleReconnect();
    }
  }

  private openSocket(wsUrl: string, port: number): void {
    if (this.isDisposed) return;

    try {
      console.log(`[AgentSocket] connecting to ws://127.0.0.1:${port}...`);
      const socket = new WebSocket(wsUrl);
      this.ws = socket;

      socket.onopen = () => {
        if (this.isDisposed || this.ws !== socket) {
          socket.close();
          return;
        }

        console.log(`[AgentSocket] connected to sidecar on port ${port}`);
        this.hasConnectedOnce = true;
        this.reconnectDelay = INITIAL_RECONNECT_MS; // Reset backoff on success
        this.setState("connected");
        this.startPing();
        this.requestModelsList();
      };

      socket.onmessage = (event: MessageEvent) => {
        if (this.isDisposed || this.ws !== socket) return;
        this.handleMessage(event.data);
      };

      socket.onerror = () => {
        if (this.isDisposed || this.ws !== socket) return;
        console.warn("[AgentSocket] WebSocket encountered an error");
      };

      socket.onclose = () => {
        if (this.isDisposed || this.ws !== socket) return;
        console.log("[AgentSocket] WebSocket connection closed");
        this.stopPing();
        this.ws = null;
        this.scheduleReconnect();
      };
    } catch (err) {
      if (this.isDisposed) return;
      console.error("[AgentSocket] error opening WebSocket:", err);
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.isDisposed) return;

    this.clearAllTimers();
    this.setState("reconnecting");

    const delay = this.reconnectDelay;
    this.reconnectDelay = Math.min(
      this.reconnectDelay * RECONNECT_MULTIPLIER,
      MAX_RECONNECT_MS
    );

    console.log(`[AgentSocket] scheduling reconnect in ${Math.round(delay)}ms...`);
    this.reconnectTimer = setTimeout(() => {
      this.startConnecting();
    }, delay);
  }

  // ── Message Handling ────────────────────────────────────────────────────────

  private handleMessage(rawData: string): void {
    let msg: ServerMessage;
    try {
      msg = JSON.parse(rawData);
    } catch (err) {
      console.error("[AgentSocket] failed to parse incoming JSON:", err);
      return;
    }

    switch (msg.type) {
      case "pong":
        this.pongListeners.forEach((cb) => {
          try {
            cb();
          } catch (e) {
            console.error(e);
          }
        });
        break;

      case "chat.delta":
        this.deltaListeners.forEach((cb) => {
          try {
            cb(msg.payload);
          } catch (e) {
            console.error(e);
          }
        });
        break;

      case "chat.done":
        this.doneListeners.forEach((cb) => {
          try {
            cb(msg.payload);
          } catch (e) {
            console.error(e);
          }
        });
        break;

      case "chat.error":
        this.errorListeners.forEach((cb) => {
          try {
            cb(msg.payload);
          } catch (e) {
            console.error(e);
          }
        });
        break;

      case "tool.call": {
        const raw = msg as any;
        const payload: ToolCallPayload = raw.payload || {
          session_id: raw.session_id,
          call_id: raw.call_id,
          tool: raw.tool,
          args: raw.args || {},
        };
        this.toolCallListeners.forEach((cb) => {
          try {
            cb(payload);
          } catch (e) {
            console.error(e);
          }
        });
        break;
      }

      case "tool.result": {
        const raw = msg as any;
        const payload: ToolResultPayload = raw.payload || {
          session_id: raw.session_id,
          call_id: raw.call_id,
          ok: raw.ok ?? false,
          summary: raw.summary || "",
          preview: raw.preview,
          truncated: raw.truncated ?? false,
          possible_injection: raw.possible_injection ?? false,
          warning: raw.warning ?? false,
          snapshot_seq: raw.snapshot_seq ?? raw.payload?.snapshot_seq ?? null,
        };
        if (payload.warning === undefined && raw.warning !== undefined) {
          payload.warning = raw.warning;
        }
        if (payload.snapshot_seq === undefined && raw.snapshot_seq !== undefined) {
          payload.snapshot_seq = raw.snapshot_seq;
        }
        this.toolResultListeners.forEach((cb) => {
          try {
            cb(payload);
          } catch (e) {
            console.error(e);
          }
        });
        break;
      }

      case "models.list": {
        const raw = msg as any;
        const payload: ModelsListPayload = raw.payload || {
          ollama_available: false,
          models: [],
        };
        this.modelsListListeners.forEach((cb) => {
          try {
            cb(payload);
          } catch (e) {
            console.error(e);
          }
        });
        break;
      }

      case "changes.list.result": {
        const raw = msg as any;
        const changes: ChangeItem[] = raw.payload?.changes || [];
        this.changesListListeners.forEach((cb) => {
          try {
            cb(changes);
          } catch (e) {
            console.error(e);
          }
        });
        break;
      }

      case "changes.undo.result": {
        const raw = msg as any;
        const payload: ChangesUndoResultPayload = raw.payload || {
          session_id: raw.session_id,
          seq: raw.seq,
          ok: raw.ok ?? false,
          path: raw.path || "",
          action: raw.action || "failed",
          summary: raw.summary || "",
          error: raw.error,
          restored_sha256: raw.restored_sha256,
        };
        this.changesUndoResultListeners.forEach((cb) => {
          try {
            cb(payload);
          } catch (e) {
            console.error(e);
          }
        });
        this.undoResultListeners.forEach((cb) => {
          try {
            cb(payload);
          } catch (e) {
            console.error(e);
          }
        });
        break;
      }

      case "approval_request": {
        const raw = msg as any;
        const payload: ApprovalRequestPayload = raw.payload || {
          request_id: raw.request_id,
          session_id: raw.session_id,
          tool: raw.tool,
          path: raw.path,
          diff_preview: raw.diff_preview,
          bytes: raw.bytes,
          timeout: raw.timeout ?? 120,
        };
        this.approvalRequestListeners.forEach((cb) => {
          try {
            cb(payload);
          } catch (e) {
            console.error(e);
          }
        });
        break;
      }

      case "session.autonomy": {
        const raw = msg as any;
        const payload: SessionAutonomyPayload = raw.payload || {
          session_id: raw.session_id,
          autonomy_level: raw.autonomy_level ?? 2,
        };
        this.sessionAutonomyListeners.forEach((cb) => {
          try {
            cb(payload);
          } catch (e) {
            console.error(e);
          }
        });
        break;
      }

      case "context.status": {
        const raw = msg as any;
        const payload: ContextStatusPayload = raw.payload;
        this.contextStatusListeners.forEach((cb) => {
          try {
            cb(payload);
          } catch (e) {
            console.error(e);
          }
        });
        break;
      }

      default:
        console.warn("[AgentSocket] unhandled message type:", (msg as { type: string }).type);
    }
  }

  // ── Ping / Pong Health Check ────────────────────────────────────────────────

  private startPing(): void {
    this.stopPing();
    this.pingTimer = setInterval(() => {
      if (this.state === "connected" && this.ws?.readyState === WebSocket.OPEN) {
        this.sendPing();
      }
    }, PING_INTERVAL_MS);
  }

  private stopPing(): void {
    if (this.pingTimer !== null) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  // ── Outbound Commands ───────────────────────────────────────────────────────

  /** Request available models metadata from sidecar. */
  public requestModelsList(): void {
    const msg: ClientModelsListMessage = {
      id: crypto.randomUUID(),
      type: "models.list",
      payload: {},
    };
    this.sendJson(msg);
  }

  /** Send a chat message to initiate or continue generation. */
  public sendChat(
    sessionId: string,
    text: string,
    modelAlias: string,
    workspaceRoot?: string | null
  ): string {
    const id = crypto.randomUUID();
    const message: ClientChatSendMessage = {
      id,
      type: "chat.send",
      payload: {
        session_id: sessionId,
        text,
        model_alias: modelAlias,
        workspace_root: workspaceRoot ?? undefined,
      },
    };
    this.sendJson(message);
    return id;
  }

  /** Cancel in-flight generation for the specified session. */
  public cancelChat(sessionId: string): string {
    const id = crypto.randomUUID();
    const message: ClientChatCancelMessage = {
      id,
      type: "chat.cancel",
      payload: {
        session_id: sessionId,
      },
    };
    this.sendJson(message);
    return id;
  }

  /** Send a health check ping. */
  public sendPing(): string {
    const id = crypto.randomUUID();
    const message: ClientPingMessage = {
      id,
      type: "ping",
      payload: {},
    };
    this.sendJson(message);
    return id;
  }

  /** Request undo of a file edit by snapshot sequence number. */
  public sendUndo(
    sessionId: string,
    seq: number,
    workspaceRoot?: string | null,
    callId?: string | null
  ): string {
    return this.requestChangesUndo(sessionId, seq, workspaceRoot, false, callId);
  }

  /** Request list of persistent file changes / snapshots. */
  public requestChangesList(
    workspaceRoot?: string | null,
    sessionId?: string | null
  ): string {
    const id = crypto.randomUUID();
    const message: ClientChangesListMessage = {
      id,
      type: "changes.list",
      payload: {
        workspace_root: workspaceRoot ?? undefined,
        session_id: sessionId ?? undefined,
      },
    };
    this.sendJson(message);
    return id;
  }

  /** Request undo of a change by sequence number with optional force. */
  public requestChangesUndo(
    sessionId: string,
    seq: number,
    workspaceRoot?: string | null,
    force?: boolean,
    callId?: string | null
  ): string {
    const id = crypto.randomUUID();
    const message: ClientChangesUndoMessage = {
      id,
      type: "changes.undo",
      payload: {
        session_id: sessionId,
        seq,
        workspace_root: workspaceRoot ?? undefined,
        force: !!force,
        call_id: callId ?? undefined,
      },
    };
    this.sendJson(message);
    return id;
  }

  /** Send user approval decision for a pending tool call. */
  public sendApprovalResponse(
    requestId: string,
    decision: "approve" | "reject",
    sessionId?: string
  ): string {
    const id = crypto.randomUUID();
    const message: ClientApprovalResponseMessage = {
      id,
      type: "approval.response",
      payload: {
        request_id: requestId,
        decision,
        session_id: sessionId,
      },
    };
    this.sendJson(message);
    return id;
  }

  /** Set session autonomy level (2: Ask, 3: Scoped). */
  public setSessionAutonomy(
    sessionId: string,
    autonomyLevel: number
  ): string {
    const id = crypto.randomUUID();
    const message: ClientSessionSetAutonomyMessage = {
      id,
      type: "session.set_autonomy",
      payload: {
        session_id: sessionId,
        autonomy_level: autonomyLevel,
      },
    };
    this.sendJson(message);
    return id;
  }

  private sendJson(data: ClientMessage): void {

    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn("[AgentSocket] cannot send message: WebSocket is not open");
      return;
    }
    this.ws.send(JSON.stringify(data));
  }
}
