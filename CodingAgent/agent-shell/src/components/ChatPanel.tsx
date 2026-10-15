import { useState, useEffect, useRef, useCallback, type KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import {
  AgentSocket,
  type ConnectionState,
  type ModelMetadata,
  type TokenUsage,
} from "../lib/agentSocket";
import { ChangesPanel } from "./ChangesPanel";
import "./ChatPanel.css";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ToolActivity {
  callId: string;
  tool: string;
  args: Record<string, unknown>;
  status: "running" | "completed" | "denied" | "error" | "interrupted";
  summary?: string;
  preview?: string | null;
  truncated?: boolean;
  possibleInjection?: boolean;
  warning?: boolean;
  expanded?: boolean;
  snapshotSeq?: number | null;
  undoStatus?: "idle" | "undoing" | "undone" | "error";
  undoMessage?: string;
  isTextMode?: boolean;
}


interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "error";
  text: string;
  usage?: TokenUsage;
  isStreaming?: boolean;
  toolActivities?: ToolActivity[];
}

function formatToolArgs(tool: string, args: Record<string, unknown>): string {
  if (tool === "list_directory") {
    return String(args.path ?? ".");
  }
  if (tool === "read_file") {
    const path = String(args.path ?? "");
    if (args.start_line) {
      return `${path}:${args.start_line}${args.end_line ? `-${args.end_line}` : ""}`;
    }
    return path;
  }
  if (tool === "write_file" || tool === "apply_patch") {
    return String(args.path ?? "");
  }
  const entries = Object.entries(args);
  if (entries.length === 0) return "";
  return entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
}

const AVAILABLE_MODELS = [
  { alias: "local-fast", label: "local-fast" },
  { alias: "local-coder", label: "local-coder" },
];

function formatErrorMessage(code: string, rawMessage: string): string {
  switch (code) {
    case "OLLAMA_UNAVAILABLE":
      return "Ollama is not running. Start Ollama and try again.";
    case "MODEL_UNAVAILABLE":
      return "The selected model is not available.";
    case "MODEL_NOT_FOUND":
      return "The specified model was not found in Ollama.";
    case "SESSION_BUSY":
      return "Another response is already being generated.";
    case "LLM_ERROR":
      return `Model error: ${rawMessage || "An error occurred while communicating with the model."}`;
    default:
      return rawMessage || "An unexpected error occurred.";
  }
}

// Explicit development/testing default workspace root for the shell UI.
// The backend security layer never depends on this; it validates whatever explicit path is supplied.
const DEV_DEFAULT_WORKSPACE_ROOT = "D:\\personal_coding_agent\\CodingAgent\\agent-shell";

// ── Component ─────────────────────────────────────────────────────────────────

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [selectedModel, setSelectedModel] = useState("local-fast");
  const [workspaceRoot, setWorkspaceRoot] = useState(DEV_DEFAULT_WORKSPACE_ROOT);
  const [connectionState, setConnectionState] = useState<ConnectionState>("connecting");
  const [isGenerating, setIsGenerating] = useState(false);
  const [modelsList, setModelsList] = useState<ModelMetadata[]>([]);
  const [ollamaAvailable, setOllamaAvailable] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<"chat" | "changes">("chat");
  const [socketInstance, setSocketInstance] = useState<AgentSocket | null>(null);

  // Single authoritative frontend session ID for this panel instance
  const sessionIdRef = useRef<string>(crypto.randomUUID());
  const socketRef = useRef<AgentSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef(true);

  // ── Auto-scroll Helper ──────────────────────────────────────────────────────

  const handleScroll = useCallback(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    // Considered near bottom if within 80px of bottom
    const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    isNearBottomRef.current = distanceToBottom < 80;
  }, []);

  const scrollToBottomIfNear = useCallback(() => {
    if (isNearBottomRef.current && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, []);

  useEffect(() => {
    scrollToBottomIfNear();
  }, [messages, scrollToBottomIfNear]);

  // ── WebSocket Lifecycle (StrictMode-safe) ───────────────────────────────────

  useEffect(() => {
    const socket = new AgentSocket();
    socketRef.current = socket;
    setSocketInstance(socket);

    // Listen to connection state changes
    const unsubState = socket.onStateChange((state) => {
      setConnectionState(state);
    });

    // Listen to models list from sidecar
    const unsubModels = socket.onModelsList((payload) => {
      setOllamaAvailable(payload.ollama_available);
      if (payload.models && payload.models.length > 0) {
        setModelsList(payload.models);
      }
    });

    // Handle streaming text chunks
    const unsubDelta = socket.onDelta((payload) => {
      if (payload.session_id !== sessionIdRef.current) return;

      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant" && last.isStreaming) {
          return [
            ...prev.slice(0, -1),
            { ...last, text: last.text + payload.text },
          ];
        }
        // Fallback: create assistant message if none exists
        return [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            text: payload.text,
            isStreaming: true,
          },
        ];
      });
    });

    // Handle tool call initiation
    const unsubToolCall = socket.onToolCall((payload) => {
      if (payload.session_id !== sessionIdRef.current) return;

      setMessages((prev) => {
        const last = prev[prev.length - 1];
        const newActivity: ToolActivity = {
          callId: payload.call_id,
          tool: payload.tool,
          args: payload.args || {},
          status: "running",
          expanded: false,
          isTextMode: payload.is_text_mode ?? false,
        };

        if (last && last.role === "assistant") {
          const activities = last.toolActivities || [];
          return [
            ...prev.slice(0, -1),
            {
              ...last,
              toolActivities: [...activities, newActivity],
            },
          ];
        }

        return [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            text: "",
            isStreaming: true,
            toolActivities: [newActivity],
          },
        ];
      });
    });

    // Handle tool result completion
    const unsubToolResult = socket.onToolResult((payload) => {
      if (payload.session_id !== sessionIdRef.current) return;

      setMessages((prev) => {
        let found = false;
        const updated = prev.map((msg) => {
          if (msg.role !== "assistant" || !msg.toolActivities) return msg;

          const updatedActivities = msg.toolActivities.map((act) => {
            if (act.callId === payload.call_id) {
              found = true;
              let status: ToolActivity["status"] = payload.ok ? "completed" : "error";
              if (!payload.ok && payload.summary && payload.summary.toLowerCase().includes("denied")) {
                status = "denied";
              }
              return {
                ...act,
                status,
                summary: payload.summary,
                preview: payload.preview,
                truncated: payload.truncated,
                possibleInjection: payload.possible_injection,
                warning: payload.warning,
                snapshotSeq: payload.snapshot_seq ?? act.snapshotSeq ?? null,
                undoStatus: act.undoStatus ?? "idle",
              };
            }
            return act;
          });

          return { ...msg, toolActivities: updatedActivities };
        });

        if (found) return updated;

        // Fallback: if unknown call_id, do not crash, append safely
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant") {
          const act: ToolActivity = {
            callId: payload.call_id,
            tool: "tool",
            args: {},
            status: payload.ok ? "completed" : "error",
            summary: payload.summary,
            preview: payload.preview,
            truncated: payload.truncated,
            possibleInjection: payload.possible_injection,
            warning: payload.warning,
            expanded: false,
            snapshotSeq: payload.snapshot_seq ?? null,
            undoStatus: "idle",
          };
          return [
            ...prev.slice(0, -1),
            {
              ...last,
              toolActivities: [...(last.toolActivities || []), act],
            },
          ];
        }

        return prev;
      });
    });

    // Handle direct Undo results
    const unsubUndoResult = socket.onUndoResult((payload) => {
      if (payload.session_id !== sessionIdRef.current) return;

      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.role !== "assistant" || !msg.toolActivities) return msg;
          const updatedActivities: ToolActivity[] = msg.toolActivities.map((act) => {
            if (act.callId === payload.call_id || (act.snapshotSeq != null && act.snapshotSeq === payload.seq)) {
              return {
                ...act,
                undoStatus: payload.ok ? ("undone" as const) : ("error" as const),
                undoMessage: (payload.summary || payload.error) ?? undefined,
              };
            }
            return act;
          });
          return { ...msg, toolActivities: updatedActivities };
        })
      );
    });

    // Handle stream completion
    const unsubDone = socket.onDone((payload) => {
      if (payload.session_id !== sessionIdRef.current) return;

      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant") {
          // Finalize any running activities so none remain stuck
          const finalizedActivities = (last.toolActivities || []).map((act) =>
            act.status === "running" ? { ...act, status: "completed" as const } : act
          );
          return [
            ...prev.slice(0, -1),
            { ...last, isStreaming: false, usage: payload.usage, toolActivities: finalizedActivities },
          ];
        }
        return prev;
      });
      setIsGenerating(false);
    });

    // Handle structured errors
    const unsubError = socket.onError((payload) => {
      const friendlyText = formatErrorMessage(payload.code, payload.message);

      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant" && last.isStreaming) {
          const finalizedActivities = (last.toolActivities || []).map((act) =>
            act.status === "running" ? { ...act, status: "interrupted" as const } : act
          );
          if (last.text.trim().length === 0 && (!last.toolActivities || last.toolActivities.length === 0)) {
            return [
              ...prev.slice(0, -1),
              { id: crypto.randomUUID(), role: "error", text: friendlyText },
            ];
          } else {
            return [
              ...prev.slice(0, -1),
              { ...last, isStreaming: false, toolActivities: finalizedActivities },
              { id: crypto.randomUUID(), role: "error", text: friendlyText },
            ];
          }
        }
        return [
          ...prev,
          { id: crypto.randomUUID(), role: "error", text: friendlyText },
        ];
      });
      setIsGenerating(false);
    });

    // Initiate connection
    socket.connect();

    // Cleanup on unmount (StrictMode safe)
    return () => {
      unsubState();
      unsubModels();
      unsubDelta();
      unsubToolCall();
      unsubToolResult();
      unsubUndoResult();
      unsubDone();
      unsubError();
      socket.disconnect();
      socketRef.current = null;
      setSocketInstance(null);
    };
  }, []);

  // ── Actions ─────────────────────────────────────────────────────────────────

  const handleUndo = useCallback((callId: string, seq: number) => {
    const socket = socketRef.current;
    if (!socket) return;

    // Set status to "undoing"
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.role !== "assistant" || !msg.toolActivities) return msg;
        return {
          ...msg,
          toolActivities: msg.toolActivities.map((act) =>
            act.callId === callId ? { ...act, undoStatus: "undoing" as const } : act
          ),
        };
      })
    );

    socket.sendUndo(sessionIdRef.current, seq, callId);
  }, []);

  const toggleToolExpanded = useCallback((messageId: string, callId: string) => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id !== messageId || !msg.toolActivities) return msg;
        return {
          ...msg,
          toolActivities: msg.toolActivities.map((act) =>
            act.callId === callId ? { ...act, expanded: !act.expanded } : act
          ),
        };
      })
    );
  }, []);

  const handleSendMessage = useCallback(() => {
    const text = inputText.trim();
    if (!text || isGenerating || connectionState !== "connected") return;

    const socket = socketRef.current;
    if (!socket) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text,
    };

    const assistantPlaceholder: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "",
      isStreaming: true,
      toolActivities: [],
    };

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setInputText("");
    setIsGenerating(true);

    // Ensure we scroll down for the new message
    isNearBottomRef.current = true;

    socket.sendChat(
      sessionIdRef.current,
      text,
      selectedModel,
      workspaceRoot.trim() || undefined
    );
  }, [inputText, isGenerating, connectionState, selectedModel, workspaceRoot]);

  const handleCancelMessage = useCallback(() => {
    if (!isGenerating) return;

    const socket = socketRef.current;
    if (socket) {
      socket.cancelChat(sessionIdRef.current);
    }

    setIsGenerating(false);
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "assistant" && last.isStreaming) {
        const finalizedActivities = (last.toolActivities || []).map((act) =>
          act.status === "running" ? { ...act, status: "interrupted" as const } : act
        );
        return [
          ...prev.slice(0, -1),
          {
            ...last,
            isStreaming: false,
            toolActivities: finalizedActivities,
            text: last.text ? `${last.text}\n\n*(Generation stopped)*` : "*(Generation stopped)*",
          },
        ];
      }
      return prev;
    });
  }, [isGenerating]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // ── Status Indicator Helper ─────────────────────────────────────────────────

  const renderStatus = () => {
    switch (connectionState) {
      case "connected":
        return <span className="chat-panel__status chat-panel__status--connected">● Connected</span>;
      case "connecting":
        return <span className="chat-panel__status chat-panel__status--connecting">○ Connecting...</span>;
      case "reconnecting":
        return <span className="chat-panel__status chat-panel__status--reconnecting">↻ Reconnecting...</span>;
      case "disconnected":
        return <span className="chat-panel__status chat-panel__status--disconnected">✕ Disconnected</span>;
    }
  };

  return (
    <div className="chat-panel">
      {/* ── Header ── */}
      <div className="chat-panel__header">
        <div className="chat-panel__header-tabs">
          <button
            type="button"
            className={`chat-panel__tab-btn ${activeTab === "chat" ? "chat-panel__tab-btn--active" : ""}`}
            onClick={() => setActiveTab("chat")}
          >
            Chat
          </button>
          <button
            type="button"
            className={`chat-panel__tab-btn ${activeTab === "changes" ? "chat-panel__tab-btn--active" : ""}`}
            onClick={() => setActiveTab("changes")}
          >
            Changes
          </button>
        </div>
        <div className="chat-panel__header-right">
          {renderStatus()}
        </div>
      </div>

      {activeTab === "changes" ? (
        <ChangesPanel
          socket={socketInstance}
          workspaceRoot={workspaceRoot}
          onWorkspaceRootChange={setWorkspaceRoot}
        />
      ) : (
        <>
          {/* ── Messages List ── */}
      <div
        className="chat-panel__messages"
        ref={messagesContainerRef}
        onScroll={handleScroll}
      >
        {messages.length === 0 ? (
          <div className="chat-panel__empty">
            <p className="chat-panel__empty-title">Ready to assist</p>
            <p className="chat-panel__empty-subtitle">
              Send a prompt using <code>{selectedModel}</code>.
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`chat-panel__message chat-panel__message--${msg.role}`}
            >
              <div className="chat-panel__message-header">
                <span className="chat-panel__message-role">
                  {msg.role === "user"
                    ? "You"
                    : msg.role === "assistant"
                    ? "Assistant"
                    : "Error"}
                </span>
                {msg.usage && (
                  <span className="chat-panel__message-usage">
                    {msg.usage.total} tokens
                  </span>
                )}
              </div>

              <div className="chat-panel__message-body">
                {msg.role === "assistant" ? (
                  <>
                    {msg.toolActivities && msg.toolActivities.length > 0 && (
                      <div className="chat-panel__tools-list">
                        {msg.toolActivities.map((act) => {
                          const argSummary = formatToolArgs(act.tool, act.args);
                          const isExpanded = !!act.expanded;
                          const canExpand = act.status !== "running" && !!(act.preview || act.summary);

                          return (
                            <div
                              key={act.callId}
                              className={`chat-panel__tool-row chat-panel__tool-row--${act.status}`}
                            >
                              <div
                                className={`chat-panel__tool-header ${canExpand ? "chat-panel__tool-header--expandable" : ""}`}
                                onClick={() => canExpand && toggleToolExpanded(msg.id, act.callId)}
                                title={canExpand ? (isExpanded ? "Collapse preview" : "Click to view preview") : undefined}
                              >
                                <div className="chat-panel__tool-left">
                                  <span className="chat-panel__tool-icon">🔧</span>
                                  <span className="chat-panel__tool-name">{act.tool}</span>
                                  {act.isTextMode && (
                                    <span
                                      className="chat-panel__tool-text-mode-tag"
                                      title="Executed via text-mode tool call fallback. You can set tool_mode: text_json in models.yaml for this model."
                                    >
                                      text-mode
                                    </span>
                                  )}
                                  {argSummary && (
                                    <span className="chat-panel__tool-args" title={argSummary}>
                                      {argSummary}
                                    </span>
                                  )}
                                </div>
                                <div className="chat-panel__tool-right">
                                  {act.status === "completed" && act.snapshotSeq != null && (
                                    <button
                                      type="button"
                                      className={`chat-panel__tool-undo-btn chat-panel__tool-undo-btn--${act.undoStatus || "idle"}`}
                                      disabled={act.undoStatus === "undoing" || act.undoStatus === "undone"}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        if (act.snapshotSeq != null) {
                                          handleUndo(act.callId, act.snapshotSeq);
                                        }
                                      }}
                                      title={
                                        act.undoStatus === "undone"
                                          ? "Edit has been undone"
                                          : act.undoStatus === "undoing"
                                          ? "Restoring snapshot..."
                                          : "Undo this edit directly from snapshot"
                                      }
                                    >
                                      {act.undoStatus === "undoing" && "Undoing..."}
                                      {act.undoStatus === "undone" && "✓ Undone"}
                                      {(!act.undoStatus || act.undoStatus === "idle" || act.undoStatus === "error") && "⎌ Undo"}
                                    </button>
                                  )}
                                  <span className={`chat-panel__tool-status chat-panel__tool-status--${act.status}`}>
                                    {act.status === "running" && "running"}
                                    {act.status === "completed" && "completed"}
                                    {act.status === "denied" && "access denied"}
                                    {act.status === "error" && "tool error"}
                                    {act.status === "interrupted" && "interrupted"}
                                  </span>
                                  {canExpand && (
                                    <span className="chat-panel__tool-toggle">
                                      {isExpanded ? "▼" : "▶"}
                                    </span>
                                  )}
                                </div>
                              </div>

                              {/* Undo result notification banner */}
                              {act.undoStatus === "undone" && (
                                <div className="chat-panel__tool-undo-banner chat-panel__tool-undo-banner--success">
                                  <span className="chat-panel__tool-undo-banner-icon">✓</span>
                                  <span>{act.undoMessage || "File successfully restored to snapshot"}</span>
                                </div>
                              )}
                              {act.undoStatus === "error" && act.undoMessage && (
                                <div className="chat-panel__tool-undo-banner chat-panel__tool-undo-banner--warning">
                                  <span className="chat-panel__tool-undo-banner-icon">⚠️</span>
                                  <span>{act.undoMessage}</span>
                                </div>
                              )}

                              {/* Warning banner for possible instruction injection */}
                              {act.possibleInjection && (
                                <div className="chat-panel__tool-warning">
                                  <span className="chat-panel__tool-warning-icon">⚠️</span>
                                  <span>Possible instruction-like content detected</span>
                                </div>
                              )}

                              {/* Warning badge for large/out-of-proportion change */}
                              {act.warning && (
                                <div className="chat-panel__tool-warning chat-panel__tool-warning--large-change">
                                  <span className="chat-panel__tool-warning-icon">⚠️</span>
                                  <span>large change, review the diff</span>
                                </div>
                              )}

                              {/* Part D: "No change was made" banner on failed write operations */}
                              {(act.tool === "write_file" || act.tool === "apply_patch") &&
                                (act.status === "error" || act.status === "denied" || act.status === "interrupted") && (
                                  <div className="chat-panel__tool-no-change-banner">
                                    <span className="chat-panel__tool-no-change-icon">ℹ️</span>
                                    <span><strong>No change was made</strong></span>
                                  </div>
                              )}

                              {/* Collapsible compact bounded preview or raw error details (Part F) */}
                              {isExpanded && (act.preview || act.summary) && (
                                <div className="chat-panel__tool-preview-container">
                                  <div className="chat-panel__tool-preview-header">
                                    <span className="chat-panel__tool-preview-label">
                                      {act.status === "denied"
                                        ? "POLICY DENIAL / SAFETY ENFORCEMENT"
                                        : act.status === "error"
                                        ? "RAW TOOL ERROR"
                                        : act.status === "interrupted"
                                        ? "EXECUTION INTERRUPTED"
                                        : "TOOL OUTPUT / DATA (preview)"}
                                    </span>
                                    {act.truncated && (
                                      <span className="chat-panel__tool-truncated-tag">truncated</span>
                                    )}
                                  </div>
                                  <pre className={`chat-panel__tool-preview-body chat-panel__tool-preview-body--${act.status}`}>
                                    {act.preview || act.summary}
                                  </pre>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                    <div className="chat-panel__markdown">
                      <ReactMarkdown>
                        {msg.text || (msg.isStreaming && (!msg.toolActivities || msg.toolActivities.length === 0) ? "…" : "")}
                      </ReactMarkdown>
                    </div>
                  </>
                ) : (
                  <p className="chat-panel__text">{msg.text}</p>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* ── Input Controls ── */}
      <div className="chat-panel__footer">
        <div className="chat-panel__controls-bar">
          <label className="chat-panel__model-label">
            Model:
            <select
              className="chat-panel__model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isGenerating}
            >
              {modelsList.length > 0
                ? modelsList.map((m) => {
                    let displayLabel = m.alias;
                    let isDisabled = false;

                    if (!ollamaAvailable) {
                      displayLabel = `${m.alias} (Ollama offline)`;
                    } else if (!m.installed) {
                      const cleanTarget = m.configured_model.replace(/^ollama\//, "");
                      displayLabel = `${m.alias} · not installed: ollama pull ${cleanTarget}`;
                      isDisabled = true;
                    } else {
                      const details: string[] = [];
                      if (m.resolved_model) details.push(m.resolved_model);
                      if (m.parameter_size) details.push(m.parameter_size);
                      if (m.quantization) details.push(m.quantization);
                      if (m.tool_mode) details.push(m.tool_mode);
                      const richStr = details.length > 0 ? details.join(" · ") : m.configured_model;
                      displayLabel = `${richStr} (${m.alias})`;
                    }

                    return (
                      <option key={m.alias} value={m.alias} disabled={isDisabled}>
                        {displayLabel}
                      </option>
                    );
                  })
                : AVAILABLE_MODELS.map((m) => (
                    <option key={m.alias} value={m.alias}>
                      {m.label}
                    </option>
                  ))}
            </select>
          </label>

          <label className="chat-panel__workspace-label" title="Explicit workspace root for tools and operations">
            Workspace:
            <input
              type="text"
              className="chat-panel__workspace-input"
              value={workspaceRoot}
              onChange={(e) => setWorkspaceRoot(e.target.value)}
              placeholder="e.g. D:\path\to\workspace"
              disabled={isGenerating}
            />
          </label>
        </div>

        <div className="chat-panel__input-row">
          <textarea
            className="chat-panel__input"
            placeholder="Type a message (Enter to send, Shift+Enter for newline)..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={connectionState !== "connected" && connectionState !== "reconnecting"}
            rows={2}
          />
          <div className="chat-panel__action-buttons">
            {isGenerating ? (
              <button
                className="chat-panel__button chat-panel__button--stop"
                onClick={handleCancelMessage}
                title="Stop generation"
              >
                Stop
              </button>
            ) : (
              <button
                className="chat-panel__button chat-panel__button--send"
                onClick={handleSendMessage}
                disabled={
                  !inputText.trim() ||
                  connectionState !== "connected" ||
                  isGenerating
                }
                title="Send message"
              >
                Send
              </button>
            )}
          </div>
        </div>
      </div>
      </>
      )}

      {/* ── Status Bar for Active Resolved Model ── */}
      <div className="chat-panel__status-bar">
        <div className="chat-panel__status-bar-left">
          <span className="chat-panel__status-bar-label">Active Model:</span>
          {(() => {
            const activeMeta = modelsList.find((m) => m.alias === selectedModel);
            if (!ollamaAvailable) {
              return (
                <span className="chat-panel__status-bar-offline">
                  Ollama offline ({selectedModel})
                </span>
              );
            }
            if (activeMeta) {
              if (activeMeta.installed) {
                const parts = [
                  activeMeta.resolved_model || activeMeta.configured_model,
                  activeMeta.parameter_size,
                  activeMeta.quantization,
                  activeMeta.tool_mode,
                ].filter(Boolean);
                return (
                  <span className="chat-panel__status-bar-model">
                    {parts.join(" · ")} <span className="chat-panel__status-bar-alias">({activeMeta.alias})</span>
                  </span>
                );
              }
              const pullCmd = activeMeta.configured_model.replace(/^ollama\//, "");
              return (
                <span className="chat-panel__status-bar-missing">
                  not installed: ollama pull {pullCmd} ({activeMeta.alias})
                </span>
              );
            }
            return (
              <span className="chat-panel__status-bar-model">{selectedModel}</span>
            );
          })()}
        </div>
        <div className="chat-panel__status-bar-right">
          {connectionState === "connected" && (
            <span className="chat-panel__status-bar-connection">Core Sidecar Ready</span>
          )}
        </div>
      </div>
    </div>
  );
}
