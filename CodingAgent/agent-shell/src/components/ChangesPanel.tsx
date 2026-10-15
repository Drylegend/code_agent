import React, { useState, useEffect, useCallback } from "react";
import { AgentSocket, ChangeItem, ChangesUndoResultPayload } from "../lib/agentSocket";
import "./ChangesPanel.css";

interface ChangesPanelProps {
  socket: AgentSocket | null;
  workspaceRoot: string;
  onWorkspaceRootChange?: (ws: string) => void;
  onClose?: () => void;
}

interface UndoModalState {
  item: ChangeItem;
  stage: "confirm" | "conflict" | "legacy";
  customWorkspace: string;
  force: boolean;
}

export const ChangesPanel: React.FC<ChangesPanelProps> = ({
  socket,
  workspaceRoot,
  onWorkspaceRootChange,
}) => {
  const [changes, setChanges] = useState<ChangeItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedChange, setSelectedChange] = useState<ChangeItem | null>(null);
  const [undoModal, setUndoModal] = useState<UndoModalState | null>(null);
  const [notice, setNotice] = useState<{ type: "success" | "error" | "info"; message: string } | null>(null);

  const fetchChanges = useCallback(() => {
    if (!socket || socket.getState() !== "connected") return;
    setLoading(true);
    socket.requestChangesList(workspaceRoot || undefined);
  }, [socket, workspaceRoot]);

  useEffect(() => {
    if (!socket) return;

    const unlist = socket.onChangesList((newChanges) => {
      setChanges(newChanges);
      setLoading(false);
    });

    const unundo = socket.onChangesUndoResult((res: ChangesUndoResultPayload) => {
      setUndoModal(null);
      if (res.ok) {
        setNotice({
          type: "success",
          message: res.summary || `Successfully restored ${res.path}`,
        });
        fetchChanges();
      } else {
        setNotice({
          type: "error",
          message: res.error || res.summary || "Undo failed",
        });
      }
    });

    fetchChanges();

    return () => {
      unlist();
      unundo();
    };
  }, [socket, fetchChanges]);

  const handleStartUndo = (item: ChangeItem) => {
    setNotice(null);
    if (!item.can_undo || item.status === "corrupted") {
      setNotice({ type: "error", message: "This snapshot is corrupted and cannot be safely restored." });
      return;
    }

    if (item.is_legacy) {
      setUndoModal({
        item,
        stage: "legacy",
        customWorkspace: workspaceRoot || "",
        force: false,
      });
      return;
    }

    if (item.status === "conflict") {
      setUndoModal({
        item,
        stage: "conflict",
        customWorkspace: item.workspace_root || workspaceRoot || "",
        force: false,
      });
      return;
    }

    setUndoModal({
      item,
      stage: "confirm",
      customWorkspace: item.workspace_root || workspaceRoot || "",
      force: false,
    });
  };

  const handleExecuteUndo = () => {
    if (!socket || !undoModal) return;
    const { item, stage, customWorkspace, force } = undoModal;

    const effectiveWs = stage === "legacy" ? customWorkspace.trim() : (item.workspace_root || customWorkspace.trim());
    if (stage === "legacy" && !effectiveWs) {
      setNotice({ type: "error", message: "Explicit workspace root is required for legacy snapshots." });
      return;
    }

    if (stage === "legacy" && effectiveWs && onWorkspaceRootChange) {
      onWorkspaceRootChange(effectiveWs);
    }

    socket.requestChangesUndo(
      item.session_id,
      item.seq,
      effectiveWs || undefined,
      force
    );
  };

  return (
    <div className="changes-panel">
      <div className="changes-panel__header">
        <div className="changes-panel__title-row">
          <h3 className="changes-panel__title">Persistent Changes & Undo</h3>
          <button
            className="changes-panel__refresh-btn"
            onClick={fetchChanges}
            disabled={loading}
            title="Refresh changes list"
          >
            {loading ? "Refreshing..." : "↻ Refresh"}
          </button>
        </div>
        <div className="changes-panel__ws-bar">
          <span className="changes-panel__ws-label">Workspace:</span>
          <span className="changes-panel__ws-value" title={workspaceRoot}>
            {workspaceRoot || "(None specified)"}
          </span>
        </div>
      </div>

      {notice && (
        <div className={`changes-panel__notice changes-panel__notice--${notice.type}`}>
          <span>{notice.message}</span>
          <button className="changes-panel__notice-close" onClick={() => setNotice(null)}>×</button>
        </div>
      )}

      <div className="changes-panel__list">
        {changes.length === 0 ? (
          <div className="changes-panel__empty">
            <p className="changes-panel__empty-title">No changes recorded</p>
            <p className="changes-panel__empty-desc">
              Edits made by the agent with pre-write snapshots will appear here and survive application restarts.
            </p>
          </div>
        ) : (
          changes.map((change) => {
            const isSelected = selectedChange?.seq === change.seq && selectedChange?.session_id === change.session_id;
            return (
              <div
                key={`${change.session_id}-${change.seq}`}
                className={`changes-panel__card ${isSelected ? "changes-panel__card--selected" : ""}`}
              >
                <div className="changes-panel__card-header">
                  <span className="changes-panel__card-path" title={change.path}>
                    {change.path}
                  </span>
                  <span className={`changes-panel__badge changes-panel__badge--${change.status}`}>
                    {change.status}
                  </span>
                </div>

                <div className="changes-panel__card-meta">
                  <span className="changes-panel__meta-tool">Tool: {change.tool || "write"}</span>
                  <span className="changes-panel__meta-time">
                    {new Date(change.timestamp).toLocaleTimeString()}
                  </span>
                </div>

                {change.is_legacy && (
                  <div className="changes-panel__legacy-tag">
                    ⚠️ Legacy snapshot — no persisted workspace root
                  </div>
                )}

                <div className="changes-panel__card-actions">
                  <button
                    className="changes-panel__btn changes-panel__btn--details"
                    onClick={() => setSelectedChange(isSelected ? null : change)}
                  >
                    {isSelected ? "Hide Details" : "View Details"}
                  </button>

                  <button
                    className="changes-panel__btn changes-panel__btn--undo"
                    onClick={() => handleStartUndo(change)}
                    disabled={!change.can_undo || change.status === "corrupted" || change.status === "undone"}
                    title={
                      change.status === "corrupted"
                        ? "Snapshot is corrupted on disk"
                        : change.status === "undone"
                        ? "Already in restored state"
                        : "Undo this change"
                    }
                  >
                    {change.status === "undone" ? "Undone" : "Undo Change"}
                  </button>
                </div>

                {isSelected && (
                  <div className="changes-panel__card-details">
                    <div className="changes-panel__detail-row">
                      <span className="detail-key">Seq:</span> <code>{change.seq}</code>
                    </div>
                    <div className="changes-panel__detail-row">
                      <span className="detail-key">Existed:</span> {change.existed ? "Yes (modified)" : "No (new file created)"}
                    </div>
                    <div className="changes-panel__detail-row">
                      <span className="detail-key">Before SHA:</span> <code>{change.before_sha256?.slice(0, 12) || "None"}</code>
                    </div>
                    <div className="changes-panel__detail-row">
                      <span className="detail-key">After SHA:</span> <code>{change.after_sha256?.slice(0, 12) || "None"}</code>
                    </div>
                    <div className="changes-panel__detail-row">
                      <span className="detail-key">Current File SHA:</span> <code>{change.current_sha256?.slice(0, 12) || "None"}</code>
                    </div>
                    <div className="changes-panel__detail-row">
                      <span className="detail-key">Session:</span> <small>{change.session_id}</small>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Confirmation & Safety Modal */}
      {undoModal && (
        <div className="changes-modal__backdrop">
          <div className="changes-modal">
            <h4 className="changes-modal__title">
              {undoModal.stage === "legacy"
                ? "Confirm Legacy Snapshot Undo"
                : undoModal.stage === "conflict"
                ? "⚠️ Conflict Detected — Overwrite Newer Changes?"
                : "Confirm Undo"}
            </h4>

            <div className="changes-modal__body">
              <p>
                File to restore: <strong>{undoModal.item.path}</strong>
              </p>
              <p>
                Restore action:{" "}
                <strong>
                  {undoModal.item.existed
                    ? "Restore original bytes (before edit)"
                    : "Delete newly created file"}
                </strong>
              </p>

              {undoModal.stage === "legacy" && (
                <div className="changes-modal__warning-box">
                  <p>
                    <strong>Security Notice:</strong> This snapshot is marked as <strong>legacy</strong> because it was created without a persisted workspace root.
                  </p>
                  <p>
                    The system will <strong>never infer or guess</strong> the workspace root. You must explicitly confirm or specify the exact workspace folder below:
                  </p>
                  <input
                    type="text"
                    className="changes-modal__input"
                    value={undoModal.customWorkspace}
                    onChange={(e) =>
                      setUndoModal({ ...undoModal, customWorkspace: e.target.value })
                    }
                    placeholder="e.g. D:\path\to\workspace"
                  />
                </div>
              )}

              {undoModal.stage === "conflict" && (
                <div className="changes-modal__conflict-box">
                  <p>
                    <strong>Conflict Warning:</strong> The file on disk currently differs from the state produced by this edit (SHA mismatch).
                  </p>
                  <p>
                    Restoring this snapshot will <strong>overwrite or delete newer changes</strong> that occurred after the edit.
                  </p>
                  <label className="changes-modal__checkbox-label">
                    <input
                      type="checkbox"
                      checked={undoModal.force}
                      onChange={(e) => setUndoModal({ ...undoModal, force: e.target.checked })}
                    />
                    I understand and want to force overwrite newer modifications.
                  </label>
                </div>
              )}

              {undoModal.stage === "confirm" && (
                <p className="changes-modal__safe-note">
                  ✓ Snapshot integrity verified. The file on disk will be restored to its exact original state (SHA: <code>{undoModal.item.before_sha256?.slice(0, 10)}...</code>).
                </p>
              )}
            </div>

            <div className="changes-modal__actions">
              <button
                className="changes-modal__btn changes-modal__btn--cancel"
                onClick={() => setUndoModal(null)}
              >
                Cancel
              </button>
              <button
                className={`changes-modal__btn changes-modal__btn--confirm ${
                  undoModal.stage === "conflict" ? "changes-modal__btn--danger" : ""
                }`}
                onClick={handleExecuteUndo}
                disabled={undoModal.stage === "conflict" && !undoModal.force}
              >
                {undoModal.stage === "conflict" ? "Force Restore" : "Confirm Restore"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChangesPanel;
