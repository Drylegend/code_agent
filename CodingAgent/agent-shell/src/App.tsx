import { useState, useRef, useCallback, useEffect } from "react";
import ChatPanel from "./components/ChatPanel";
import { ChangesPanel } from "./components/ChangesPanel";
import TerminalPanel from "./components/TerminalPanel";
import type { AgentSocket } from "./lib/agentSocket";
import "./App.css";

const DEFAULT_CHAT_WIDTH = 380;
const MIN_CHAT_WIDTH = 260;
const DEFAULT_CHANGES_WIDTH = 320;
const MIN_CHANGES_WIDTH = 240;

const STORAGE_CHAT_KEY = "agent_shell_chat_width";
const STORAGE_CHANGES_KEY = "agent_shell_changes_width";
const DEV_DEFAULT_WORKSPACE_ROOT = "D:\\personal_coding_agent\\CodingAgent\\agent-shell";

function App() {
  const [socket, setSocket] = useState<AgentSocket | null>(null);
  const [workspaceRoot, setWorkspaceRoot] = useState<string>(DEV_DEFAULT_WORKSPACE_ROOT);

  // Chat panel resizing & collapsing
  const [chatWidth, setChatWidth] = useState<number>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_CHAT_KEY);
      if (saved) {
        const parsed = parseInt(saved, 10);
        if (!isNaN(parsed) && parsed >= MIN_CHAT_WIDTH) return parsed;
      }
    } catch {
      // Ignore localStorage access issues
    }
    return DEFAULT_CHAT_WIDTH;
  });
  const [isChatCollapsed, setIsChatCollapsed] = useState(false);

  // Changes panel resizing & collapsing
  const [changesWidth, setChangesWidth] = useState<number>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_CHANGES_KEY);
      if (saved) {
        const parsed = parseInt(saved, 10);
        if (!isNaN(parsed) && parsed >= MIN_CHANGES_WIDTH) return parsed;
      }
    } catch {
      // Ignore localStorage access issues
    }
    return DEFAULT_CHANGES_WIDTH;
  });
  const [isChangesCollapsed, setIsChangesCollapsed] = useState(false);

  // Drag state: null | 'chat' | 'changes'
  const [draggingResizer, setDraggingResizer] = useState<"chat" | "changes" | null>(null);

  const chatWidthRef = useRef(chatWidth);
  chatWidthRef.current = chatWidth;
  const changesWidthRef = useRef(changesWidth);
  changesWidthRef.current = changesWidth;

  const startDragChat = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setDraggingResizer("chat");
  }, []);

  const startDragChanges = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setDraggingResizer("changes");
  }, []);

  const toggleChatCollapse = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsChatCollapsed((prev) => !prev);
  }, []);

  const toggleChangesCollapse = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsChangesCollapsed((prev) => !prev);
  }, []);

  useEffect(() => {
    if (!draggingResizer) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (draggingResizer === "chat") {
        const maxW = Math.max(MIN_CHAT_WIDTH, window.innerWidth - changesWidthRef.current - 300);
        const newW = Math.min(Math.max(e.clientX, MIN_CHAT_WIDTH), maxW);
        setChatWidth(newW);
        if (isChatCollapsed) setIsChatCollapsed(false);
      } else if (draggingResizer === "changes") {
        const effectiveChatW = isChatCollapsed ? 0 : chatWidthRef.current;
        const pointerOffset = e.clientX - effectiveChatW;
        const maxW = Math.max(MIN_CHANGES_WIDTH, window.innerWidth - effectiveChatW - 300);
        const newW = Math.min(Math.max(pointerOffset, MIN_CHANGES_WIDTH), maxW);
        setChangesWidth(newW);
        if (isChangesCollapsed) setIsChangesCollapsed(false);
      }
    };

    const handleMouseUp = () => {
      setDraggingResizer(null);
      try {
        localStorage.setItem(STORAGE_CHAT_KEY, String(chatWidthRef.current));
        localStorage.setItem(STORAGE_CHANGES_KEY, String(changesWidthRef.current));
      } catch {
        // Ignore localStorage error
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [draggingResizer, isChatCollapsed, isChangesCollapsed]);

  return (
    <div className={`app ${draggingResizer ? "app--resizing" : ""}`}>
      {/* ── Pane 1: Chat Panel ── */}
      <div
        className="app__chat-container"
        style={{
          width: isChatCollapsed ? 0 : `${chatWidth}px`,
          display: isChatCollapsed ? "none" : "flex",
        }}
      >
        <ChatPanel
          onSocketReady={setSocket}
          workspaceRoot={workspaceRoot}
          onWorkspaceRootChange={setWorkspaceRoot}
        />
      </div>

      {/* ── Resizer 1: Chat / Changes divider ── */}
      <div
        className={`app__resizer ${draggingResizer === "chat" ? "app__resizer--dragging" : ""} ${isChatCollapsed ? "app__resizer--collapsed" : ""}`}
        onMouseDown={startDragChat}
        title={isChatCollapsed ? "Click to expand Chat panel" : "Drag to resize Chat panel"}
      >
        <button
          className="app__collapse-btn"
          onClick={toggleChatCollapse}
          title={isChatCollapsed ? "Expand Chat" : "Collapse Chat"}
          type="button"
        >
          {isChatCollapsed ? "▶" : "◀"}
        </button>
        <div className="app__resizer-handle" />
      </div>

      {/* ── Pane 2: Changes Panel ── */}
      <div
        className="app__changes-container"
        style={{
          width: isChangesCollapsed ? 0 : `${changesWidth}px`,
          display: isChangesCollapsed ? "none" : "flex",
        }}
      >
        <ChangesPanel
          socket={socket}
          workspaceRoot={workspaceRoot}
          onWorkspaceRootChange={setWorkspaceRoot}
        />
      </div>

      {/* ── Resizer 2: Changes / Terminal divider ── */}
      <div
        className={`app__resizer ${draggingResizer === "changes" ? "app__resizer--dragging" : ""} ${isChangesCollapsed ? "app__resizer--collapsed" : ""}`}
        onMouseDown={startDragChanges}
        title={isChangesCollapsed ? "Click to expand Changes panel" : "Drag to resize Changes panel"}
      >
        <button
          className="app__collapse-btn"
          onClick={toggleChangesCollapse}
          title={isChangesCollapsed ? "Expand Changes" : "Collapse Changes"}
          type="button"
        >
          {isChangesCollapsed ? "▶" : "◀"}
        </button>
        <div className="app__resizer-handle" />
      </div>

      {/* ── Pane 3: Terminal Panel ── */}
      <div className="app__terminal-container">
        <TerminalPanel cwd={workspaceRoot} />
      </div>
    </div>
  );
}

export default App;
