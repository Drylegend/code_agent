import { useState, useRef, useCallback, useEffect } from "react";
import ChatPanel from "./components/ChatPanel";
import TerminalPanel from "./components/TerminalPanel";
import "./App.css";

const DEFAULT_WIDTH = 420;
const MIN_WIDTH = 280;
const STORAGE_KEY = "agent_shell_sidebar_width";

function App() {
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = parseInt(saved, 10);
        if (!isNaN(parsed) && parsed >= MIN_WIDTH) return parsed;
      }
    } catch {
      // Ignore localStorage access issues
    }
    return DEFAULT_WIDTH;
  });

  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const sidebarWidthRef = useRef(sidebarWidth);
  sidebarWidthRef.current = sidebarWidth;

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const toggleCollapse = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsCollapsed((prev) => !prev);
  }, []);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const maxWidth = Math.max(MIN_WIDTH, window.innerWidth - 200);
      const newWidth = Math.min(Math.max(e.clientX, MIN_WIDTH), maxWidth);
      setSidebarWidth(newWidth);
      if (isCollapsed) setIsCollapsed(false);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      try {
        localStorage.setItem(STORAGE_KEY, String(sidebarWidthRef.current));
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
  }, [isDragging, isCollapsed]);

  return (
    <div className={`app ${isDragging ? "app--resizing" : ""}`}>
      <div
        className="app__chat-container"
        style={{
          width: isCollapsed ? 0 : `${sidebarWidth}px`,
          display: isCollapsed ? "none" : "flex",
        }}
      >
        <ChatPanel />
      </div>

      <div
        className={`app__resizer ${isDragging ? "app__resizer--dragging" : ""} ${isCollapsed ? "app__resizer--collapsed" : ""}`}
        onMouseDown={handleMouseDown}
        title={isCollapsed ? "Click to expand sidebar" : "Drag to resize sidebar"}
      >
        <button
          className="app__collapse-btn"
          onClick={toggleCollapse}
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          type="button"
        >
          {isCollapsed ? "▶" : "◀"}
        </button>
        <div className="app__resizer-handle" />
      </div>

      <div className="app__terminal-container">
        <TerminalPanel />
      </div>
    </div>
  );
}

export default App;
