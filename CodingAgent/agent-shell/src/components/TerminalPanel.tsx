import { useEffect, useRef, useCallback, useState } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
// TEMPORARILY DISABLED for diagnostic — see WebGL note in useEffect
// import { WebglAddon } from "@xterm/addon-webgl";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import "@xterm/xterm/css/xterm.css";
import "./TerminalPanel.css";

// ── Configuration ───────────────────────────────────────────────────────────

/** Default shell to spawn. Will be configurable via a future shell selector. */
const DEFAULT_SHELL = "powershell";

/**
 * Default working directory.
 *
 * We use the USERPROFILE environment variable as the initial CWD, which is the
 * Windows equivalent of $HOME. This is a reasonable default for an interactive
 * terminal.  A future iteration can obtain the CWD dynamically via a Tauri
 * command or let the user configure it.
 *
 * NOTE: On Windows, %USERPROFILE% is typically "C:\Users\<name>".
 * We pass a static fallback here; the Rust backend will resolve the actual
 * shell in the real working directory.
 */
const DEFAULT_CWD = "C:\\";

// ── Terminal theme ──────────────────────────────────────────────────────────

const TERMINAL_THEME = {
  background: "#1a1b26",
  foreground: "#c0caf5",
  cursor: "#c0caf5",
  cursorAccent: "#1a1b26",
  selectionBackground: "#33467c",
  selectionForeground: "#c0caf5",
  black: "#15161e",
  red: "#f7768e",
  green: "#9ece6a",
  yellow: "#e0af68",
  blue: "#7aa2f7",
  magenta: "#bb9af7",
  cyan: "#7dcfff",
  white: "#a9b1d6",
  brightBlack: "#414868",
  brightRed: "#f7768e",
  brightGreen: "#9ece6a",
  brightYellow: "#e0af68",
  brightBlue: "#7aa2f7",
  brightMagenta: "#bb9af7",
  brightCyan: "#7dcfff",
  brightWhite: "#c0caf5",
};

// ── Props ───────────────────────────────────────────────────────────────────

interface TerminalPanelProps {
  /** Shell to spawn. Defaults to "powershell". */
  shell?: string;
  /** Initial working directory. Defaults to "C:\\". */
  cwd?: string;
}

// ── Component ───────────────────────────────────────────────────────────────

export default function TerminalPanel({
  shell = DEFAULT_SHELL,
  cwd = DEFAULT_CWD,
}: TerminalPanelProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [activeShell, setActiveShell] = useState<string>(shell);
  const [exited, setExited] = useState(false);
  const exitedRef = useRef(false);

  // Keep the ref in sync so the restart callback can read it without a stale closure.
  exitedRef.current = exited;

  const restart = useCallback(() => {
    // Force a full remount by toggling exited state, which triggers the effect.
    setExited(false);
  }, []);

  useEffect(() => {
    // If the process exited and the user hasn't requested a restart, bail.
    if (exited) return;

    const container = containerRef.current;
    if (!container) return;

    console.log("[terminal] component mount — starting setup");

    // ── Abort flag for async safety ───────────────────────────────────────
    let disposed = false;

    // ── Xterm setup ──────────────────────────────────────────────────────
    console.log("[terminal] creating Terminal instance");
    const term = new Terminal({
      cursorBlink: true,
      cursorStyle: "bar",
      scrollback: 10000,
      fontFamily: "'Cascadia Code', 'Fira Code', 'JetBrains Mono', Menlo, Monaco, 'Courier New', monospace",
      fontSize: 14,
      lineHeight: 1.2,
      theme: TERMINAL_THEME,
      allowProposedApi: true,
    });

    console.log("[terminal] loading FitAddon");
    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);

    console.log("[terminal] calling terminal.open()");
    term.open(container);
    console.log("[terminal] terminal.open() completed");

    // WebGL addon TEMPORARILY DISABLED for diagnostic purposes.
    // The WebglAddon import is kept but not loaded.
    console.log("[terminal] WebGL addon DISABLED (diagnostic mode)");

    console.log("[terminal] calling fitAddon.fit()");
    fitAddon.fit();
    console.log("[terminal] fitAddon.fit() completed");

    // ── Collect disposables ──────────────────────────────────────────────
    const unlisteners: UnlistenFn[] = [];
    let onDataDisposable: { dispose: () => void } | null = null;
    let resizeObserver: ResizeObserver | null = null;
    let sessionId: number | null = null;
    let resizeTimer: ReturnType<typeof setTimeout> | null = null;

    // ── Spawn and wire up ────────────────────────────────────────────────
    const cols = term.cols;
    const rows = term.rows;

    (async () => {
      try {
        console.log("[terminal] before pty_spawn, cols=%d, rows=%d", cols, rows);
        const id = await invoke<number>("pty_spawn", {
          shell: activeShell,
          cwd,
          cols,
          rows,
        });
        console.log("[terminal] after pty_spawn, session_id=%d", id);

        if (disposed) {
          // Component unmounted before spawn returned — kill the orphan.
          console.log("[terminal] disposed before spawn returned, killing orphan session=%d", id);
          invoke("pty_kill", { sessionId: id }).catch(() => {});
          return;
        }

        sessionId = id;

        // ── PTY output → xterm ─────────────────────────────────────────
        console.log("[terminal] registering pty-output listener");
        const unlistenOutput = await listen<string>(
          `pty-output-${id}`,
          (event) => {
            if (!disposed) {
              term.write(event.payload);
            }
          }
        );
        unlisteners.push(unlistenOutput);
        console.log("[terminal] pty-output listener registered");

        // ── PTY exit event ─────────────────────────────────────────────
        console.log("[terminal] registering pty-exit listener");
        const unlistenExit = await listen<{ code: number }>(
          `pty-exit-${id}`,
          (event) => {
            if (!disposed) {
              console.log("[terminal] exit event received, code=%d", event.payload.code);
              const code = event.payload.code;
              term.write(
                `\r\n\x1b[90m[process exited with code ${code}]\x1b[0m\r\n`
              );
              term.write("\x1b[90mPress any key to restart…\x1b[0m\r\n");
              setExited(true);
            }
          }
        );
        unlisteners.push(unlistenExit);
        console.log("[terminal] pty-exit listener registered");

        // ── Keyboard input → PTY ───────────────────────────────────────
        console.log("[terminal] registering onData handler");
        onDataDisposable = term.onData((data: string) => {
          if (exitedRef.current) {
            // Process has exited — any key restarts.
            setExited(false);
            return;
          }
          invoke("pty_write", { sessionId: id, data }).catch(() => {});
        });

        // ── Clipboard: Ctrl+Shift+C / Ctrl+Shift+V / right-click paste ─
        term.attachCustomKeyEventHandler((event: KeyboardEvent) => {
          if (event.type !== "keydown") return true;

          // Ctrl+Shift+C → copy selection
          if (event.ctrlKey && event.shiftKey && event.key === "C") {
            const sel = term.getSelection();
            if (sel) {
              navigator.clipboard.writeText(sel).catch(() => {});
            }
            return false; // prevent terminal from processing
          }

          // Ctrl+Shift+V → paste
          if (event.ctrlKey && event.shiftKey && event.key === "V") {
            navigator.clipboard.readText().then((text) => {
              if (text && !exitedRef.current) {
                invoke("pty_write", { sessionId: id, data: text }).catch(
                  () => {}
                );
              }
            }).catch(() => {});
            return false;
          }

          return true; // let all other keys (including Ctrl+C) pass through
        });

        // Right-click paste
        const handleContextMenu = (e: MouseEvent) => {
          e.preventDefault();
          navigator.clipboard.readText().then((text) => {
            if (text && !exitedRef.current && sessionId !== null) {
              invoke("pty_write", { sessionId: id, data: text }).catch(
                () => {}
              );
            }
          }).catch(() => {});
        };
        container.addEventListener("contextmenu", handleContextMenu);

        // ── Resize handling ────────────────────────────────────────────
        console.log("[terminal] setting up ResizeObserver");
        let isResizing = false;
        resizeObserver = new ResizeObserver(() => {
          // Guard against recursive resize loops:
          // fit() can change the container size, which would re-trigger
          // the ResizeObserver. The isResizing flag breaks the cycle.
          if (isResizing) return;

          // Debounce resize to avoid excessive calls.
          if (resizeTimer !== null) {
            clearTimeout(resizeTimer);
          }
          resizeTimer = setTimeout(() => {
            if (disposed) return;
            isResizing = true;
            try {
              fitAddon.fit();
              const newCols = term.cols;
              const newRows = term.rows;
              invoke("pty_resize", {
                sessionId: id,
                cols: newCols,
                rows: newRows,
              }).catch(() => {});
            } catch {
              // fit can throw if the container is detached
            } finally {
              isResizing = false;
            }
          }, 100);
        });
        resizeObserver.observe(container);
        console.log("[terminal] ResizeObserver registered");

        // Also push contextmenu cleanup
        unlisteners.push(() => {
          container.removeEventListener("contextmenu", handleContextMenu);
        });

        console.log("[terminal] all event listeners registered");
      } catch (err) {
        if (!disposed) {
          const message =
            err instanceof Error ? err.message : String(err);
          console.error("[terminal] spawn error:", message);
          term.write(`\x1b[31mFailed to spawn shell: ${message}\x1b[0m\r\n`);
          term.write("\x1b[90mPress any key to retry…\x1b[0m\r\n");
          onDataDisposable = term.onData(() => {
            setExited(false); // triggers remount / retry
          });
        }
      }
    })();

    // ── Cleanup (StrictMode-safe) ────────────────────────────────────────
    return () => {
      console.log("[terminal] cleanup started");
      disposed = true;

      if (resizeTimer !== null) {
        clearTimeout(resizeTimer);
      }

      resizeObserver?.disconnect();
      onDataDisposable?.dispose();

      for (const unlisten of unlisteners) {
        unlisten();
      }

      // Kill the PTY session if still alive.
      if (sessionId !== null) {
        invoke("pty_kill", { sessionId }).catch(() => {});
      }

      term.dispose();
      console.log("[terminal] cleanup completed");
    };
    // Re-run the effect when activeShell/cwd changes or when restarting after exit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeShell, cwd, exited]);

  return (
    <div className="terminal-panel">
      <div className="terminal-panel__header">
        <div className="terminal-panel__header-left">
          <span className="terminal-panel__title">Terminal</span>
          <select
            className="terminal-panel__shell-select"
            value={activeShell}
            onChange={(e) => {
              setActiveShell(e.target.value);
              setExited(false);
            }}
            title="Switch active shell session (PowerShell or cmd)"
          >
            <option value="powershell">PowerShell</option>
            <option value="cmd">Command Prompt (cmd)</option>
          </select>
        </div>
        {exited && (
          <button
            className="terminal-panel__restart-btn"
            onClick={restart}
            title="Restart shell"
          >
            ↻ Restart
          </button>
        )}
      </div>
      <div ref={containerRef} className="terminal-panel__container" />
    </div>
  );
}
