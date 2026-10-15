mod agent_core;
mod pty;

use agent_core::{AgentCoreSupervisor, ConnectionInfo};
use pty::{PtyManager, SessionId};
use tauri::{AppHandle, Manager, RunEvent, State};

// ── Tauri commands ──────────────────────────────────────────────────────────

// IMPORTANT: All PTY commands are marked `async` so Tauri runs them on a
// background thread pool instead of the main (UI) thread.  Blocking the
// main thread causes the "Not Responding" state on Windows.

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn pty_spawn(
    app: AppHandle,
    state: State<'_, PtyManager>,
    shell: String,
    cwd: String,
    cols: u16,
    rows: u16,
) -> Result<SessionId, String> {
    eprintln!("[pty_spawn] command entered: shell={shell}, cwd={cwd}, cols={cols}, rows={rows}");
    let result = state
        .spawn(app, &shell, &cwd, cols, rows)
        .map_err(|e| e.to_string());
    match &result {
        Ok(id) => eprintln!("[pty_spawn] success, session_id={id}"),
        Err(e) => eprintln!("[pty_spawn] error: {e}"),
    }
    result
}

#[tauri::command]
async fn pty_write(
    state: State<'_, PtyManager>,
    session_id: SessionId,
    data: String,
) -> Result<(), String> {
    state
        .write(session_id, data.as_bytes())
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn pty_resize(
    state: State<'_, PtyManager>,
    session_id: SessionId,
    cols: u16,
    rows: u16,
) -> Result<(), String> {
    state
        .resize(session_id, cols, rows)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn pty_kill(
    state: State<'_, PtyManager>,
    session_id: SessionId,
) -> Result<(), String> {
    state.kill(session_id).map_err(|e| e.to_string())
}

#[tauri::command]
async fn pty_kill_all(state: State<'_, PtyManager>) -> Result<(), String> {
    state.kill_all();
    Ok(())
}

// ── Agent-core sidecar command ──────────────────────────────────────────────

/// Returns current Python sidecar connection information.
///
/// The frontend uses this to establish a WebSocket connection to agent-core.
/// The token is only returned via this IPC command — it is never logged or
/// emitted through events.
#[tauri::command]
async fn agent_core_get_connection(
    state: State<'_, AgentCoreSupervisor>,
) -> Result<ConnectionInfo, String> {
    Ok(state.get_connection_info())
}

// ── Application entry point ─────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Resolve the Tauri project root at compile time for development-mode
    // path resolution.  CARGO_MANIFEST_DIR points to `src-tauri/`.
    let tauri_src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

    let supervisor = AgentCoreSupervisor::new(tauri_src_dir);

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(PtyManager::new())
        .manage(supervisor)
        .invoke_handler(tauri::generate_handler![
            greet,
            pty_spawn,
            pty_write,
            pty_resize,
            pty_kill,
            pty_kill_all,
            agent_core_get_connection,
        ])
        .setup(|app| {
            // Start the Python sidecar supervisor in a background thread.
            // This does NOT block the UI.
            let supervisor = app.state::<AgentCoreSupervisor>();
            supervisor.start();
            eprintln!("[setup] agent-core supervisor started");
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            // Shut down the Python sidecar first (stop supervision + kill child)
            let supervisor = app_handle.state::<AgentCoreSupervisor>();
            supervisor.shutdown();

            // Kill all PTY sessions to avoid orphaned shell processes.
            let manager = app_handle.state::<PtyManager>();
            manager.kill_all();
        }
    });
}
