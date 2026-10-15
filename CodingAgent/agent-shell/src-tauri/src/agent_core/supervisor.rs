//! Python sidecar supervisor for agent-core.
//!
//! Responsible for:
//! - Generating a per-launch authentication token
//! - Choosing an available localhost port
//! - Spawning the Python sidecar process
//! - Detecting readiness via stdout ("AGENT_CORE_READY port=<port>")
//! - Restarting on unexpected crash with exponential backoff
//! - Providing connection information to the frontend
//! - Clean shutdown on application exit

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;
use serde::Serialize;

// ── Public types ──────────────────────────────────────────────────────────────

/// Connection information exposed to the frontend via a Tauri command.
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionInfo {
    pub port: u16,
    pub token: String,
    pub ready: bool,
}

// ── Internal state ────────────────────────────────────────────────────────────

/// Snapshot of sidecar state, protected by a mutex.
struct SupervisorState {
    /// Current connection info (port/token change on every restart).
    connection_info: ConnectionInfo,
    /// Handle to the running child process (Some while alive).
    child: Option<Child>,
    /// Whether the supervisor loop should keep restarting after a crash.
    should_run: bool,
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// How the supervisor locates the Python executable.
///
/// In development mode we resolve relative to the Tauri project root:
///     `../agent-core/.venv/Scripts/python.exe`
///
/// A future `Packaged` variant can point to a bundled Python.
#[derive(Debug, Clone)]
enum PythonMode {
    Development {
        /// Absolute path to the agent-core project directory.
        agent_core_dir: PathBuf,
        /// Absolute path to the venv python.exe.
        python_exe: PathBuf,
    },
}

// ── Backoff constants ─────────────────────────────────────────────────────────

const INITIAL_BACKOFF_MS: u64 = 1_000;
const MAX_BACKOFF_MS: u64 = 30_000;
const BACKOFF_MULTIPLIER: u64 = 2;
/// After running stably for this long, reset the backoff.
const STABLE_THRESHOLD: Duration = Duration::from_secs(30);

// ── Token generation ──────────────────────────────────────────────────────────

/// Generate a 32-byte (64 hex-char) cryptographically random token.
fn generate_token() -> String {
    let mut rng = rand::rng();
    let bytes: [u8; 32] = rng.random();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ── Port allocation ───────────────────────────────────────────────────────────

/// Ask the OS for an available TCP port on 127.0.0.1.
///
/// The listener is dropped immediately, releasing the port for the Python
/// sidecar to bind.  There is a small TOCTOU race, but this is the standard
/// approach when the child process chooses the port itself.
fn find_available_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("failed to bind ephemeral port")
        .local_addr()
        .expect("failed to read local addr")
        .port()
}

// ── Readiness parsing ─────────────────────────────────────────────────────────

/// Parse the Python sidecar readiness line.
///
/// Expected format: `AGENT_CORE_READY port=<number>`
fn parse_ready_line(line: &str) -> Option<u16> {
    let line = line.trim();
    if let Some(rest) = line.strip_prefix("AGENT_CORE_READY port=") {
        rest.trim().parse::<u16>().ok()
    } else {
        None
    }
}

// ── Supervisor ────────────────────────────────────────────────────────────────

pub struct AgentCoreSupervisor {
    state: Arc<Mutex<SupervisorState>>,
    mode: PythonMode,
}

impl AgentCoreSupervisor {
    /// Create a new supervisor.
    ///
    /// `tauri_src_dir` should be the `src-tauri` directory (used to resolve the
    /// relative path to `../agent-core`).  For Tauri apps this is obtained via
    /// `env!("CARGO_MANIFEST_DIR")` at compile time.
    pub fn new(tauri_src_dir: &std::path::Path) -> Self {
        // Resolve development-mode paths:
        // src-tauri/../agent-core  →  <project>/CodingAgent/agent-core
        let agent_core_dir = tauri_src_dir
            .join("..")
            .join("..")
            .join("agent-core")
            .canonicalize()
            .unwrap_or_else(|_| {
                // Fallback: try the absolute known development path
                PathBuf::from(r"D:\personal_coding_agent\CodingAgent\agent-core")
            });

        let python_exe = agent_core_dir
            .join(".venv")
            .join("Scripts")
            .join("python.exe");

        let mode = PythonMode::Development {
            agent_core_dir,
            python_exe,
        };

        let state = SupervisorState {
            connection_info: ConnectionInfo {
                port: 0,
                token: String::new(),
                ready: false,
            },
            child: None,
            should_run: true,
        };

        Self {
            state: Arc::new(Mutex::new(state)),
            mode,
        }
    }

    /// Get the current connection information snapshot.
    ///
    /// Called by the Tauri `agent_core_get_connection` command.
    pub fn get_connection_info(&self) -> ConnectionInfo {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.connection_info.clone()
    }

    /// Start the sidecar and the supervision loop in a background thread.
    ///
    /// This must NOT block the Tauri main/UI thread.
    pub fn start(&self) {
        let state = Arc::clone(&self.state);
        let mode = self.mode.clone();

        thread::Builder::new()
            .name("agent-core-supervisor".into())
            .spawn(move || {
                supervision_loop(state, mode);
            })
            .expect("failed to start agent-core supervisor thread");
    }

    /// Signal the supervisor to stop and terminate the child process.
    pub fn shutdown(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.should_run = false;
        state.connection_info.ready = false;

        if let Some(ref mut child) = state.child {
            eprintln!("[agent-core] shutting down Python sidecar (pid={})", child.id());
            let _ = child.kill();
            let _ = child.wait();
            eprintln!("[agent-core] Python sidecar terminated");
        }
        state.child = None;
    }
}

// ── Supervision loop (runs in background thread) ──────────────────────────────

fn supervision_loop(state: Arc<Mutex<SupervisorState>>, mode: PythonMode) {
    let mut backoff_ms = INITIAL_BACKOFF_MS;

    loop {
        // Check if we should keep running
        {
            let s = state.lock().unwrap_or_else(|e| e.into_inner());
            if !s.should_run {
                eprintln!("[agent-core] supervisor: should_run=false, exiting loop");
                return;
            }
        }

        // Generate fresh credentials for each launch
        let token = generate_token();
        let port = find_available_port();

        eprintln!("[agent-core] supervisor: spawning sidecar on port={port}");

        let (agent_core_dir, python_exe) = match &mode {
            PythonMode::Development {
                agent_core_dir,
                python_exe,
            } => (agent_core_dir.clone(), python_exe.clone()),
        };

        // Verify Python executable exists before attempting to spawn
        if !python_exe.exists() {
            eprintln!(
                "[agent-core] ERROR: Python executable not found at {}",
                python_exe.display()
            );
            eprintln!("[agent-core] supervisor: waiting {backoff_ms}ms before retry");
            thread::sleep(Duration::from_millis(backoff_ms));
            backoff_ms = (backoff_ms * BACKOFF_MULTIPLIER).min(MAX_BACKOFF_MS);
            continue;
        }

        // Spawn the Python process
        let spawn_result = Command::new(&python_exe)
            .args(["-m", "agent_core", "--port", &port.to_string()])
            .current_dir(&agent_core_dir)
            .env("AGENT_CORE_AUTH_TOKEN", &token)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // let Python stderr flow to Tauri console
            .spawn();

        let mut child = match spawn_result {
            Ok(child) => child,
            Err(e) => {
                eprintln!("[agent-core] ERROR: failed to spawn Python sidecar: {e}");
                thread::sleep(Duration::from_millis(backoff_ms));
                backoff_ms = (backoff_ms * BACKOFF_MULTIPLIER).min(MAX_BACKOFF_MS);
                continue;
            }
        };

        let child_pid = child.id();
        eprintln!("[agent-core] supervisor: Python process spawned (pid={child_pid})");

        // Read stdout for the readiness signal
        let stdout = child.stdout.take();
        let mut detected_ready = false;
        let mut actual_port = port;

        if let Some(stdout) = stdout {
            let reader = BufReader::new(stdout);
            for line_result in reader.lines() {
                match line_result {
                    Ok(line) => {
                        if let Some(p) = parse_ready_line(&line) {
                            actual_port = p;
                            detected_ready = true;
                            eprintln!(
                                "[agent-core] supervisor: readiness signal received (port={actual_port})"
                            );
                            break;
                        }
                        // Print non-readiness stdout lines for debugging
                        eprintln!("[agent-core] stdout: {line}");
                    }
                    Err(e) => {
                        eprintln!("[agent-core] ERROR: reading stdout failed: {e}");
                        break;
                    }
                }
            }
        }

        if !detected_ready {
            eprintln!("[agent-core] ERROR: sidecar exited before readiness signal");
            let _ = child.kill();
            let _ = child.wait();

            // Mark not ready
            {
                let mut s = state.lock().unwrap_or_else(|e| e.into_inner());
                s.connection_info.ready = false;
                s.child = None;
                if !s.should_run {
                    return;
                }
            }

            thread::sleep(Duration::from_millis(backoff_ms));
            backoff_ms = (backoff_ms * BACKOFF_MULTIPLIER).min(MAX_BACKOFF_MS);
            continue;
        }

        // ── Sidecar is ready ────────────────────────────────────────────────

        let startup_time = Instant::now();

        // Update shared state with new connection info
        {
            let mut s = state.lock().unwrap_or_else(|e| e.into_inner());
            if !s.should_run {
                // User quit while we were starting up
                let _ = child.kill();
                let _ = child.wait();
                return;
            }
            s.connection_info = ConnectionInfo {
                port: actual_port,
                token: token.clone(),
                ready: true,
            };
            s.child = Some(child);
        }

        eprintln!("[agent-core] supervisor: sidecar is ready and serving");

        // Wait for the child process to exit (blocks this thread)
        let exit_status = {
            let mut s = state.lock().unwrap_or_else(|e| e.into_inner());
            if s.child.is_some() {
                // Take the child out of the mutex so we can call wait()
                // outside the lock — avoids blocking other threads that
                // need connection_info.
                s.child.take()
            } else {
                None
            }
        };

        let exit_result = if let Some(mut child_owned) = exit_status {
            let result = child_owned.wait();
            // Child exited.  Mark not ready.
            {
                let mut s = state.lock().unwrap_or_else(|e| e.into_inner());
                s.connection_info.ready = false;
            }
            result
        } else {
            // Child was already taken (e.g. shutdown).  Stop the loop.
            return;
        };

        // ── Decide whether to restart ───────────────────────────────────────
        {
            let s = state.lock().unwrap_or_else(|e| e.into_inner());
            if !s.should_run {
                eprintln!("[agent-core] supervisor: shutdown requested, not restarting");
                return;
            }
        }

        match &exit_result {
            Ok(status) => {
                eprintln!(
                    "[agent-core] supervisor: sidecar exited (status={status})"
                );
            }
            Err(e) => {
                eprintln!(
                    "[agent-core] supervisor: error waiting for sidecar: {e}"
                );
            }
        }

        // Reset backoff if process ran stably for long enough
        if startup_time.elapsed() >= STABLE_THRESHOLD {
            backoff_ms = INITIAL_BACKOFF_MS;
            eprintln!("[agent-core] supervisor: backoff reset (process was stable)");
        }

        eprintln!(
            "[agent-core] supervisor: scheduling restart in {backoff_ms}ms"
        );
        thread::sleep(Duration::from_millis(backoff_ms));
        backoff_ms = (backoff_ms * BACKOFF_MULTIPLIER).min(MAX_BACKOFF_MS);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_generation_length_and_uniqueness() {
        let t1 = generate_token();
        let t2 = generate_token();
        // 32 bytes → 64 hex characters
        assert_eq!(t1.len(), 64, "Token must be 64 hex chars");
        assert_eq!(t2.len(), 64);
        // All characters should be valid hex
        assert!(t1.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(t2.chars().all(|c| c.is_ascii_hexdigit()));
        // Tokens must differ (cryptographically random)
        assert_ne!(t1, t2, "Consecutive tokens must be unique");
    }

    #[test]
    fn test_find_available_port() {
        let p1 = find_available_port();
        assert!(p1 >= 1024, "Port should be in user range");
        // Should be able to call multiple times
        let p2 = find_available_port();
        assert!(p2 >= 1024);
    }

    #[test]
    fn test_parse_ready_line_valid() {
        assert_eq!(parse_ready_line("AGENT_CORE_READY port=18765"), Some(18765));
        assert_eq!(
            parse_ready_line("AGENT_CORE_READY port=8080\n"),
            Some(8080)
        );
        assert_eq!(
            parse_ready_line("  AGENT_CORE_READY port=9999  "),
            Some(9999)
        );
    }

    #[test]
    fn test_parse_ready_line_invalid() {
        assert_eq!(parse_ready_line(""), None);
        assert_eq!(parse_ready_line("some other output"), None);
        assert_eq!(parse_ready_line("AGENT_CORE_READY"), None);
        assert_eq!(parse_ready_line("AGENT_CORE_READY port="), None);
        assert_eq!(parse_ready_line("AGENT_CORE_READY port=abc"), None);
    }

    #[test]
    fn test_connection_info_serialization() {
        let info = ConnectionInfo {
            port: 12345,
            token: "abc123".into(),
            ready: true,
        };
        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["port"], 12345);
        assert_eq!(json["token"], "abc123");
        assert_eq!(json["ready"], true);
    }

    #[test]
    fn test_backoff_constants() {
        assert!(INITIAL_BACKOFF_MS > 0);
        assert!(MAX_BACKOFF_MS >= INITIAL_BACKOFF_MS);
        assert!(BACKOFF_MULTIPLIER >= 2);
    }
}
