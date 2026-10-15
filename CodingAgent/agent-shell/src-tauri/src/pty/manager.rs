use std::collections::HashMap;
use std::sync::Mutex;
use std::thread;

use anyhow::{anyhow, Context, Result};
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use tauri::{AppHandle, Emitter};

use super::session::PtySession;

pub type SessionId = u64;

/// Internal wrapper stored in the sessions map.
///
/// `PtySession` is **not** `Sync` (because `MasterPty` is not `Sync`),
/// so we must never put it inside an `Arc`.  Instead we store it directly
/// in the `HashMap` which is itself behind a `Mutex` — access is always
/// serialized through `self.sessions.lock()`.
struct ManagedSession {
    session: PtySession,
}

pub struct PtyManager {
    sessions: Mutex<HashMap<SessionId, ManagedSession>>,
    next_id: Mutex<SessionId>,
}

impl PtyManager {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            next_id: Mutex::new(1),
        }
    }

    fn allocate_id(&self) -> Result<SessionId> {
        let mut next_id = self
            .next_id
            .lock()
            .map_err(|_| anyhow!("PTY ID mutex was poisoned"))?;

        let id = *next_id;

        *next_id = next_id
            .checked_add(1)
            .ok_or_else(|| anyhow!("PTY session ID exhausted"))?;

        Ok(id)
    }

    pub fn spawn(
        &self,
        app: AppHandle,
        shell: &str,
        cwd: &str,
        cols: u16,
        rows: u16,
    ) -> Result<SessionId> {
        let session_id = self.allocate_id()?;

        eprintln!("[pty] session {session_id}: openpty starting");
        let pty_system = native_pty_system();

        let pair = pty_system
            .openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("failed to create PTY")?;
        eprintln!("[pty] session {session_id}: openpty completed");

        let shell_path = match shell.to_ascii_lowercase().as_str() {
            "cmd" | "cmd.exe" => {
                std::env::var("COMSPEC")
                    .unwrap_or_else(|_| "C:\\Windows\\System32\\cmd.exe".into())
            }

            "powershell" | "powershell.exe" => {
                std::env::var("WINDIR")
                    .map(|windir| {
                        format!(
                            "{}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
                            windir
                        )
                    })
                    .unwrap_or_else(|_| {
                        "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe".into()
                    })
            }

            other => {
                return Err(anyhow!(
                    "unsupported shell '{}'; use 'powershell' or 'cmd'",
                    other
                ));
            }
        };

        let mut command = CommandBuilder::new(&shell_path);

        if shell.eq_ignore_ascii_case("cmd") || shell.eq_ignore_ascii_case("cmd.exe") {
            command.arg("/Q");
        } else {
            command.args([
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
            ]);
        }

        command.cwd(cwd);

        eprintln!("[pty] session {session_id}: shell spawn starting ({shell_path})");
        let child = pair
            .slave
            .spawn_command(command)
            .context("failed to spawn shell")?;
        eprintln!("[pty] session {session_id}: shell spawn completed");

        eprintln!("[pty] session {session_id}: acquiring writer");
        let writer = pair
            .master
            .take_writer()
            .context("failed to acquire PTY writer")?;
        eprintln!("[pty] session {session_id}: writer acquired");

        eprintln!("[pty] session {session_id}: acquiring reader");
        let reader = pair
            .master
            .try_clone_reader()
            .context("failed to acquire PTY reader")?;
        eprintln!("[pty] session {session_id}: reader acquired");

        let session = PtySession::new(
            pair.master,
            writer,
            child,
        );

        // Grab the child handle *before* moving the session into the map.
        // `child_handle()` returns an `Arc<Mutex<…>>` clone — fully Send + Sync.
        let child_handle = session.child_handle();

        {
            let mut sessions = self
                .sessions
                .lock()
                .map_err(|_| anyhow!("PTY session mutex was poisoned"))?;

            sessions.insert(
                session_id,
                ManagedSession { session },
            );
        }
        eprintln!("[pty] session {session_id}: session inserted into manager");

        // ── Output reader thread ────────────────────────────────────────────
        // Owns only the `reader` (a plain Read handle, fully Send).

        let output_app = app.clone();
        let output_session_id = session_id;

        thread::Builder::new()
            .name(format!("pty-output-{session_id}"))
            .spawn(move || {
                eprintln!("[pty] session {output_session_id}: output thread started");
                let mut reader = reader;
                let mut buffer = [0u8; 8192];

                loop {
                    match std::io::Read::read(&mut reader, &mut buffer) {
                        Ok(0) => {
                            eprintln!("[pty] session {output_session_id}: output reader EOF");
                            break;
                        }

                        Ok(count) => {
                            let output =
                                String::from_utf8_lossy(&buffer[..count]).into_owned();

                            let event = format!("pty-output-{output_session_id}");

                            if output_app.emit(&event, output).is_err() {
                                eprintln!("[pty] session {output_session_id}: emit failed, stopping output thread");
                                break;
                            }
                        }

                        Err(e) => {
                            eprintln!("[pty] session {output_session_id}: output reader error: {e}");
                            break;
                        }
                    }
                }
            })
            .context("failed to start PTY output thread")?;

        // ── Wait thread ─────────────────────────────────────────────────────
        // Owns only the `child_handle` (`Arc<Mutex<Box<dyn Child>>>>`).
        // Does NOT need the `MasterPty` at all, so no Sync issue.

        let exit_app = app.clone();

        thread::Builder::new()
            .name(format!("pty-wait-{session_id}"))
            .spawn(move || {
                eprintln!("[pty] session {session_id}: wait thread started");
                let exit_code = {
                    let mut child = match child_handle.lock() {
                        Ok(c) => c,
                        Err(_) => {
                            eprintln!("[pty] session {session_id}: child mutex poisoned in wait thread");
                            // Mutex poisoned; report failure.
                            let event = format!("pty-exit-{session_id}");
                            let _ = exit_app.emit(
                                &event,
                                serde_json::json!({ "code": 1 }),
                            );
                            return;
                        }
                    };

                    eprintln!("[pty] session {session_id}: waiting for child to exit...");
                    match child.wait() {
                        Ok(status) => {
                            let code = status.exit_code();
                            eprintln!("[pty] session {session_id}: child exited with code {code}");
                            code
                        }
                        Err(e) => {
                            eprintln!("[pty] session {session_id}: child wait error: {e}");
                            1
                        }
                    }
                };

                let event = format!("pty-exit-{session_id}");

                let _ = exit_app.emit(
                    &event,
                    serde_json::json!({
                        "code": exit_code,
                    }),
                );
            })
            .context("failed to start PTY wait thread")?;

        eprintln!("[pty] session {session_id}: spawn returning session_id={session_id}");
        Ok(session_id)
    }

    pub fn write(&self, session_id: SessionId, data: &[u8]) -> Result<()> {
        let sessions = self
            .sessions
            .lock()
            .map_err(|_| anyhow!("PTY session mutex was poisoned"))?;

        let managed = sessions
            .get(&session_id)
            .ok_or_else(|| anyhow!("PTY session {} not found", session_id))?;

        managed.session.write(data)
    }

    pub fn resize(
        &self,
        session_id: SessionId,
        cols: u16,
        rows: u16,
    ) -> Result<()> {
        let sessions = self
            .sessions
            .lock()
            .map_err(|_| anyhow!("PTY session mutex was poisoned"))?;

        let managed = sessions
            .get(&session_id)
            .ok_or_else(|| anyhow!("PTY session {} not found", session_id))?;

        managed.session.resize(cols, rows)
    }

    pub fn kill(&self, session_id: SessionId) -> Result<()> {
        let managed = {
            let mut sessions = self
                .sessions
                .lock()
                .map_err(|_| anyhow!("PTY session mutex was poisoned"))?;

            sessions
                .remove(&session_id)
                .ok_or_else(|| anyhow!("PTY session {} not found", session_id))?
        };

        managed.session.kill()
    }

    pub fn kill_all(&self) {
        let sessions = {
            let mut sessions = match self.sessions.lock() {
                Ok(sessions) => sessions,
                Err(_) => return,
            };

            std::mem::take(&mut *sessions)
        };

        for (_, managed) in sessions {
            let _ = managed.session.kill();
        }
    }
}

impl Default for PtyManager {
    fn default() -> Self {
        Self::new()
    }
}
