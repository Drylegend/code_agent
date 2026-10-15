use std::io::Write;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use portable_pty::{Child, MasterPty, PtySize};

/// Handle to a running PTY session.
///
/// Design note: `MasterPty` is `Send` but **not** `Sync`, so it must never
/// live inside an `Arc`.  We keep it behind a plain `Mutex` so the owning
/// `ManagedSession` (which is itself behind the manager's `Mutex<HashMap>`)
/// can call `resize` safely.
///
/// The `writer` and `child` handles *are* wrapped in `Arc<Mutex<…>>` because
/// they need to be independently cloneable into background threads (output
/// reader, wait thread) without dragging the non-Sync `MasterPty` along.
pub struct PtySession {
    /// Used only for `resize`. Never shared across threads via `Arc`.
    master: Mutex<Box<dyn MasterPty + Send>>,

    /// Cloneable handle for writing to the PTY from any thread.
    writer: Arc<Mutex<Box<dyn Write + Send>>>,

    /// Cloneable handle for kill / wait from any thread.
    child: Arc<Mutex<Box<dyn Child + Send + Sync>>>,
}

impl PtySession {
    pub fn new(
        master: Box<dyn MasterPty + Send>,
        writer: Box<dyn Write + Send>,
        child: Box<dyn Child + Send + Sync>,
    ) -> Self {
        Self {
            master: Mutex::new(master),
            writer: Arc::new(Mutex::new(writer)),
            child: Arc::new(Mutex::new(child)),
        }
    }

    // ── operations called via the manager (same thread, behind manager lock) ──

    pub fn write(&self, data: &[u8]) -> Result<()> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| anyhow::anyhow!("PTY writer mutex was poisoned"))?;

        writer.write_all(data).context("failed to write to PTY")?;
        writer.flush().context("failed to flush PTY")?;

        Ok(())
    }

    pub fn resize(&self, cols: u16, rows: u16) -> Result<()> {
        let master = self
            .master
            .lock()
            .map_err(|_| anyhow::anyhow!("PTY master mutex was poisoned"))?;

        master
            .resize(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("failed to resize PTY")
    }

    pub fn kill(&self) -> Result<()> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| anyhow::anyhow!("PTY child mutex was poisoned"))?;

        child.kill().context("failed to kill PTY child")
    }

    // ── handle extractors for background threads ─────────────────────────────

    /// Returns a cloneable child handle for the wait thread.
    /// The wait thread only needs `child` — it never touches `master`.
    pub fn child_handle(&self) -> Arc<Mutex<Box<dyn Child + Send + Sync>>> {
        Arc::clone(&self.child)
    }
}
