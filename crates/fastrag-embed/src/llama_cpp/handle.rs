//! RAII lifecycle manager for a `llama-server` subprocess.
//!
//! `LlamaServerHandle::spawn` launches the subprocess, polls `/health` until
//! it returns 200, and exposes a blocking reqwest client bound to the same
//! port. `Drop` terminates the child gracefully (SIGTERM → bounded wait →
//! SIGKILL on unix; plain kill on non-unix).

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::EmbedError;

/// Minimum llama-server build number. Older builds may lack the
/// OpenAI-compatible `/v1/embeddings` endpoint or have known embedding bugs.
pub const MIN_LLAMA_SERVER_BUILD: u32 = 5000;

/// Configuration for spawning a `llama-server` subprocess.
#[derive(Debug, Clone)]
pub struct LlamaServerConfig {
    pub binary_path: PathBuf,
    pub port: u16,
    pub health_timeout: Duration,
    pub extra_args: Vec<String>,
    /// Skip the `--version` check. Useful in tests with the fake binary when
    /// `--fake-version` is not set.
    pub skip_version_check: bool,
}

impl LlamaServerConfig {
    /// Convenience constructor with a 30s health timeout.
    pub fn new(binary_path: PathBuf, port: u16) -> Self {
        Self {
            binary_path,
            port,
            health_timeout: Duration::from_secs(30),
            extra_args: Vec::new(),
            skip_version_check: false,
        }
    }
}

/// A running `llama-server` subprocess. Dropping this handle terminates the
/// subprocess.
pub struct LlamaServerHandle {
    child: Child,
    base_url: String,
    http: reqwest::blocking::Client,
    port: u16,
    /// The config used to spawn this handle, retained so [`LlamaServerHandle::restart`]
    /// can re-invoke [`LlamaServerHandle::spawn`] after a subprocess crash.
    cfg: LlamaServerConfig,
}

impl LlamaServerHandle {
    pub fn spawn(cfg: LlamaServerConfig) -> Result<Self, EmbedError> {
        if !cfg.skip_version_check {
            check_version(&cfg.binary_path, &cfg.extra_args)?;
        }

        let mut cmd = Command::new(&cfg.binary_path);
        cmd.arg("--port")
            .arg(cfg.port.to_string())
            .args(&cfg.extra_args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| EmbedError::LlamaServerSpawn(e.to_string()))?;

        let base_url = format!("http://127.0.0.1:{}", cfg.port);
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()
            .map_err(|e| EmbedError::Http(e.to_string()))?;

        let deadline = Instant::now() + cfg.health_timeout;
        let health_url = format!("{base_url}/health");

        loop {
            // Did the child die on us?
            match child.try_wait() {
                Ok(Some(status)) => {
                    let stderr = child
                        .stderr
                        .take()
                        .and_then(|mut s| {
                            let mut buf = String::new();
                            use std::io::Read as _;
                            s.read_to_string(&mut buf).ok().map(|_| buf)
                        })
                        .unwrap_or_default();
                    let detail = if stderr.is_empty() {
                        status.to_string()
                    } else {
                        format!("{status}: {stderr}")
                    };
                    return Err(EmbedError::LlamaServerExitedEarly { status: detail });
                }
                Ok(None) => {}
                Err(e) => {
                    let _ = child.kill();
                    return Err(EmbedError::LlamaServerSpawn(e.to_string()));
                }
            }

            if let Ok(resp) = http.get(&health_url).send()
                && resp.status().is_success()
            {
                break;
            }

            if Instant::now() >= deadline {
                // Best-effort cleanup before returning.
                terminate(&mut child);
                return Err(EmbedError::LlamaServerHealthTimeout {
                    port: cfg.port,
                    waited: cfg.health_timeout,
                });
            }

            thread::sleep(Duration::from_millis(100));
        }

        Ok(Self {
            child,
            base_url,
            http,
            port: cfg.port,
            cfg,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn client(&self) -> &reqwest::blocking::Client {
        &self.http
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Non-blocking aliveness probe.
    ///
    /// Returns `false` once the child subprocess has exited for any reason.
    /// A `true` return guarantees the subprocess was alive at the instant of
    /// the call; it does not guarantee it is still alive afterwards.
    pub fn check_alive(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(None) => true,           // still running
            Ok(Some(_status)) => false, // exited
            Err(_) => false,            // treat probe errors as dead
        }
    }

    /// Re-spawn the `llama-server` subprocess using the original config.
    ///
    /// Intended to be called after [`LlamaServerHandle::check_alive`] reports
    /// false. Calling it while the subprocess is still healthy will first tear
    /// down the running process (via the old child's `Drop`) and then spawn a
    /// replacement on the same port with the same GGUF.
    pub fn restart(&mut self) -> Result<(), EmbedError> {
        let cfg = self.cfg.clone();
        let new_handle = Self::spawn(cfg)?;
        *self = new_handle;
        Ok(())
    }
}

impl Drop for LlamaServerHandle {
    fn drop(&mut self) {
        terminate(&mut self.child);
    }
}

fn check_version(binary: &Path, extra_args: &[String]) -> Result<(), EmbedError> {
    let output = Command::new(binary)
        .arg("--version")
        .args(extra_args)
        .output()
        .map_err(|e| EmbedError::LlamaServerSpawn(e.to_string()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");
    let build = parse_build_number(&combined)
        .ok_or_else(|| EmbedError::LlamaServerVersionParse(combined.trim().to_string()))?;
    if build < MIN_LLAMA_SERVER_BUILD {
        return Err(EmbedError::LlamaServerVersionTooOld {
            found: build,
            minimum: MIN_LLAMA_SERVER_BUILD,
        });
    }
    Ok(())
}

/// Extract the build number from a version string like `"version: b5234 (deadbeef)"`.
fn parse_build_number(text: &str) -> Option<u32> {
    // Try 'bNNNN' tokens first (older llama.cpp format).
    for word in text.split_whitespace() {
        if let Some(num_str) = word.strip_prefix('b')
            && let Ok(n) = num_str.parse::<u32>()
        {
            return Some(n);
        }
    }
    // Newer format: "version: NNNN (hash)" — bare number after "version:".
    if let Some(rest) = text
        .split('\n')
        .find_map(|line| line.strip_prefix("version:"))
        && let Some(word) = rest.split_whitespace().next()
        && let Ok(n) = word.parse::<u32>()
    {
        return Some(n);
    }
    None
}

fn terminate(child: &mut Child) {
    #[cfg(unix)]
    {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;
        let pid = Pid::from_raw(child.id() as i32);
        let _ = kill(pid, Signal::SIGTERM);

        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => thread::sleep(Duration::from_millis(50)),
                Err(_) => break,
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_standard_version_string() {
        assert_eq!(parse_build_number("version: b5234 (deadbeef)"), Some(5234));
    }

    #[test]
    fn parse_build_only() {
        assert_eq!(parse_build_number("b9999"), Some(9999));
    }

    #[test]
    fn parse_missing_build_number() {
        assert_eq!(parse_build_number("llama-server v1.2.3"), None);
    }

    #[test]
    fn parse_build_with_trailing_text() {
        // Some builds append extra info after the number on the same word.
        // We only match clean 'bNNNN' tokens.
        assert_eq!(parse_build_number("version: b5000-beta"), None);
        assert_eq!(parse_build_number("version: b5000"), Some(5000));
    }

    #[test]
    fn parse_bare_number_after_version_colon() {
        // Newer llama.cpp releases (b8739+) output "version: 8739 (hash)" without the 'b' prefix.
        assert_eq!(parse_build_number("version: 8739 (d132f22fc)"), Some(8739));
    }

    /// Build a handle wrapping an arbitrary `Child`, bypassing the real spawn
    /// path. Test-only; callers must ensure the provided `cfg` is cloneable if
    /// they intend to exercise `restart()`.
    fn test_handle_wrapping(child: Child, cfg: LlamaServerConfig) -> LlamaServerHandle {
        let base_url = format!("http://127.0.0.1:{}", cfg.port);
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()
            .expect("reqwest client");
        let port = cfg.port;
        LlamaServerHandle {
            child,
            base_url,
            http,
            port,
            cfg,
        }
    }

    #[test]
    fn check_alive_reports_dead_after_kill() {
        // Spawn a dummy long-running process to stand in for llama-server. We
        // only exercise the `try_wait`-based aliveness probe, so any child
        // that stays up for the duration of the test works.
        let child = Command::new("sleep")
            .arg("60")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleep");
        let cfg = LlamaServerConfig {
            binary_path: PathBuf::from("/bin/sleep"),
            port: 0,
            health_timeout: Duration::from_secs(1),
            extra_args: Vec::new(),
            skip_version_check: true,
        };
        let mut handle = test_handle_wrapping(child, cfg);

        assert!(handle.check_alive(), "newly spawned handle should be alive");

        let pid = handle.pid();
        #[cfg(unix)]
        {
            use nix::sys::signal::{Signal, kill};
            use nix::unistd::Pid;
            kill(Pid::from_raw(pid as i32), Signal::SIGKILL).expect("kill");
        }
        #[cfg(not(unix))]
        {
            let _ = pid;
            unimplemented!("test is unix-only");
        }

        // Let the kernel reap the exit.
        thread::sleep(Duration::from_millis(300));

        assert!(
            !handle.check_alive(),
            "handle should report dead after SIGKILL"
        );
    }
}
