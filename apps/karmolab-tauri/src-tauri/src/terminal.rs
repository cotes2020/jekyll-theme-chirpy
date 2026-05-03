// terminal.rs
// KarmoLab 안 단일 PowerShell IO 터미널 (TASK-KL-006 옵션 A — 단순 line-IO).
//
// 명령:
//   - terminal_start() -> TerminalStartResult       (이미 실행 중이면 already_running=true)
//   - terminal_send_stdin(line)                     (stdin write + cwd marker 자동 append)
//   - terminal_stop()                               (taskkill /T /F + state 정리)
//   - terminal_status() -> TerminalStatus
//
// 이벤트:
//   - karmolab://terminal-line  { stream: "stdout"|"stderr", line }
//   - karmolab://terminal-cwd   { cwd }   (PowerShell 의 [__KL_CWD__]<path> 마커 라인 가공)
//   - karmolab://terminal-exit  { code }
//
// 정책:
//   - 단일 셸 (멀티 탭은 후속 — xterm 옵션 B)
//   - pwsh.exe 우선, 없으면 powershell.exe
//   - 사용자 라인 뒤에 `; Write-Host "[__KL_CWD__]..."` 자동 append → cd / Set-Location 추적
//   - 트레이 종료 시 lib.rs 가 shutdown(state) 호출 → 자식 kill (orphan PowerShell 방지)

use serde::Serialize;
use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use tauri::{AppHandle, Emitter, State};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// stdout 라인이 이 prefix 로 시작하면 일반 line 이 아니라 cwd 변경 통지로 간주.
const CWD_MARKER_PREFIX: &str = "[__KL_CWD__]";

fn apply_no_window(cmd: &mut Command) {
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    #[cfg(not(windows))]
    let _ = cmd;
}

#[derive(Default)]
pub struct TerminalState {
    inner: Mutex<TerminalInner>,
}

#[derive(Default)]
struct TerminalInner {
    stdin: Option<ChildStdin>,
    child_id: Option<u32>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TerminalStartResult {
    pub running: bool,
    pub cwd: String,
    pub shell: String,
    pub already_running: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TerminalStatus {
    pub running: bool,
    pub child_id: Option<u32>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct TerminalLineEvt {
    stream: String,
    line: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct TerminalCwdEvt {
    cwd: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct TerminalExitEvt {
    code: Option<i32>,
}

/// pwsh.exe 우선 (PATH 에 있으면), 없으면 powershell.exe. 비-Windows 는 bash.
fn pick_shell() -> (String, String) {
    #[cfg(windows)]
    {
        let pwsh_exists = Command::new("cmd.exe")
            .arg("/C")
            .arg("where")
            .arg("pwsh.exe")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .creation_flags(CREATE_NO_WINDOW)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if pwsh_exists {
            return ("pwsh.exe".to_string(), "pwsh".to_string());
        }
        ("powershell.exe".to_string(), "powershell".to_string())
    }
    #[cfg(not(windows))]
    {
        ("bash".to_string(), "bash".to_string())
    }
}

/// 시작 시 셸의 working directory. USERPROFILE / HOME / current_dir 순.
fn initial_cwd() -> String {
    std::env::var("USERPROFILE")
        .ok()
        .or_else(|| std::env::var("HOME").ok())
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|p| p.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| ".".to_string())
}

#[tauri::command]
pub fn terminal_start(
    app: AppHandle,
    state: State<'_, TerminalState>,
) -> Result<TerminalStartResult, String> {
    let mut inner = state.inner.lock().map_err(|e| format!("state lock 실패: {}", e))?;
    if inner.child_id.is_some() {
        return Ok(TerminalStartResult {
            running: true,
            cwd: initial_cwd(),
            shell: "(이미 실행 중)".to_string(),
            already_running: true,
        });
    }

    let (shell_path, shell_label) = pick_shell();
    let cwd = initial_cwd();

    let mut cmd = Command::new(&shell_path);
    #[cfg(windows)]
    cmd.args(["-NoLogo"]);
    cmd.current_dir(&cwd);
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    apply_no_window(&mut cmd);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("{} 실행 실패: {}", shell_label, e))?;
    let pid = child.id();
    let stdin_handle = child
        .stdin
        .take()
        .ok_or_else(|| "stdin pipe 없음".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "stdout pipe 없음".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "stderr pipe 없음".to_string())?;

    inner.stdin = Some(stdin_handle);
    inner.child_id = Some(pid);

    spawn_reader(stdout, app.clone(), "stdout");
    spawn_reader(stderr, app.clone(), "stderr");

    let app_for_wait = app.clone();
    thread::spawn(move || {
        let exit = child.wait().ok();
        let code = exit.and_then(|s| s.code());
        let _ = app_for_wait.emit("karmolab://terminal-exit", TerminalExitEvt { code });
    });

    // 시작 직후 cwd 마커 한 번 — 위젯이 표시할 cwd 보장.
    if let Some(stdin_handle) = inner.stdin.as_mut() {
        let _ = writeln!(
            stdin_handle,
            "Write-Host \"{}$((Get-Location).Path)\"",
            CWD_MARKER_PREFIX
        );
        let _ = stdin_handle.flush();
    }

    Ok(TerminalStartResult {
        running: true,
        cwd,
        shell: shell_label,
        already_running: false,
    })
}

#[tauri::command]
pub fn terminal_send_stdin(
    line: String,
    state: State<'_, TerminalState>,
) -> Result<(), String> {
    let mut inner = state.inner.lock().map_err(|e| format!("state lock 실패: {}", e))?;
    let stdin = inner
        .stdin
        .as_mut()
        .ok_or_else(|| "터미널이 실행 중이 아닙니다".to_string())?;

    let trimmed = line.trim_end_matches(['\r', '\n']);
    let payload = if trimmed.is_empty() {
        format!("Write-Host \"{}$((Get-Location).Path)\"\n", CWD_MARKER_PREFIX)
    } else {
        format!(
            "{}; Write-Host \"{}$((Get-Location).Path)\"\n",
            trimmed, CWD_MARKER_PREFIX
        )
    };

    stdin
        .write_all(payload.as_bytes())
        .map_err(|e| format!("stdin write 실패: {}", e))?;
    stdin
        .flush()
        .map_err(|e| format!("stdin flush 실패: {}", e))?;
    Ok(())
}

#[tauri::command]
pub fn terminal_stop(state: State<'_, TerminalState>) -> Result<(), String> {
    let mut inner = state.inner.lock().map_err(|e| format!("state lock 실패: {}", e))?;
    let pid = inner.child_id.take();
    inner.stdin = None;
    if let Some(pid) = pid {
        kill_pid(pid);
    }
    Ok(())
}

#[tauri::command]
pub fn terminal_status(state: State<'_, TerminalState>) -> Result<TerminalStatus, String> {
    let inner = state.inner.lock().map_err(|e| format!("state lock 실패: {}", e))?;
    Ok(TerminalStatus {
        running: inner.child_id.is_some(),
        child_id: inner.child_id,
    })
}

/// 트레이 "종료" 시 lib.rs 에서 호출 — 자식 셸이 KarmoLab 종료 후 orphan 으로 남지 않게 강제 kill.
pub fn shutdown(state: &TerminalState) {
    if let Ok(mut inner) = state.inner.lock() {
        let pid = inner.child_id.take();
        inner.stdin = None;
        if let Some(pid) = pid {
            kill_pid(pid);
        }
    }
}

fn kill_pid(pid: u32) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/T", "/F", "/PID", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .creation_flags(CREATE_NO_WINDOW)
            .status();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

fn spawn_reader<R>(reader: R, app: AppHandle, stream: &'static str) -> thread::JoinHandle<()>
where
    R: std::io::Read + Send + 'static,
{
    thread::spawn(move || {
        let mut buf_reader = BufReader::new(reader);
        let mut buf = String::new();
        loop {
            buf.clear();
            match buf_reader.read_line(&mut buf) {
                Ok(0) => break,
                Ok(_) => {
                    let line = buf.trim_end_matches(['\r', '\n']).to_string();
                    if let Some(rest) = line.strip_prefix(CWD_MARKER_PREFIX) {
                        let _ = app.emit(
                            "karmolab://terminal-cwd",
                            TerminalCwdEvt {
                                cwd: rest.to_string(),
                            },
                        );
                    } else {
                        let _ = app.emit(
                            "karmolab://terminal-line",
                            TerminalLineEvt {
                                stream: stream.to_string(),
                                line,
                            },
                        );
                    }
                }
                Err(_) => break,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cwd_marker_prefix_shape() {
        assert!(CWD_MARKER_PREFIX.starts_with('['));
        assert!(CWD_MARKER_PREFIX.ends_with(']'));
        assert!(CWD_MARKER_PREFIX.len() > 4);
    }

    #[test]
    fn pick_shell_returns_nonempty() {
        let (path, label) = pick_shell();
        assert!(!path.is_empty(), "shell path should be non-empty");
        assert!(!label.is_empty(), "shell label should be non-empty");
    }

    #[test]
    fn initial_cwd_returns_nonempty() {
        let cwd = initial_cwd();
        assert!(!cwd.is_empty(), "initial cwd should be non-empty");
    }

    /// 마커 라인 분리 — strip_prefix 가 reader_thread 의 분기 키.
    #[test]
    fn marker_strip_prefix_extracts_path() {
        let line = format!("{}C:\\Users\\foo", CWD_MARKER_PREFIX);
        let stripped = line.strip_prefix(CWD_MARKER_PREFIX).unwrap();
        assert_eq!(stripped, "C:\\Users\\foo");
    }

    /// 일반 라인은 마커 prefix 없음 — 분기에서 line 으로 처리됨을 확인.
    #[test]
    fn regular_line_has_no_marker() {
        let line = "git status";
        assert!(line.strip_prefix(CWD_MARKER_PREFIX).is_none());
    }
}
