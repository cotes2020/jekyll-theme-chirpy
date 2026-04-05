use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::State;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const CONFIG_REL_PATH: &str = "apps/karmolab/data/servermonitor-config.json";

#[derive(Default)]
pub struct LocalDevState {
    pub repo_root: Mutex<Option<String>>,
    pub pids: Mutex<HashMap<String, u32>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DevProfile {
    id: String,
    #[allow(dead_code)]
    label: String,
    cwd: String,
    program: String,
    args: Vec<String>,
    #[serde(default)]
    npm_install: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ServerMonitorFile {
    #[serde(default)]
    dev_profiles: Vec<DevProfile>,
}

fn args_are_safe(args: &[String]) -> bool {
    for a in args {
        if a.chars()
            .any(|c| matches!(c, ';' | '|' | '&' | '$' | '`' | '\n' | '\r'))
        {
            return false;
        }
    }
    true
}

fn program_allowed(program: &str) -> bool {
    let lower = program.to_lowercase();
    let base = lower
        .trim_end_matches(".exe")
        .trim_end_matches(".cmd")
        .trim_end_matches(".bat");
    matches!(base, "npm" | "npx" | "bundle" | "ruby" | "node")
}

fn normalize_repo_root(path: &str) -> Result<String, String> {
    let t = path.trim();
    if t.is_empty() {
        return Err("경로가 비어 있습니다.".into());
    }
    let p = PathBuf::from(t);
    if !p.is_dir() {
        return Err("폴더가 아니거나 없습니다.".into());
    }
    let canon = std::fs::canonicalize(&p).map_err(|e| e.to_string())?;
    canon
        .to_str()
        .ok_or_else(|| "경로에 비 UTF-8 문자가 있습니다.".into())
        .map(|s| s.to_string())
}

fn read_profiles(repo: &Path) -> Result<Vec<DevProfile>, String> {
    let cfg_path = repo.join(CONFIG_REL_PATH);
    if !cfg_path.is_file() {
        return Err(format!(
            "dev 프로필 설정이 없습니다: {}",
            cfg_path.display()
        ));
    }
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| format!("설정 읽기 실패 ({}): {}", cfg_path.display(), e))?;
    let parsed: ServerMonitorFile =
        serde_json::from_str(&raw).map_err(|e| format!("JSON 파싱 실패: {}", e))?;
    Ok(parsed.dev_profiles)
}

fn profile_by_id<'a>(profiles: &'a [DevProfile], id: &str) -> Result<&'a DevProfile, String> {
    profiles
        .iter()
        .find(|p| p.id == id)
        .ok_or_else(|| format!("알 수 없는 프로필: {}", id))
}

fn resolve_cwd(repo: &Path, profile: &DevProfile) -> Result<PathBuf, String> {
    let joined = repo.join(profile.cwd.trim());
    if !joined.is_dir() {
        return Err(format!("cwd가 폴더가 아님: {}", joined.display()));
    }
    let canon_repo = std::fs::canonicalize(repo).map_err(|e| e.to_string())?;
    let canon_cwd = std::fs::canonicalize(&joined).map_err(|e| e.to_string())?;
    if canon_cwd.strip_prefix(&canon_repo).is_err() {
        return Err("cwd가 저장소 루트 밖입니다.".to_string());
    }
    Ok(canon_cwd)
}

fn apply_no_window(cmd: &mut Command) {
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
}

/// Windows에서 `npm`/`npx`는 `cmd /C`로 실행해 PATH의 `.cmd` 런처와 맞춘다.
fn spawn_detached_process(program: &str, args: &[String], cwd: &Path) -> Result<u32, String> {
    if !program_allowed(program) {
        return Err(format!("허용되지 않은 program: {}", program));
    }
    if !args_are_safe(args) {
        return Err("인자에 허용되지 않은 문자가 있습니다.".into());
    }

    let mut cmd;
    #[cfg(windows)]
    {
        let p = program.to_lowercase();
        let base = p
            .trim_end_matches(".exe")
            .trim_end_matches(".cmd")
            .trim_end_matches(".bat");
        // .bat/.cmd 런처(bundle, 일부 ruby 설치)는 CreateProcess로 직접 실행하면 실패하는 경우가 많음
        if matches!(base, "npm" | "npx" | "bundle" | "ruby") {
            let mut c = Command::new("cmd.exe");
            c.arg("/C").arg(program).args(args);
            cmd = c;
        } else {
            cmd = Command::new(program);
            cmd.args(args);
        }
    }
    #[cfg(not(windows))]
    {
        cmd = Command::new(program);
        cmd.args(args);
    }

    cmd.current_dir(cwd);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    apply_no_window(&mut cmd);

    let mut child = cmd.spawn().map_err(|e| format!("실행 실패: {}", e))?;
    let pid = child.id();

    std::thread::spawn(move || {
        let _ = child.wait();
    });

    Ok(pid)
}

fn run_npm_install_blocking(cwd: &Path) -> Result<String, String> {
    let mut cmd;
    #[cfg(windows)]
    {
        let mut c = Command::new("cmd.exe");
        c.arg("/C").arg("npm").arg("install");
        cmd = c;
    }
    #[cfg(not(windows))]
    {
        cmd = Command::new("npm");
        cmd.arg("install");
    }
    cmd.current_dir(cwd);
    cmd.stdin(Stdio::null());
    apply_no_window(&mut cmd);

    let output = cmd.output().map_err(|e| format!("npm install 실행 실패: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "npm install 실패 (exit {})\n{}\n{}",
            output.status,
            stderr.trim(),
            stdout.trim()
        ));
    }
    Ok("npm install 완료".into())
}

#[cfg(windows)]
fn kill_process_tree(pid: u32) -> Result<(), String> {
    let status = Command::new("taskkill.exe")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .creation_flags(CREATE_NO_WINDOW)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("taskkill 실패 (PID {})", pid));
    }
    Ok(())
}

#[cfg(not(windows))]
fn kill_process_tree(pid: u32) -> Result<(), String> {
    let p = pid.to_string();
    let s = Command::new("kill")
        .args(["-TERM", &p])
        .status()
        .map_err(|e| e.to_string())?;
    if s.success() {
        return Ok(());
    }
    let _ = Command::new("kill").args(["-9", &p]).status();
    Ok(())
}

#[tauri::command]
pub fn localdev_set_repo_root(path: String, state: State<'_, LocalDevState>) -> Result<(), String> {
    let normalized = normalize_repo_root(&path)?;
    let mut g = state.repo_root.lock().map_err(|e| e.to_string())?;
    *g = Some(normalized);
    Ok(())
}

#[tauri::command]
pub fn localdev_get_repo_root(state: State<'_, LocalDevState>) -> Option<String> {
    state
        .repo_root
        .lock()
        .ok()
        .and_then(|g| g.clone())
}

#[tauri::command]
pub fn localdev_list_tracked(state: State<'_, LocalDevState>) -> Result<Vec<String>, String> {
    let pids = state.pids.lock().map_err(|e| e.to_string())?;
    Ok(pids.keys().cloned().collect())
}

#[tauri::command]
pub fn localdev_start(profile_id: String, state: State<'_, LocalDevState>) -> Result<(), String> {
    let repo_str = {
        let g = state.repo_root.lock().map_err(|e| e.to_string())?;
        g.clone()
            .ok_or_else(|| "저장소 루트를 먼저 설정하세요.".to_string())?
    };
    let repo = PathBuf::from(&repo_str);

    {
        let pids = state.pids.lock().map_err(|e| e.to_string())?;
        if pids.contains_key(&profile_id) {
            return Err("이미 추적 중인 프로세스가 있습니다. 먼저 종료하세요.".into());
        }
    }

    let profiles = read_profiles(&repo)?;
    let profile = profile_by_id(&profiles, &profile_id)?;
    if !program_allowed(&profile.program) {
        return Err(format!("허용되지 않은 program: {}", profile.program));
    }
    if !args_are_safe(&profile.args) {
        return Err("프로필 인자에 허용되지 않은 문자가 있습니다.".into());
    }

    let cwd = resolve_cwd(&repo, profile)?;
    let pid = spawn_detached_process(&profile.program, &profile.args, &cwd)?;

    let mut pids = state.pids.lock().map_err(|e| e.to_string())?;
    pids.insert(profile_id, pid);
    Ok(())
}

#[tauri::command]
pub fn localdev_stop(profile_id: String, state: State<'_, LocalDevState>) -> Result<(), String> {
    let pid = {
        let mut pids = state.pids.lock().map_err(|e| e.to_string())?;
        pids.remove(&profile_id)
    };
    let Some(pid) = pid else {
        return Err("실행 중으로 기록된 프로세스가 없습니다.".into());
    };
    kill_process_tree(pid)?;
    Ok(())
}

#[tauri::command]
pub fn localdev_npm_install(profile_id: String, state: State<'_, LocalDevState>) -> Result<String, String> {
    let repo_str = {
        let g = state.repo_root.lock().map_err(|e| e.to_string())?;
        g.clone()
            .ok_or_else(|| "저장소 루트를 먼저 설정하세요.".to_string())?
    };
    let repo = PathBuf::from(&repo_str);

    let profiles = read_profiles(&repo)?;
    let profile = profile_by_id(&profiles, &profile_id)?;
    if !profile.npm_install {
        return Err("이 프로필은 npm install이 비활성화되어 있습니다.".into());
    }

    let cwd = resolve_cwd(&repo, profile)?;
    run_npm_install_blocking(&cwd)
}
