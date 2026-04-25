use serde::Deserialize;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{Emitter, Manager, State};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const CONFIG_REL_PATH: &str = "apps/karmolab/data/servermonitor-config.json";

#[derive(Default)]
pub struct LocalDevState {
    pub repo_root: Mutex<Option<String>>,
    pub pids: Mutex<HashMap<String, u32>>,
    /// 프로필당 동시에 하나의 스트리밍 npm/deploy 작업만 허용
    stream_busy: Mutex<HashSet<String>>,
    /// profile_id → 살아있는 로그 follow thread의 stop flag.
    /// `localdev_follow_log`로 등록되고 `localdev_stop_log_follow` 또는
    /// thread 자체 종료 시점에 제거된다.
    log_followers: Mutex<HashMap<String, Arc<AtomicBool>>>,
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
    /// e.g. `["run", "deploy:yawnbot"]` — optional **Deploy** button in Server Monitor
    #[serde(default)]
    deploy_args: Option<Vec<String>>,
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

/// 카모랩 로그 디렉토리. `<app_local_data_dir>/localdev-logs/`. 매 호출 시 dir 보장.
fn log_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let base = app
        .path()
        .app_local_data_dir()
        .map_err(|e| format!("app_local_data_dir 조회 실패: {}", e))?;
    let dir = base.join("localdev-logs");
    fs::create_dir_all(&dir).map_err(|e| format!("로그 디렉토리 생성 실패: {}", e))?;
    Ok(dir)
}

/// 프로필 로그 파일 경로 (디렉토리는 보장됨).
fn log_file_path(app: &tauri::AppHandle, profile_id: &str) -> Result<PathBuf, String> {
    Ok(log_dir(app)?.join(format!("{}.log", profile_id)))
}

/// Windows에서 `npm`/`npx`는 `cmd /C`로 실행해 PATH의 `.cmd` 런처와 맞춘다.
/// stdout/stderr는 `log_path`에 truncate redirect — 카모랩이 죽어도 자식이
/// 직접 파일 핸들을 들고 있으므로 계속 기록된다.
fn spawn_detached_process(
    program: &str,
    args: &[String],
    cwd: &Path,
    log_path: &Path,
) -> Result<u32, String> {
    if !program_allowed(program) {
        return Err(format!("허용되지 않은 program: {}", program));
    }
    if !args_are_safe(args) {
        return Err("인자에 허용되지 않은 문자가 있습니다.".into());
    }

    let log_file = File::create(log_path)
        .map_err(|e| format!("로그 파일 생성 실패 ({}): {}", log_path.display(), e))?;
    let log_file_err = log_file
        .try_clone()
        .map_err(|e| format!("로그 파일 핸들 복제 실패: {}", e))?;

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
    cmd.stdout(Stdio::from(log_file));
    cmd.stderr(Stdio::from(log_file_err));
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

fn run_npm_deploy_blocking(cwd: &Path, deploy_args: &[String]) -> Result<String, String> {
    if deploy_args.is_empty() {
        return Err("deploy_args가 비어 있습니다.".into());
    }
    if !args_are_safe(deploy_args) {
        return Err("deploy 인자에 허용되지 않은 문자가 있습니다.".into());
    }

    let mut cmd;
    #[cfg(windows)]
    {
        let mut c = Command::new("cmd.exe");
        c.arg("/C").arg("npm").args(deploy_args);
        cmd = c;
    }
    #[cfg(not(windows))]
    {
        cmd = Command::new("npm");
        cmd.args(deploy_args);
    }
    cmd.current_dir(cwd);
    cmd.stdin(Stdio::null());
    apply_no_window(&mut cmd);

    let output = cmd
        .output()
        .map_err(|e| format!("npm deploy 스크립트 실행 실패: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "deploy 실패 (exit {})\n{}\n{}",
            output.status,
            stderr.trim(),
            stdout.trim()
        ));
    }
    Ok("deploy 완료".into())
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocaldevLogLineEvt {
    run_id: String,
    profile_id: String,
    stream: String,
    line: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocaldevLogDoneEvt {
    run_id: String,
    profile_id: String,
    kind: String,
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<i32>,
}

fn pipe_reader_thread(
    pipe: impl std::io::Read + Send + 'static,
    app: tauri::AppHandle,
    run_id: String,
    profile_id: String,
    stream: &'static str,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut reader = BufReader::new(pipe);
        let mut buf = String::new();
        loop {
            buf.clear();
            match reader.read_line(&mut buf) {
                Ok(0) => break,
                Ok(_) => {
                    let line = buf.trim_end_matches(['\r', '\n']).to_string();
                    let payload = LocaldevLogLineEvt {
                        run_id: run_id.clone(),
                        profile_id: profile_id.clone(),
                        stream: stream.to_string(),
                        line,
                    };
                    let _ = app.emit("localdev-log", &payload);
                }
                Err(_) => break,
            }
        }
    })
}

fn npm_install_piped_command(cwd: &Path) -> Command {
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
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    apply_no_window(&mut cmd);
    cmd
}

fn npm_deploy_piped_command(cwd: &Path, deploy_args: &[String]) -> Command {
    let mut cmd;
    #[cfg(windows)]
    {
        let mut c = Command::new("cmd.exe");
        c.arg("/C").arg("npm").args(deploy_args);
        cmd = c;
    }
    #[cfg(not(windows))]
    {
        cmd = Command::new("npm");
        cmd.args(deploy_args);
    }
    cmd.current_dir(cwd);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    apply_no_window(&mut cmd);
    cmd
}

fn run_npm_command_streamed(
    app: tauri::AppHandle,
    profile_id: String,
    kind: &'static str,
    ok_msg: &'static str,
    err_label: &'static str,
    mut cmd: Command,
) -> Result<String, String> {
    let run_id = uuid::Uuid::new_v4().to_string();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("{err_label} 실행 실패: {e}", err_label = err_label))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| format!("{err_label}: stdout 파이프 없음"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| format!("{err_label}: stderr 파이프 없음"))?;

    let h_out = pipe_reader_thread(
        stdout,
        app.clone(),
        run_id.clone(),
        profile_id.clone(),
        "out",
    );
    let h_err = pipe_reader_thread(
        stderr,
        app.clone(),
        run_id.clone(),
        profile_id.clone(),
        "err",
    );

    let status = child
        .wait()
        .map_err(|e| format!("{err_label} 대기 실패: {e}", err_label = err_label))?;

    let _ = h_out.join();
    let _ = h_err.join();

    let success = status.success();
    let code = status.code();
    let done = LocaldevLogDoneEvt {
        run_id: run_id.clone(),
        profile_id: profile_id.clone(),
        kind: kind.to_string(),
        success,
        code,
    };
    let _ = app.emit("localdev-log-done", &done);

    if success {
        Ok(ok_msg.into())
    } else {
        Err(format!(
            "{err_label} 실패 (exit {})\n로그는 카드 패널을 확인하세요.",
            status,
        ))
    }
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

/// 프로필 로그 파일을 background에서 tail. 새 라인이 들어올 때마다
/// `localdev-log` 이벤트로 emit (run_id="follow"). 파일이 아직 없거나
/// EOF에 도달하면 짧게 sleep 후 재시도. `localdev_stop_log_follow`로 중단.
/// 같은 profile_id로 이미 follower가 있으면 noop.
#[tauri::command]
pub fn localdev_follow_log(
    profile_id: String,
    app: tauri::AppHandle,
    state: State<'_, LocalDevState>,
) -> Result<(), String> {
    let log_path = log_file_path(&app, &profile_id)?;

    {
        let mut followers = state.log_followers.lock().map_err(|e| e.to_string())?;
        if followers.contains_key(&profile_id) {
            return Ok(());
        }
        let stop = Arc::new(AtomicBool::new(false));
        followers.insert(profile_id.clone(), stop.clone());

        let app_thread = app.clone();
        let pid_thread = profile_id.clone();
        let stop_thread = stop;
        thread::spawn(move || {
            tail_log_loop(app_thread.clone(), pid_thread.clone(), log_path, stop_thread);
            // 자연 종료 시 스스로 followers map에서 제거 (정상 stop이면
            // localdev_stop_log_follow가 이미 제거했으니 noop)
            if let Some(state) = app_thread.try_state::<LocalDevState>() {
                if let Ok(mut f) = state.log_followers.lock() {
                    f.remove(&pid_thread);
                }
            }
        });
    }
    Ok(())
}

#[tauri::command]
pub fn localdev_stop_log_follow(
    profile_id: String,
    state: State<'_, LocalDevState>,
) -> Result<(), String> {
    let mut followers = state.log_followers.lock().map_err(|e| e.to_string())?;
    if let Some(stop) = followers.remove(&profile_id) {
        stop.store(true, Ordering::Relaxed);
    }
    Ok(())
}

/// 파일이 없으면 잠깐 기다렸다 재시도. EOF면 폴링.
/// stop flag가 true가 되면 즉시 종료.
fn tail_log_loop(
    app: tauri::AppHandle,
    profile_id: String,
    log_path: PathBuf,
    stop: Arc<AtomicBool>,
) {
    let mut reader: Option<BufReader<File>> = None;
    let mut buf = String::new();

    while !stop.load(Ordering::Relaxed) {
        if reader.is_none() {
            match File::open(&log_path) {
                Ok(f) => reader = Some(BufReader::new(f)),
                Err(_) => {
                    thread::sleep(Duration::from_millis(500));
                    continue;
                }
            }
        }
        let r = reader.as_mut().expect("reader present");
        buf.clear();
        match r.read_line(&mut buf) {
            Ok(0) => {
                thread::sleep(Duration::from_millis(300));
            }
            Ok(_) => {
                let line = buf.trim_end_matches(['\r', '\n']).to_string();
                let payload = LocaldevLogLineEvt {
                    run_id: "follow".to_string(),
                    profile_id: profile_id.clone(),
                    stream: "out".to_string(),
                    line,
                };
                let _ = app.emit("localdev-log", &payload);
            }
            Err(_) => {
                // 파일이 truncate(재시작)됐거나 IO 에러 — reader 버리고 재오픈
                reader = None;
                thread::sleep(Duration::from_millis(500));
            }
        }
    }
}

#[tauri::command]
pub fn localdev_start(
    profile_id: String,
    app: tauri::AppHandle,
    state: State<'_, LocalDevState>,
) -> Result<(), String> {
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
    let log_path = log_file_path(&app, &profile_id)?;
    let pid = spawn_detached_process(&profile.program, &profile.args, &cwd, &log_path)?;

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

#[tauri::command]
pub fn localdev_deploy(profile_id: String, state: State<'_, LocalDevState>) -> Result<String, String> {
    let repo_str = {
        let g = state.repo_root.lock().map_err(|e| e.to_string())?;
        g.clone()
            .ok_or_else(|| "저장소 루트를 먼저 설정하세요.".to_string())?
    };
    let repo = PathBuf::from(&repo_str);

    let profiles = read_profiles(&repo)?;
    let profile = profile_by_id(&profiles, &profile_id)?;
    let Some(ref da) = profile.deploy_args else {
        return Err("이 프로필에 deploy가 설정되어 있지 않습니다.".into());
    };

    let cwd = resolve_cwd(&repo, profile)?;
    run_npm_deploy_blocking(&cwd, da)
}

#[tauri::command]
pub async fn localdev_deploy_stream(
    profile_id: String,
    app: tauri::AppHandle,
    state: State<'_, LocalDevState>,
) -> Result<String, String> {
    let repo_str = {
        let g = state.repo_root.lock().map_err(|e| e.to_string())?;
        g.clone()
            .ok_or_else(|| "저장소 루트를 먼저 설정하세요.".to_string())?
    };
    let repo = PathBuf::from(&repo_str);

    let profiles = read_profiles(&repo)?;
    let profile = profile_by_id(&profiles, &profile_id)?;
    let Some(ref da) = profile.deploy_args else {
        return Err("이 프로필에 deploy가 설정되어 있지 않습니다.".into());
    };
    if da.is_empty() {
        return Err("deploy_args가 비어 있습니다.".into());
    }
    if !args_are_safe(da) {
        return Err("deploy 인자에 허용되지 않은 문자가 있습니다.".into());
    }
    let cwd = resolve_cwd(&repo, profile)?;

    {
        let mut busy = state.stream_busy.lock().map_err(|e| e.to_string())?;
        if !busy.insert(profile_id.clone()) {
            return Err("이 프로필에서 deploy 또는 npm install이 이미 실행 중입니다.".into());
        }
    }

    let cmd = npm_deploy_piped_command(&cwd, da);
    let app2 = app.clone();
    let pid = profile_id.clone();
    let join = tauri::async_runtime::spawn_blocking(move || {
        run_npm_command_streamed(app2, pid, "deploy", "deploy 완료", "deploy", cmd)
    });
    let out = match join.await {
        Ok(r) => r,
        Err(e) => {
            let mut busy = state.stream_busy.lock().map_err(|e2| e2.to_string())?;
            busy.remove(&profile_id);
            return Err(format!("deploy 작업 스레드 실패: {}", e));
        }
    };

    {
        let mut busy = state.stream_busy.lock().map_err(|e| e.to_string())?;
        busy.remove(&profile_id);
    }

    out
}

#[tauri::command]
pub async fn localdev_npm_install_stream(
    profile_id: String,
    app: tauri::AppHandle,
    state: State<'_, LocalDevState>,
) -> Result<String, String> {
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

    {
        let mut busy = state.stream_busy.lock().map_err(|e| e.to_string())?;
        if !busy.insert(profile_id.clone()) {
            return Err("이 프로필에서 deploy 또는 npm install이 이미 실행 중입니다.".into());
        }
    }

    let cmd = npm_install_piped_command(&cwd);
    let app2 = app.clone();
    let pid = profile_id.clone();
    let join = tauri::async_runtime::spawn_blocking(move || {
        run_npm_command_streamed(
            app2,
            pid,
            "npm_install",
            "npm install 완료",
            "npm install",
            cmd,
        )
    });
    let out = match join.await {
        Ok(r) => r,
        Err(e) => {
            let mut busy = state.stream_busy.lock().map_err(|e2| e2.to_string())?;
            busy.remove(&profile_id);
            return Err(format!("npm install 작업 스레드 실패: {}", e));
        }
    };

    {
        let mut busy = state.stream_busy.lock().map_err(|e| e.to_string())?;
        busy.remove(&profile_id);
    }

    out
}
