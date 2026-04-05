//! 저장소 루트 아래 파일만 열기·편집 (`.env` 등). `..`·절대 경로 차단.

use crate::local_dev::LocalDevState;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tauri::State;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const MAX_BYTES: u64 = 512 * 1024;

fn apply_no_window(cmd: &mut Command) {
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
}

fn repo_root_path(state: &LocalDevState) -> Result<PathBuf, String> {
    let g = state.repo_root.lock().map_err(|e| e.to_string())?;
    let s = g
        .as_ref()
        .ok_or_else(|| "저장소 루트를 먼저 설정하세요.".to_string())?;
    let p = Path::new(s);
    if !p.is_dir() {
        return Err("저장소 루트가 유효하지 않습니다.".into());
    }
    Ok(p.to_path_buf())
}

/// `rel_path`는 저장소 기준 POSIX 스타일 상대 경로만 허용.
pub fn resolve_in_repo(repo_root: &Path, rel_path: &str) -> Result<PathBuf, String> {
    let trimmed = rel_path.trim().replace('\\', "/");
    if trimmed.is_empty() {
        return Err("경로가 비어 있습니다.".into());
    }
    if trimmed.contains("..") {
        return Err("경로에 .. 를 사용할 수 없습니다.".into());
    }
    if Path::new(&trimmed).is_absolute() {
        return Err("상대 경로만 허용됩니다.".into());
    }
    let joined = repo_root.join(trimmed.trim_start_matches('/'));
    let canon_repo = fs::canonicalize(repo_root).map_err(|e| e.to_string())?;

    if joined.exists() {
        let c = fs::canonicalize(&joined).map_err(|e| e.to_string())?;
        if !c.starts_with(&canon_repo) {
            return Err("저장소 밖 경로입니다.".into());
        }
        return Ok(c);
    }

    let mut anc = joined.as_path();
    while !anc.exists() {
        anc = anc
            .parent()
            .ok_or_else(|| "유효한 부모 폴더가 없습니다.".to_string())?;
    }
    let canon_anc = fs::canonicalize(anc).map_err(|e| e.to_string())?;
    if !canon_anc.starts_with(&canon_repo) {
        return Err("저장소 밖 경로입니다.".into());
    }
    Ok(joined)
}

fn resolve_cmd(state: &LocalDevState, rel_path: &str) -> Result<PathBuf, String> {
    let root = repo_root_path(state)?;
    resolve_in_repo(&root, rel_path)
}

fn reveal_in_explorer(path: &Path) -> Result<(), String> {
    if path.is_file() {
        #[cfg(windows)]
        {
            let p = path.to_string_lossy().to_string();
            let mut c = Command::new("explorer.exe");
            c.arg(format!("/select,{}", p));
            apply_no_window(&mut c);
            c.spawn().map_err(|e| e.to_string())?;
            return Ok(());
        }
        #[cfg(target_os = "macos")]
        {
            Command::new("open")
                .args(["-R", &path.to_string_lossy()])
                .spawn()
                .map_err(|e| e.to_string())?;
            return Ok(());
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            let dir = path.parent().unwrap_or(path);
            open::that(dir).map_err(|e| e.to_string())?;
            return Ok(());
        }
    }
    let dir = if path.is_dir() {
        path
    } else {
        path.parent().ok_or_else(|| "폴더를 찾을 수 없습니다.".to_string())?
    };
    open::that(dir).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn repofile_open_default(rel_path: String, state: State<'_, LocalDevState>) -> Result<(), String> {
    let path = resolve_cmd(&*state, &rel_path)?;
    if !path.is_file() {
        return Err("파일이 없습니다. 먼저 편집에서 저장해 만들거나 템플릿을 복사하세요.".into());
    }
    open::that(&path).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn repofile_reveal(rel_path: String, state: State<'_, LocalDevState>) -> Result<(), String> {
    let path = resolve_cmd(&*state, &rel_path)?;
    reveal_in_explorer(&path)
}

#[tauri::command]
pub fn repofile_read(rel_path: String, state: State<'_, LocalDevState>) -> Result<String, String> {
    let path = resolve_cmd(&*state, &rel_path)?;
    if !path.is_file() {
        return Err("FILE_NOT_FOUND".into());
    }
    let meta = fs::metadata(&path).map_err(|e| e.to_string())?;
    if meta.len() > MAX_BYTES {
        return Err("파일이 너무 큽니다(512KB 한도).".into());
    }
    fs::read_to_string(&path).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn repofile_write(rel_path: String, content: String, state: State<'_, LocalDevState>) -> Result<(), String> {
    let path = resolve_cmd(&*state, &rel_path)?;
    if content.len() > MAX_BYTES as usize {
        return Err("내용이 너무 깁니다(512KB 한도).".into());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let name = path
        .file_name()
        .ok_or_else(|| "파일 이름이 없습니다.".to_string())?;
    let tmp = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{}.karmolabtmp", name.to_string_lossy()));
    fs::write(&tmp, content.as_bytes()).map_err(|e| e.to_string())?;
    fs::rename(&tmp, &path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        e.to_string()
    })?;
    Ok(())
}
