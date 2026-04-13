//! 저장된 선택에 따라 메인 WebView를 배포 URL 또는 로컬 dev URL로 맞춤.

use serde::{Deserialize, Serialize};
use std::fs;
use tauri::{AppHandle, Manager, Url, WebviewWindow};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum KarmolabOrigin {
    #[default]
    Remote,
    Local,
}

impl KarmolabOrigin {
    pub fn url(self) -> &'static str {
        match self {
            Self::Remote => "https://mascari4615.github.io/karmolab/",
            Self::Local => "http://127.0.0.1:8899/apps/karmolab/",
        }
    }
}

#[derive(Serialize, Deserialize)]
struct OriginPrefFile {
    origin: KarmolabOrigin,
}

fn pref_path(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    app.path()
        .app_config_dir()
        .map_err(|e| e.to_string())
        .map(|p| p.join("karmolab-origin.json"))
}

/// 저장된 선택만 읽습니다. 파일이 없으면 `None`(→ 첫 로드는 `tauri.conf` 그대로).
fn load_saved_origin(app: &AppHandle) -> Option<KarmolabOrigin> {
    let path = pref_path(app).ok()?;
    let raw = fs::read_to_string(&path).ok()?;
    serde_json::from_str::<OriginPrefFile>(&raw).ok().map(|f| f.origin)
}

fn save(app: &AppHandle, origin: KarmolabOrigin) -> Result<(), String> {
    let path = pref_path(app)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let j = serde_json::to_string_pretty(&OriginPrefFile { origin }).map_err(|e| e.to_string())?;
    fs::write(&path, j).map_err(|e| e.to_string())
}

fn same_document_url(a: &str, b: &str) -> bool {
    let norm = |s: &str| s.trim_end_matches('/').to_lowercase();
    norm(a) == norm(b)
}

/// `karmolab-origin.json`이 있을 때만, 그 선택과 현재 URL이 다르면 `navigate`합니다.
pub fn apply_if_needed(app: &AppHandle, window: &WebviewWindow) -> Result<(), String> {
    let Some(origin) = load_saved_origin(app) else {
        return Ok(());
    };
    let target = Url::parse(origin.url()).map_err(|e| e.to_string())?;
    let current = window.url().map_err(|e| e.to_string())?;
    if same_document_url(current.as_str(), target.as_str()) {
        return Ok(());
    }
    window.navigate(target).map_err(|e| e.to_string())
}

pub fn persist_and_navigate(app: &AppHandle, origin: KarmolabOrigin) {
    let Some(w) = app.get_webview_window("main") else {
        return;
    };
    let target = match Url::parse(origin.url()) {
        Ok(u) => u,
        Err(_) => return,
    };
    if let Err(e) = w.navigate(target) {
        let msg = format!("주소 전환 실패: {}", e);
        let _ = notify_rust::Notification::new()
            .summary("KarmoLab")
            .body(&msg)
            .appname("KarmoLab")
            .show();
        return;
    }
    if let Err(e) = save(app, origin) {
        let msg = format!("설정 저장 실패: {}", e);
        let _ = notify_rust::Notification::new()
            .summary("KarmoLab")
            .body(&msg)
            .appname("KarmoLab")
            .show();
        return;
    }
    let body = match origin {
        KarmolabOrigin::Remote => "배포 주소(GitHub)로 열었습니다.",
        KarmolabOrigin::Local => "로컬 주소(8899)로 열었습니다.",
    };
    let _ = notify_rust::Notification::new()
        .summary("KarmoLab")
        .body(body)
        .appname("KarmoLab")
        .show();
}
