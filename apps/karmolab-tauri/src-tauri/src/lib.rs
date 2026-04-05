mod local_dev;

use local_dev::{
    localdev_get_repo_root, localdev_list_tracked, localdev_npm_install, localdev_set_repo_root,
    localdev_start, localdev_stop, LocalDevState,
};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
#[cfg(windows)]
use tauri::tray::{MouseButton, TrayIconEvent};
use tauri::webview::{NewWindowResponse, WebviewWindowBuilder};
use tauri::Manager;
use tauri::Url;
use tauri::WindowEvent;
use tauri_plugin_updater::UpdaterExt;

#[cfg(windows)]
#[link(name = "user32")]
extern "system" {
    /// MB_ICONASTERISK — 토스트 무음일 때 청각 피드백용.
    fn MessageBeep(u_type: u32) -> i32;
}

const KARMOLAB_WEB_URL: &str = "https://mascari4615.github.io/karmolab/";

fn spawn_tray_update_check(handle: tauri::AppHandle) {
    tauri::async_runtime::spawn(async move {
        let msg = match handle.updater() {
            Ok(updater) => match updater.check().await {
                Ok(Some(update)) => {
                    let ver = update.version.clone();
                    match update
                        .download_and_install(|_chunk, _total| {}, || {})
                        .await
                    {
                        Ok(()) => format!(
                            "{} 설치됨. 앱을 완전히 종료한 뒤 다시 실행해 주세요.",
                            ver
                        ),
                        Err(e) => format!("업데이트 설치 실패: {}", e),
                    }
                }
                Ok(None) => "현재 버전이 최신입니다.".to_string(),
                Err(e) => format!("업데이트 확인 실패: {}", e),
            },
            Err(e) => format!("업데이터 초기화 실패: {}", e),
        };
        let _ = notify_rust::Notification::new()
            .summary("KarmoLab 업데이트")
            .body(&msg)
            .appname("KarmoLab")
            .show();
    });
}

/// 데스크톱 앱 플래그만 주입. `__karmolabSetNotifyInvokeDebug`는 예전 디버그 UI용 훅으로, 호출은 무해하게 무시.
fn karmolab_desktop_init_script() -> &'static str {
    r#"window.__KARMOLAB_DESKTOP__=!0;window.__karmolabSetNotifyInvokeDebug=function(){};"#
}

fn allow_in_webview(url: &Url) -> bool {
    match url.scheme() {
        "http" | "https" => {
            let Some(host) = url.host_str() else {
                return false;
            };
            if host == "mascari4615.github.io" {
                return true;
            }
            if cfg!(debug_assertions) && (host == "localhost" || host == "127.0.0.1") {
                return true;
            }
            false
        }
        "mailto" | "tel" | "sms" => false,
        _ => true,
    }
}

#[cfg(target_os = "windows")]
fn winrt_notification_sound_token(raw: &str) -> &'static str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "default" | "im" => "IM",
        "mail" => "Mail",
        "sms" => "SMS",
        "reminder" => "Reminder",
        "alarm" => "Alarm",
        "alarm2" => "Alarm2",
        "alarm3" => "Alarm3",
        "alarm4" => "Alarm4",
        "alarm5" => "Alarm5",
        "alarm6" => "Alarm6",
        "alarm7" => "Alarm7",
        "alarm8" => "Alarm8",
        "alarm9" => "Alarm9",
        "alarm10" => "Alarm10",
        "call" => "Call",
        "call2" => "Call2",
        "call3" => "Call3",
        "call4" => "Call4",
        "call5" => "Call5",
        "call6" => "Call6",
        "call7" => "Call7",
        "call8" => "Call8",
        "call9" => "Call9",
        "call10" => "Call10",
        _ => "Mail",
    }
}

#[tauri::command]
fn desktop_notify(
    title: String,
    body: String,
    sound: Option<String>,
    image_path: Option<String>,
) -> Result<(), String> {
    let summary = title.trim();
    if summary.is_empty() {
        return Ok(());
    }
    let text = body.trim();
    let body_line = if text.is_empty() { "KarmoLab" } else { text };

    #[cfg(windows)]
    let want_sound = match &sound {
        Some(s) => {
            let k = s.trim();
            !k.is_empty() && !k.eq_ignore_ascii_case("silent")
        }
        None => false,
    };

    let mut n = notify_rust::Notification::new();
    n.summary(summary).body(body_line).appname("KarmoLab");

    if let Some(ref path) = image_path {
        let t = path.trim();
        if !t.is_empty() {
            n.image_path(t);
        }
    }

    if let Some(ref s) = sound {
        let key = s.trim();
        if !key.is_empty() && !key.eq_ignore_ascii_case("silent") {
            #[cfg(target_os = "windows")]
            {
                // WinRT는 대소문자까지 맞아야 파싱됨; 실패 시 .ok() → 무음.
                // `Default`는 빈 <audio>라 무음이므로 IM으로 돌림.
                n.sound_name(winrt_notification_sound_token(key));
            }
            #[cfg(not(target_os = "windows"))]
            {
                n.sound_name(key);
            }
        }
    }

    n.show().map_err(|e| e.to_string())?;

    #[cfg(windows)]
    if want_sound {
        unsafe {
            let _ = MessageBeep(0x0000_0040);
        }
    }

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(LocalDevState::default())
        .invoke_handler(tauri::generate_handler![
            desktop_notify,
            localdev_set_repo_root,
            localdev_get_repo_root,
            localdev_list_tracked,
            localdev_start,
            localdev_stop,
            localdev_npm_install
        ])
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.show();
                let _ = w.set_focus();
            }
        }))
        .setup(|app| {
            let handle = app.handle().clone();
            let window_conf = app
                .config()
                .app
                .windows
                .iter()
                .find(|w| w.label == "main")
                .expect(r#"tauri.conf.json must include a window with label "main""#);

            let main_window = WebviewWindowBuilder::from_config(app, window_conf)?
                .initialization_script(karmolab_desktop_init_script())
                .on_navigation(|url| {
                    if allow_in_webview(url) {
                        true
                    } else {
                        if matches!(url.scheme(), "mailto" | "tel" | "sms" | "http" | "https") {
                            let _ = open::that(url.as_str());
                        }
                        false
                    }
                })
                .on_new_window(|url, _features| {
                    let _ = open::that(url.as_str());
                    NewWindowResponse::Deny
                })
                .build()?;

            main_window.on_window_event(move |event| {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    if let Some(w) = handle.get_webview_window("main") {
                        let _ = w.hide();
                    }
                }
            });

            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            {
                let show_i =
                    MenuItem::with_id(app, "tray_show", "KarmoLab 창 보이기", true, None::<&str>)?;
                let browser_i =
                    MenuItem::with_id(app, "tray_browser", "브라우저에서 열기", true, None::<&str>)?;
                let update_i =
                    MenuItem::with_id(app, "tray_update", "업데이트 확인…", true, None::<&str>)?;
                let quit_i = MenuItem::with_id(app, "tray_quit", "종료", true, None::<&str>)?;
                let menu = Menu::with_items(app, &[&show_i, &browser_i, &update_i, &quit_i])?;

                if let Some(icon) = app.default_window_icon().cloned() {
                    let _ = TrayIconBuilder::new()
                        .icon(icon)
                        .menu(&menu)
                        .tooltip("KarmoLab — 트레이 메뉴에서 업데이트 확인 · 닫기(X)는 숨김")
                        .show_menu_on_left_click(true)
                        .on_menu_event(|app, event| {
                            if event.id == "tray_show" {
                                if let Some(w) = app.get_webview_window("main") {
                                    let _ = w.unminimize();
                                    let _ = w.show();
                                    let _ = w.set_focus();
                                }
                            } else if event.id == "tray_browser" {
                                let _ = open::that(KARMOLAB_WEB_URL);
                            } else if event.id == "tray_update" {
                                spawn_tray_update_check(app.clone());
                            } else if event.id == "tray_quit" {
                                app.exit(0);
                            }
                        })
                        .on_tray_icon_event(|tray, event| {
                            // Windows only: double-click tray to raise the window without opening the menu.
                            #[cfg(windows)]
                            if let TrayIconEvent::DoubleClick {
                                button: MouseButton::Left,
                                ..
                            } = event
                            {
                                let app = tray.app_handle();
                                if let Some(w) = app.get_webview_window("main") {
                                    let _ = w.unminimize();
                                    let _ = w.show();
                                    let _ = w.set_focus();
                                }
                            }
                            #[cfg(not(windows))]
                            let _ = (tray, event);
                        })
                        .build(app)?;
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
