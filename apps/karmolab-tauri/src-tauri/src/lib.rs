use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, TrayIconBuilder, TrayIconEvent};
use tauri::webview::{NewWindowResponse, WebviewWindowBuilder};
use tauri::Manager;
use tauri::Url;
use tauri::WindowEvent;

const KARMOLAB_WEB_URL: &str = "https://mascari4615.github.io/karmolab/";

/// `window.__KARMOLAB_DESKTOP__` + WebView가 오래된 `toolbox.js`를 캐시해도 디버그 위젯이 빠지지 않도록
/// `apps/karmolab/js/widgets/devtools.js`를 바이너리에 넣어 `Toolbox.init` 직전에 실행합니다.
fn karmolab_desktop_init_script() -> String {
    fn escape_js_string_literal(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 16);
        out.push('"');
        for c in s.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\0' => out.push_str("\\0"),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    let devtools = include_str!("../../../../apps/karmolab/js/widgets/devtools.js");
    let lit = escape_js_string_literal(devtools);
    format!(
        r#"window.__KARMOLAB_DESKTOP__=!0;(function(){{function k(){{if(typeof Toolbox==='undefined'||!Toolbox.init||Toolbox.__karmolabDevtoolsInjected)return;Toolbox.__karmolabDevtoolsInjected=1;var i=Toolbox.init;Toolbox.init=function(){{try{{if(typeof Toolbox.getTools==='function'&&!Toolbox.getTools().some(function(t){{return t.id==='devtools'}})){{var n=document.createElement('script');n.textContent={lit};document.documentElement.appendChild(n);n.remove();}}}}catch(e){{}}return i.apply(this,arguments)}}}}document.addEventListener('DOMContentLoaded',k);k();setTimeout(k,0)}})();"#,
        lit = lit
    )
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

#[tauri::command]
fn desktop_notify(title: String, body: String) -> Result<(), String> {
    let summary = title.trim();
    if summary.is_empty() {
        return Ok(());
    }
    let text = body.trim();
    let body_line = if text.is_empty() { "KarmoLab" } else { text };
    notify_rust::Notification::new()
        .summary(summary)
        .body(body_line)
        .appname("KarmoLab")
        .show()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![desktop_notify])
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
                let quit_i = MenuItem::with_id(app, "tray_quit", "종료", true, None::<&str>)?;
                let menu = Menu::with_items(app, &[&show_i, &browser_i, &quit_i])?;

                if let Some(icon) = app.default_window_icon().cloned() {
                    let _ = TrayIconBuilder::new()
                        .icon(icon)
                        .menu(&menu)
                        .tooltip("KarmoLab — 닫기는 트레이로 숨김 · 왼쪽 클릭: 메뉴 · Windows: 더블클릭: 창")
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
                        })
                        .build(app)?;
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
