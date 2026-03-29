use tauri::webview::{NewWindowResponse, WebviewWindowBuilder};
use tauri::Url;

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let window_conf = app
                .config()
                .app
                .windows
                .iter()
                .find(|w| w.label == "main")
                .expect(r#"tauri.conf.json must include a window with label "main""#);

            WebviewWindowBuilder::from_config(app, window_conf)?
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

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
