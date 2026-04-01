use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
#[cfg(windows)]
use tauri::tray::{MouseButton, TrayIconEvent};
use tauri::{Manager, PhysicalPosition, PhysicalSize, RunEvent, WindowEvent};

#[derive(Debug, Serialize, Deserialize)]
struct WindowState {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

fn overlay_state_dir(app: &tauri::AppHandle) -> Option<PathBuf> {
    let mut dir = app.path().app_data_dir().ok()?;
    dir.push("chat-overlay");
    Some(dir)
}

fn state_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    Some(overlay_state_dir(app)?.join("window-state.json"))
}

fn load_window_state(app: &tauri::AppHandle) -> Option<WindowState> {
    let path = state_path(app)?;
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

fn persist_window_state(
    pos: tauri::PhysicalPosition<i32>,
    size: tauri::PhysicalSize<u32>,
    app: &tauri::AppHandle,
) {
    let st = WindowState {
        x: pos.x,
        y: pos.y,
        width: size.width,
        height: size.height,
    };
    let Some(path) = state_path(app) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(bytes) = serde_json::to_string(&st) {
        let _ = std::fs::write(path, bytes);
    }
}

fn save_window_state(window: &tauri::Window, app: &tauri::AppHandle) {
    let (Ok(pos), Ok(size)) = (window.outer_position(), window.outer_size()) else {
        return;
    };
    persist_window_state(pos, size, app);
}

fn apply_window_state(window: &tauri::WebviewWindow, app: &tauri::AppHandle) {
    if let Some(st) = load_window_state(app) {
        let _ = window.set_position(PhysicalPosition::new(st.x, st.y));
        let _ = window.set_size(PhysicalSize::new(st.width, st.height));
    }
}

fn save_window_state_webview(window: &tauri::WebviewWindow, app: &tauri::AppHandle) {
    let (Ok(pos), Ok(size)) = (window.outer_position(), window.outer_size()) else {
        return;
    };
    persist_window_state(pos, size, app);
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let ignore_mouse = Arc::new(AtomicBool::new(true));

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.show();
                let _ = w.set_focus();
            }
        }))
        .setup({
            let ignore_mouse = ignore_mouse.clone();
            move |app| {
                let handle = app.handle().clone();
                let window = app
                    .get_webview_window("main")
                    .expect("main webview window must exist");
                apply_window_state(&window, &handle);
                let _ = window.set_always_on_top(true);
                let _ = window.set_ignore_cursor_events(ignore_mouse.load(Ordering::SeqCst));

                #[cfg(not(any(target_os = "android", target_os = "ios")))]
                {
                    let show_i = MenuItem::with_id(app, "tray_show", "창 보이기", true, None::<&str>)?;
                    let hide_i = MenuItem::with_id(app, "tray_hide", "창 숨기기", true, None::<&str>)?;
                    let toggle_i = MenuItem::with_id(
                        app,
                        "tray_toggle_ct",
                        "클릭 통과 전환",
                        true,
                        None::<&str>,
                    )?;
                    let quit_i = MenuItem::with_id(app, "tray_quit", "종료", true, None::<&str>)?;
                    let menu = Menu::with_items(app, &[&show_i, &hide_i, &toggle_i, &quit_i])?;

                    let ig = ignore_mouse.clone();
                    if let Some(icon) = app.default_window_icon().cloned() {
                        let _ = TrayIconBuilder::new()
                            .icon(icon)
                            .menu(&menu)
                            .tooltip("chat-overlay — 트레이에서 종료·클릭 통과 전환")
                            .show_menu_on_left_click(true)
                            .on_menu_event(move |app, event| {
                                if event.id == "tray_show" {
                                    if let Some(w) = app.get_webview_window("main") {
                                        let _ = w.show();
                                        let _ = w.set_focus();
                                    }
                                } else if event.id == "tray_hide" {
                                    if let Some(w) = app.get_webview_window("main") {
                                        let _ = w.hide();
                                    }
                                } else if event.id == "tray_toggle_ct" {
                                    let v = !ig.load(Ordering::SeqCst);
                                    ig.store(v, Ordering::SeqCst);
                                    if let Some(w) = app.get_webview_window("main") {
                                        let _ = w.set_ignore_cursor_events(v);
                                    }
                                } else if event.id == "tray_quit" {
                                    if let Some(w) = app.get_webview_window("main") {
                                        save_window_state_webview(&w, app);
                                    }
                                    app.exit(0);
                                }
                            })
                            .on_tray_icon_event(|tray, event| {
                                #[cfg(windows)]
                                if let TrayIconEvent::DoubleClick {
                                    button: MouseButton::Left,
                                    ..
                                } = event
                                {
                                    let app = tray.app_handle();
                                    if let Some(w) = app.get_webview_window("main") {
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
            }
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                save_window_state(window, &window.app_handle());
                let _ = window.hide();
            }
        })
        .build(tauri::generate_context!())
        .expect("error building chat-overlay")
        .run(|app_handle, event| {
            if let RunEvent::Exit = event {
                if let Some(w) = app_handle.get_webview_window("main") {
                    save_window_state_webview(&w, app_handle);
                }
            }
        });
}
