//! 브라우저 확장(KarmoWebExtension 등)이 `POST /ingest` 로 보낸 채팅을 Webview로 넘김.

use axum::extract::State;
use axum::http::{Method, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use std::net::SocketAddr;

/// `CHAT_OVERLAY_INGEST_PORT` (기본 17376). 확장 `DEFAULT_INGEST_URL` 과 맞출 것.
pub fn ingest_port() -> u16 {
    std::env::var("CHAT_OVERLAY_INGEST_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(17376)
}
use tauri::{AppHandle, Emitter};
use tower_http::cors::{Any, CorsLayer};

#[derive(Deserialize)]
pub struct IngestBody {
    pub author: String,
    pub text: String,
    #[serde(default)]
    pub ts: Option<i64>,
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

async fn ingest_handler(State(app): State<AppHandle>, Json(body): Json<IngestBody>) -> StatusCode {
    let ts = body.ts.unwrap_or_else(now_ms);
    if let Err(e) = app.emit(
        "extension-ingest",
        json!({
            "author": body.author,
            "text": body.text,
            "ts": ts,
        }),
    ) {
        eprintln!("[ingest] emit 실패: {e}");
        return StatusCode::INTERNAL_SERVER_ERROR;
    }
    StatusCode::NO_CONTENT
}

pub fn spawn_ingest_server(app: AppHandle) {
    std::thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[ingest] tokio runtime 생성 실패: {e}");
                return;
            }
        };
        rt.block_on(async move {
            let port = ingest_port();
            let router = Router::new()
                .route("/ingest", post(ingest_handler))
                .layer(
                    CorsLayer::new()
                        .allow_origin(Any)
                        .allow_methods([Method::POST, Method::OPTIONS])
                        .allow_headers(Any),
                )
                .with_state(app);
            let addr = SocketAddr::from(([127, 0, 0, 1], port));
            let listener = match tokio::net::TcpListener::bind(addr).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!(
                        "[ingest] 127.0.0.1:{port} 바인드 실패 (CHAT_OVERLAY_INGEST_PORT·다른 프로세스 확인): {e}"
                    );
                    return;
                }
            };
            eprintln!("[ingest] http://127.0.0.1:{port}/ingest 대기 중 (KarmoWebExtension)");
            if let Err(e) = axum::serve(listener, router).await {
                eprintln!("[ingest] 서버 오류: {e}");
            }
        });
    });
}
