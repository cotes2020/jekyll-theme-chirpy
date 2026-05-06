// quest_watcher.rs
// QuestLog 위젯용 — memo TASK 디렉토리 6개 파일 변경 감시.
//
// 동작:
//   1. 앱 시작 시 단일 watcher 생성 (notify-rs)
//   2. memo/{wm/tasks, projects/karmolab/tasks, projects/yawnbot/tasks, life/tasks, hobby/tasks, learning/tasks}
//      6개 디렉토리 각각 NonRecursive 등록
//   3. fs event 받으면 debounce (300ms) — 외부 에디터 atomic save 의 다중 이벤트 묶음
//   4. debounce 만료 시 Tauri emit("quest-tree-changed") — 위젯이 받아서 fetchMemoTree 재호출
//
// 라이프사이클: app lifetime 동안 살아있음. setup() 에서 Box::leak (Tauri 일반 패턴).

use crate::quest_index::DOMAIN_DIRS;
use notify::{recommended_watcher, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

const DEBOUNCE_MS: u64 = 300;
const POLL_TIMEOUT_MS: u64 = 100;

/// 사용자 홈 (USERPROFILE / HOME). quest_index 와 동일 정책.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

fn default_memo_path() -> Option<PathBuf> {
    home_dir().map(|h| h.join("repos").join("karmoddrine").join("memo"))
}

/// 앱 시작 시 호출. memo TASK 디렉토리 6개를 감시하고 변경 시 'quest-tree-changed' 이벤트 emit.
/// 실패 (memo 없음, 디렉토리 일부 없음 등) 는 panic 안 함 — 로그 후 skip.
/// watcher 는 Box::leak 으로 app lifetime 동안 유지 (process 종료 시 같이 해제).
pub fn start(app_handle: AppHandle) {
    let memo_path = match default_memo_path() {
        Some(path) => path,
        None => {
            eprintln!("[quest_watcher] home dir resolve 실패 — watcher 시작 안 함");
            return;
        }
    };

    if !memo_path.exists() {
        eprintln!(
            "[quest_watcher] memo path 없음 ({}) — watcher 시작 안 함",
            memo_path.display()
        );
        return;
    }

    let (event_sender, event_receiver) = channel();
    let mut watcher = match recommended_watcher(event_sender) {
        Ok(w) => w,
        Err(err) => {
            eprintln!("[quest_watcher] watcher 생성 실패: {}", err);
            return;
        }
    };

    let mut watched_count = 0usize;
    for domain in DOMAIN_DIRS {
        let dir = memo_path.join(domain);
        if !dir.exists() {
            continue;
        }
        match watcher.watch(&dir, RecursiveMode::NonRecursive) {
            Ok(()) => {
                watched_count += 1;
            }
            Err(err) => {
                eprintln!("[quest_watcher] {} watch 실패: {}", dir.display(), err);
            }
        }
    }

    if watched_count == 0 {
        eprintln!("[quest_watcher] watch 가능한 디렉토리 0 — watcher 시작 안 함");
        return;
    }

    eprintln!(
        "[quest_watcher] {} 도메인 감시 시작 (memo: {})",
        watched_count,
        memo_path.display()
    );

    // Debounce thread — fs event 받으면 last_event 시각 갱신, 마지막 이벤트 후 DEBOUNCE_MS 지나면 emit.
    std::thread::spawn(move || {
        let mut pending_since: Option<Instant> = None;
        let debounce = Duration::from_millis(DEBOUNCE_MS);
        let poll_timeout = Duration::from_millis(POLL_TIMEOUT_MS);

        loop {
            match event_receiver.recv_timeout(poll_timeout) {
                Ok(Ok(_event)) => {
                    pending_since = Some(Instant::now());
                }
                Ok(Err(err)) => {
                    eprintln!("[quest_watcher] fs event 에러: {}", err);
                }
                Err(RecvTimeoutError::Timeout) => {
                    if let Some(since) = pending_since {
                        if since.elapsed() >= debounce {
                            if let Err(err) = app_handle.emit("quest-tree-changed", ()) {
                                eprintln!("[quest_watcher] emit 실패: {}", err);
                            }
                            pending_since = None;
                        }
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    eprintln!("[quest_watcher] channel disconnect — debounce thread 종료");
                    break;
                }
            }
        }
    });

    // Watcher 가 drop 되면 thread 가 disconnect 받음. App lifetime 동안 살리려고 leak.
    Box::leak(Box::new(watcher));
}
