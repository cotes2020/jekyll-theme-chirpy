/// PC 활동 트래커 — 포그라운드 윈도우/프로세스를 일정 주기로 샘플링해 일별 JSONL로 저장.
///
/// 데이터는 모두 로컬 (`{app_data_dir}/activity/YYYY-MM-DD.jsonl`). 외부 전송 0.
/// 사용자가 일정 시간 입력이 없으면(idle) 샘플 라벨에 "idle"로 표시한다.
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Manager, Runtime};

const SAMPLE_INTERVAL_SECS: u64 = 5;
const IDLE_THRESHOLD_SECS: u64 = 60;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivitySample {
    /// Unix epoch seconds (UTC).
    pub ts: u64,
    /// 프로세스 실행 파일명 (예: "Code.exe"). idle인 경우 빈 문자열.
    pub process: String,
    /// 포그라운드 윈도우 타이틀. idle인 경우 빈 문자열.
    pub title: String,
    /// 마지막 입력으로부터 IDLE_THRESHOLD_SECS 이상 지났으면 true.
    pub idle: bool,
}

#[cfg(windows)]
mod platform {
    use super::ActivitySample;
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use std::time::{SystemTime, UNIX_EPOCH};

    use winapi::shared::minwindef::{DWORD, FALSE};
    use winapi::um::handleapi::CloseHandle;
    use winapi::um::processthreadsapi::OpenProcess;
    use winapi::um::psapi::GetModuleBaseNameW;
    use winapi::um::winnt::{PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ};
    use winapi::um::winuser::{
        GetForegroundWindow, GetLastInputInfo, GetWindowTextLengthW, GetWindowTextW,
        GetWindowThreadProcessId, LASTINPUTINFO,
    };

    pub fn sample(idle_threshold_secs: u64) -> ActivitySample {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let idle = idle_seconds().map(|s| s >= idle_threshold_secs).unwrap_or(false);
        if idle {
            return ActivitySample {
                ts,
                process: String::new(),
                title: String::new(),
                idle: true,
            };
        }

        let (process, title) = unsafe { foreground_info() };
        ActivitySample {
            ts,
            process,
            title,
            idle: false,
        }
    }

    fn idle_seconds() -> Option<u64> {
        unsafe {
            let mut info: LASTINPUTINFO = std::mem::zeroed();
            info.cbSize = std::mem::size_of::<LASTINPUTINFO>() as u32;
            if GetLastInputInfo(&mut info) == 0 {
                return None;
            }
            // GetTickCount는 wraparound이지만 단기 비교엔 충분.
            let now_ms = winapi::um::sysinfoapi::GetTickCount() as u64;
            let last_ms = info.dwTime as u64;
            Some(now_ms.saturating_sub(last_ms) / 1000)
        }
    }

    unsafe fn foreground_info() -> (String, String) {
        let hwnd = GetForegroundWindow();
        if hwnd.is_null() {
            return (String::new(), String::new());
        }

        // 윈도우 타이틀
        let title_len = GetWindowTextLengthW(hwnd);
        let title = if title_len > 0 {
            let cap = (title_len as usize) + 1;
            let mut buf: Vec<u16> = vec![0; cap];
            let copied = GetWindowTextW(hwnd, buf.as_mut_ptr(), cap as i32);
            if copied > 0 {
                OsString::from_wide(&buf[..copied as usize])
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // 프로세스 PID + 실행 파일명
        let mut pid: DWORD = 0;
        GetWindowThreadProcessId(hwnd, &mut pid);
        let process = process_name(pid).unwrap_or_default();

        (process, title)
    }

    unsafe fn process_name(pid: DWORD) -> Option<String> {
        if pid == 0 {
            return None;
        }
        let handle = OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
            FALSE,
            pid,
        );
        if handle.is_null() {
            return None;
        }
        let mut buf: [u16; 260] = [0; 260];
        let len = GetModuleBaseNameW(handle, std::ptr::null_mut(), buf.as_mut_ptr(), buf.len() as u32);
        CloseHandle(handle);
        if len == 0 {
            return None;
        }
        Some(
            OsString::from_wide(&buf[..len as usize])
                .to_string_lossy()
                .into_owned(),
        )
    }
}

#[cfg(not(windows))]
mod platform {
    use super::ActivitySample;
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn sample(_idle_threshold_secs: u64) -> ActivitySample {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        ActivitySample {
            ts,
            process: String::from("(unsupported)"),
            title: String::new(),
            idle: false,
        }
    }
}

pub struct ActivityState {
    running: Arc<AtomicBool>,
    base_dir: PathBuf,
}

impl ActivityState {
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            base_dir,
        }
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// 백그라운드 폴링 시작. 이미 돌고 있으면 무시.
    pub fn start(&self) {
        if self
            .running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let running = self.running.clone();
        let base_dir = self.base_dir.clone();
        std::thread::spawn(move || {
            while running.load(Ordering::Acquire) {
                let sample = platform::sample(IDLE_THRESHOLD_SECS);
                if let Err(e) = append_sample(&base_dir, &sample) {
                    eprintln!("[activity] write failed: {}", e);
                }
                std::thread::sleep(Duration::from_secs(SAMPLE_INTERVAL_SECS));
            }
        });
    }

    /// 현재 진입점에선 호출처가 없지만 종료 훅에서 깔끔하게 멈추기 위해 남겨둠.
    #[allow(dead_code)]
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }
}

fn day_path(base_dir: &Path, ts: u64) -> PathBuf {
    let day = ts_to_day_utc(ts);
    base_dir.join(format!("{}.jsonl", day))
}

fn ts_to_day_utc(ts: u64) -> String {
    // UTC 일자 (YYYY-MM-DD). 사용자 표시 시점에서 KST 변환은 위젯 쪽에서 처리.
    let secs_per_day: u64 = 86400;
    let days = ts / secs_per_day;
    // 1970-01-01부터 days일 후의 그레고리력 날짜 계산.
    let (y, m, d) = days_to_ymd(days as i64);
    format!("{:04}-{:02}-{:02}", y, m, d)
}

/// Howard Hinnant의 days_from_civil 역변환. 1970-01-01 = day 0.
fn days_to_ymd(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn append_sample(base_dir: &Path, sample: &ActivitySample) -> std::io::Result<()> {
    create_dir_all(base_dir)?;
    let path = day_path(base_dir, sample.ts);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(sample).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    })?;
    file.write_all(line.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

/// 단일 일자(JSONL) 로드. 파일 없으면 빈 Vec.
fn load_day(base_dir: &Path, day: &str) -> std::io::Result<Vec<ActivitySample>> {
    let path = base_dir.join(format!("{}.jsonl", day));
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(s) = serde_json::from_str::<ActivitySample>(&line) {
            samples.push(s);
        }
    }
    Ok(samples)
}

#[derive(Debug, Serialize)]
pub struct DayActivity {
    pub day: String,
    pub samples: Vec<ActivitySample>,
}

#[tauri::command]
pub fn activity_query_day<R: Runtime>(
    app: AppHandle<R>,
    day: String,
) -> Result<DayActivity, String> {
    let state = app.state::<ActivityState>();
    let samples = load_day(state.base_dir(), &day).map_err(|e| e.to_string())?;
    Ok(DayActivity { day, samples })
}

#[tauri::command]
pub fn activity_status<R: Runtime>(app: AppHandle<R>) -> Result<serde_json::Value, String> {
    let state = app.state::<ActivityState>();
    Ok(serde_json::json!({
        "base_dir": state.base_dir().to_string_lossy(),
        "sample_interval_secs": SAMPLE_INTERVAL_SECS,
        "idle_threshold_secs": IDLE_THRESHOLD_SECS,
    }))
}

/// 저장된 일자(YYYY-MM-DD, UTC 기준 파일명) 목록 — 정렬된 채로.
/// 위젯의 "전체" 기간 모드가 사용. 디렉토리가 없거나 비었으면 빈 배열.
#[tauri::command]
pub fn activity_list_days<R: Runtime>(app: AppHandle<R>) -> Result<Vec<String>, String> {
    let state = app.state::<ActivityState>();
    let dir = state.base_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut days = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| e.to_string())?;
    for entry in entries.flatten() {
        let name_os = entry.file_name();
        let Some(name) = name_os.to_str() else { continue };
        if let Some(stem) = name.strip_suffix(".jsonl") {
            days.push(stem.to_string());
        }
    }
    days.sort();
    Ok(days)
}
