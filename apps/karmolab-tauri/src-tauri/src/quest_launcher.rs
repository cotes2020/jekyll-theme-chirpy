// quest_launcher.rs
// TASK launcher 위젯 (TASK-KL-025) — 외부 에디터 오픈 + 신규 TASK 생성.
//
// 명령 2개:
//   1. open_task_in_editor — file_path 화이트리스트 검증 후 OS 기본 에디터로 오픈
//   2. create_task — 도메인 + 제목 → 다음 ID 자동 발급 + frontmatter skeleton 생성

use crate::quest_index::DOMAIN_DIRS;
use crate::quest_writeback::validate_task_path;
use std::fs;
use std::path::PathBuf;

/// 사용자 홈 (USERPROFILE / HOME). quest_writeback 와 동일 정책.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

fn default_memo_path() -> Option<PathBuf> {
    home_dir().map(|h| h.join("repos").join("karmoddrine").join("memo"))
}

/// 위젯에서 TASK 파일 클릭 → OS 기본 에디터로 오픈.
#[tauri::command]
pub fn open_task_in_editor(file_path: String, memo_path: Option<String>) -> Result<(), String> {
    let memo = memo_path
        .map(PathBuf::from)
        .or_else(default_memo_path)
        .ok_or_else(|| "memo path could not be resolved".to_string())?;

    let canonical_file = validate_task_path(&file_path, &memo)?;
    open::that(canonical_file).map_err(|err| format!("open failed: {}", err))
}

/// 도메인 → TASK ID prefix 매핑. 6 도메인 화이트리스트.
fn domain_to_prefix(domain: &str) -> Option<&'static str> {
    match domain {
        "wm" => Some("WM"),
        "karmolab" => Some("KL"),
        "yawnbot" => Some("YB"),
        "life" => Some("LIFE"),
        "hobby" => Some("HOBBY"),
        "learning" => Some("LEARN"),
        _ => None,
    }
}

/// 도메인 → tasks 디렉토리 상대경로 (memo 기준).
fn domain_to_tasks_dir(domain: &str) -> Option<&'static str> {
    match domain {
        "wm" => Some("wm/tasks"),
        "karmolab" => Some("projects/karmolab/tasks"),
        "yawnbot" => Some("projects/yawnbot/tasks"),
        "life" => Some("life/tasks"),
        "hobby" => Some("hobby/tasks"),
        "learning" => Some("learning/tasks"),
        _ => None,
    }
}

/// 디렉토리 안의 TASK-{PREFIX}-{NNN} 패턴에서 max NNN + 1 반환.
/// 디렉토리 없거나 비었으면 1.
fn next_task_id_number(tasks_dir: &PathBuf, prefix: &str) -> u32 {
    let mut max_nnn: u32 = 0;
    let pattern_prefix = format!("TASK-{}-", prefix);

    if let Ok(entries) = fs::read_dir(tasks_dir) {
        for entry in entries.flatten() {
            let file_name = match entry.file_name().to_str() {
                Some(s) => s.to_string(),
                None => continue,
            };
            if !file_name.starts_with(&pattern_prefix) {
                continue;
            }
            // TASK-WM-007-foo.md → after_prefix = "007-foo.md"
            let after_prefix = &file_name[pattern_prefix.len()..];
            // 다음 `-` 또는 `.` 까지가 NNN 문자열
            let nnn_end = after_prefix
                .find(|c: char| c == '-' || c == '.')
                .unwrap_or(after_prefix.len());
            let nnn_str = &after_prefix[..nnn_end];
            if let Ok(n) = nnn_str.parse::<u32>() {
                if n > max_nnn {
                    max_nnn = n;
                }
            }
        }
    }

    max_nnn + 1
}

/// 사용자 입력 title 을 파일명 안전 형태로 정규화.
/// - 공백 → `-`
/// - 안전하지 않은 문자 (`/ \ : * ? " < > |`) 제거
/// - 양 끝 `-` trim, 연속 `-` → 단일
fn normalize_title_for_filename(title: &str) -> String {
    let unsafe_chars: &[char] = &['/', '\\', ':', '*', '?', '"', '<', '>', '|'];
    let mut result = String::new();
    let mut last_was_dash = false;
    for ch in title.trim().chars() {
        if unsafe_chars.contains(&ch) {
            continue;
        }
        if ch.is_whitespace() || ch == '-' {
            if !last_was_dash && !result.is_empty() {
                result.push('-');
                last_was_dash = true;
            }
        } else {
            result.push(ch);
            last_was_dash = false;
        }
    }
    while result.ends_with('-') {
        result.pop();
    }
    result
}

/// 위젯에서 "+ 새 TASK" → 다음 ID 자동 발급 + frontmatter skeleton 생성.
/// 반환: 생성된 파일 절대 경로 (위젯이 즉시 open_task_in_editor 호출).
#[tauri::command]
pub fn create_task(
    domain: String,
    title: String,
    memo_path: Option<String>,
) -> Result<String, String> {
    let memo = memo_path
        .map(PathBuf::from)
        .or_else(default_memo_path)
        .ok_or_else(|| "memo path could not be resolved".to_string())?;

    let prefix = domain_to_prefix(&domain).ok_or_else(|| format!("invalid domain: {}", domain))?;
    let tasks_rel = domain_to_tasks_dir(&domain).ok_or_else(|| "internal: domain mapping inconsistent".to_string())?;

    if title.trim().is_empty() {
        return Err("title empty".to_string());
    }
    let normalized_title = normalize_title_for_filename(&title);
    if normalized_title.is_empty() {
        return Err("title invalid after normalization".to_string());
    }

    let tasks_dir = memo.join(tasks_rel);
    fs::create_dir_all(&tasks_dir).map_err(|err| format!("mkdir failed: {}", err))?;

    let next_nnn = next_task_id_number(&tasks_dir, prefix);
    let task_id = format!("TASK-{}-{:03}", prefix, next_nnn);
    let file_name = format!("{}-{}.md", task_id, normalized_title);
    let file_path = tasks_dir.join(&file_name);

    if file_path.exists() {
        return Err(format!("task id collision: {}", file_path.display()));
    }

    // 의도적으로 DOMAIN_DIRS 검증은 prefix/dir 매핑이 일관함을 확인하는 sanity check.
    debug_assert!(DOMAIN_DIRS.iter().any(|d| *d == tasks_rel));

    let frontmatter = format!(
        "---\nid: {}\nstatus: seeded\npriority: normal\npath: [{}]\ntags: []\n---\n\n## 목표\n\n",
        task_id, domain
    );
    fs::write(&file_path, frontmatter).map_err(|err| format!("write failed: {}", err))?;

    let canonical = file_path
        .canonicalize()
        .map_err(|err| format!("canonicalize after write failed: {}", err))?;
    Ok(canonical.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_to_prefix_all_six() {
        assert_eq!(domain_to_prefix("wm"), Some("WM"));
        assert_eq!(domain_to_prefix("karmolab"), Some("KL"));
        assert_eq!(domain_to_prefix("yawnbot"), Some("YB"));
        assert_eq!(domain_to_prefix("life"), Some("LIFE"));
        assert_eq!(domain_to_prefix("hobby"), Some("HOBBY"));
        assert_eq!(domain_to_prefix("learning"), Some("LEARN"));
    }

    #[test]
    fn domain_to_prefix_invalid() {
        assert_eq!(domain_to_prefix("invalid"), None);
        assert_eq!(domain_to_prefix(""), None);
        assert_eq!(domain_to_prefix("WM"), None); // case-sensitive
    }

    #[test]
    fn normalize_title_basic() {
        assert_eq!(normalize_title_for_filename("hello world"), "hello-world");
        assert_eq!(normalize_title_for_filename("새  TASK  생성"), "새-TASK-생성");
    }

    #[test]
    fn normalize_title_strips_unsafe_chars() {
        assert_eq!(normalize_title_for_filename("test/foo:bar"), "testfoobar");
        assert_eq!(normalize_title_for_filename("a*b?c<d>"), "abcd");
    }

    #[test]
    fn normalize_title_trims_dashes() {
        assert_eq!(normalize_title_for_filename("  hello  "), "hello");
        assert_eq!(normalize_title_for_filename("---test---"), "test");
        assert_eq!(normalize_title_for_filename("a---b"), "a-b");
    }

    #[test]
    fn normalize_title_empty_or_only_unsafe() {
        assert_eq!(normalize_title_for_filename(""), "");
        assert_eq!(normalize_title_for_filename("   "), "");
        assert_eq!(normalize_title_for_filename("///"), "");
    }

    #[test]
    fn next_task_id_empty_dir_returns_one() {
        let temp = std::env::temp_dir().join("kl-025-test-empty");
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        assert_eq!(next_task_id_number(&temp, "WM"), 1);
        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn next_task_id_max_plus_one() {
        let temp = std::env::temp_dir().join("kl-025-test-max");
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("TASK-WM-001-foo.md"), "x").unwrap();
        fs::write(temp.join("TASK-WM-007-bar.md"), "x").unwrap();
        fs::write(temp.join("TASK-WM-003-baz.md"), "x").unwrap();
        assert_eq!(next_task_id_number(&temp, "WM"), 8);
        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn next_task_id_ignores_other_prefixes() {
        let temp = std::env::temp_dir().join("kl-025-test-other");
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("TASK-WM-005-x.md"), "x").unwrap();
        fs::write(temp.join("TASK-KL-099-y.md"), "x").unwrap();
        // WM 만 카운트 — KL 99 무시
        assert_eq!(next_task_id_number(&temp, "WM"), 6);
        // KL 카운트 — WM 무시
        assert_eq!(next_task_id_number(&temp, "KL"), 100);
        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn next_task_id_handles_sub_phase_letters() {
        // TASK-KL-009-A-foo.md, TASK-KL-009-B-bar.md → 모두 NNN=9. next = 10.
        let temp = std::env::temp_dir().join("kl-025-test-sub");
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("TASK-KL-009-A-foo.md"), "x").unwrap();
        fs::write(temp.join("TASK-KL-009-B-bar.md"), "x").unwrap();
        assert_eq!(next_task_id_number(&temp, "KL"), 10);
        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn next_task_id_handles_no_dash_after_nnn() {
        // TASK-KL-005.md (제목 없음) — `.` 이 NNN 끝.
        let temp = std::env::temp_dir().join("kl-025-test-nodash");
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("TASK-KL-005.md"), "x").unwrap();
        assert_eq!(next_task_id_number(&temp, "KL"), 6);
        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn next_task_id_skips_non_numeric() {
        let temp = std::env::temp_dir().join("kl-025-test-skip");
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("TASK-WM-foo-bar.md"), "x").unwrap(); // not a number
        fs::write(temp.join("TASK-WM-002-real.md"), "x").unwrap();
        assert_eq!(next_task_id_number(&temp, "WM"), 3);
        let _ = fs::remove_dir_all(&temp);
    }
}
