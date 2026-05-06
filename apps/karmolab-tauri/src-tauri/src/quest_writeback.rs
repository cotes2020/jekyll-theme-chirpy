// quest_writeback.rs
// QuestLog 위젯 v2 — 체크박스 토글 write-back.
//
// 위젯에서 체크박스 클릭 → invoke('toggle_quest_check', { filePath, lineNumber, expectedText, memoPath })
// → 파일 경로 화이트리스트 검증 → 라인 텍스트 검증 → `- [ ]` ↔ `- [x]` 토글 → 파일 write.
//
// 보안:
//   - filePath 는 무조건 {memoPath}/(wm|projects/karmolab|projects/yawnbot|life|hobby|learning)/tasks/TASK-*.md
//   - 위젯이 임의 파일 쓰기 못하게 차단
//
// 정합성 (race-safe):
//   - expectedText 와 실제 라인 텍스트가 다르면 Err("text mismatch") — 위젯에 강제 새로고침 trigger

use crate::quest_index::DOMAIN_DIRS;
use std::fs;
use std::path::{Path, PathBuf};

/// 사용자 홈 디렉토리. quest_index 와 동일 정책.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

fn default_memo_path() -> Option<PathBuf> {
    home_dir().map(|h| h.join("repos").join("karmoddrine").join("memo"))
}

/// file_path 가 허용된 memo 도메인 안의 TASK-*.md 인지 검증. canonicalize 후 PathBuf 반환.
fn validate_task_path(file_path: &str, memo_path: &Path) -> Result<PathBuf, String> {
    let canonical_file = PathBuf::from(file_path)
        .canonicalize()
        .map_err(|e| format!("file not found: {}", e))?;
    let canonical_memo = memo_path
        .canonicalize()
        .map_err(|e| format!("memo path invalid: {}", e))?;

    let rel = canonical_file
        .strip_prefix(&canonical_memo)
        .map_err(|_| "file outside memo".to_string())?;
    let rel_str = rel.to_string_lossy().replace('\\', "/");

    let domain_ok = DOMAIN_DIRS
        .iter()
        .any(|domain| rel_str.starts_with(&format!("{}/", domain)));
    if !domain_ok {
        return Err(format!("file not in allowed domain: {}", rel_str));
    }

    let file_name = canonical_file
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "invalid file name".to_string())?;
    if !file_name.starts_with("TASK-") || !file_name.ends_with(".md") {
        return Err(format!("filename must match TASK-*.md: {}", file_name));
    }

    Ok(canonical_file)
}

/// 단일 라인을 토글한 새 콘텐츠 생성. 라인 엔딩 (CRLF/LF) 보존.
/// 반환: (new_content, new_done_state)
fn toggle_line(content: &str, line_number: u32, expected_text: &str) -> Result<(String, bool), String> {
    if line_number == 0 {
        return Err("line_number must be 1-base".to_string());
    }

    let mut lines: Vec<String> = content.split('\n').map(|line| line.to_string()).collect();
    let line_idx = (line_number - 1) as usize;
    if line_idx >= lines.len() {
        return Err(format!(
            "line out of range: {} (file has {} lines)",
            line_number,
            lines.len()
        ));
    }

    let raw_line = lines[line_idx].clone();
    let has_cr = raw_line.ends_with('\r');
    let line_no_cr = if has_cr {
        &raw_line[..raw_line.len() - 1]
    } else {
        &raw_line[..]
    };
    let trimmed = line_no_cr.trim_start();
    let leading_ws_len = line_no_cr.len() - trimmed.len();
    let leading_ws = &line_no_cr[..leading_ws_len];

    let (rest_after_box, current_done, new_marker) = if let Some(rest) = trimmed.strip_prefix("- [ ]") {
        (rest, false, "- [x]")
    } else if let Some(rest) = trimmed.strip_prefix("- [x]") {
        (rest, true, "- [ ]")
    } else if let Some(rest) = trimmed.strip_prefix("- [X]") {
        (rest, true, "- [ ]")
    } else {
        return Err("not a checkbox line".to_string());
    };

    let actual_text = rest_after_box.trim();
    if actual_text != expected_text.trim() {
        return Err(format!(
            "text mismatch — expected '{}', got '{}'",
            expected_text.trim(),
            actual_text
        ));
    }

    let cr_suffix = if has_cr { "\r" } else { "" };
    let new_line = format!("{}{}{}{}", leading_ws, new_marker, rest_after_box, cr_suffix);
    lines[line_idx] = new_line;

    Ok((lines.join("\n"), !current_done))
}

/// QuestLog 위젯에서 호출. 라인 번호 + 텍스트 검증 후 토글하여 파일 write.
/// 반환: 새로운 done 상태 (true = checked).
#[tauri::command]
pub fn toggle_quest_check(
    file_path: String,
    line_number: u32,
    expected_text: String,
    memo_path: Option<String>,
) -> Result<bool, String> {
    let memo = memo_path
        .map(PathBuf::from)
        .or_else(default_memo_path)
        .ok_or_else(|| "memo path could not be resolved".to_string())?;

    let canonical_file = validate_task_path(&file_path, &memo)?;
    let content = fs::read_to_string(&canonical_file).map_err(|e| format!("read failed: {}", e))?;
    let (new_content, new_done) = toggle_line(&content, line_number, &expected_text)?;
    fs::write(&canonical_file, new_content).map_err(|e| format!("write failed: {}", e))?;
    Ok(new_done)
}

/// memo TASK frontmatter status 값 6종 — KL-018 status write-back 화이트리스트.
const ALLOWED_STATUSES: &[&str] = &["seed", "ready", "active", "hold", "done", "sealed"];

/// frontmatter 안의 `status:` 라인을 새 값으로 교체. expected_status 와 현재 값이 다르면 fail-safe.
/// CRLF/LF 보존. 반환: (new_content, new_status).
fn replace_status(
    content: &str,
    new_status: &str,
    expected_status: &str,
) -> Result<(String, String), String> {
    if !ALLOWED_STATUSES.contains(&new_status) {
        return Err(format!("invalid status: {}", new_status));
    }

    let normalized = content.replace("\r\n", "\n");
    let after_open = normalized
        .strip_prefix("---\n")
        .ok_or_else(|| "no frontmatter".to_string())?;
    let close_idx = after_open
        .find("\n---")
        .ok_or_else(|| "no frontmatter close".to_string())?;
    let fm_text = &after_open[..close_idx];

    // status 라인 검색 — fm_text 안에서 1회만 등장한다고 가정.
    let mut status_line_idx_in_fm: Option<usize> = None;
    let mut current_status: Option<String> = None;
    for (idx, raw_line) in fm_text.split('\n').enumerate() {
        let trimmed = raw_line.trim();
        if let Some((key, value)) = trimmed.split_once(':') {
            if key.trim() == "status" {
                status_line_idx_in_fm = Some(idx);
                current_status = Some(value.trim().to_string());
                break;
            }
        }
    }

    let status_idx = status_line_idx_in_fm.ok_or_else(|| "no status field".to_string())?;
    let actual = current_status.unwrap_or_default();
    if actual != expected_status {
        return Err(format!(
            "status mismatch — expected '{}', got '{}'",
            expected_status, actual
        ));
    }

    // 원본 content 의 라인 단위로 교체. CRLF/LF 보존을 위해 `split('\n')` 후 join.
    let mut lines: Vec<String> = content.split('\n').map(|line| line.to_string()).collect();
    // 파일 라인 = 1 (open ---) + status_idx (fm 안의 0-base) + 1 (1-base 변환) - 1 (인덱싱) = status_idx + 1
    // 실제 인덱싱: lines[status_idx + 1] (open --- 뒤 첫 fm 라인 = lines[1])
    let file_line_idx = status_idx + 1;
    if file_line_idx >= lines.len() {
        return Err("internal: status line out of range".to_string());
    }

    let raw_line = lines[file_line_idx].clone();
    let has_cr = raw_line.ends_with('\r');
    let cr_suffix = if has_cr { "\r" } else { "" };
    lines[file_line_idx] = format!("status: {}{}", new_status, cr_suffix);

    Ok((lines.join("\n"), new_status.to_string()))
}

/// QuestLog 위젯에서 호출. status 변경 후 파일 write.
/// 반환: 새로 쓰여진 status 문자열.
#[tauri::command]
pub fn set_quest_status(
    file_path: String,
    new_status: String,
    expected_status: String,
    memo_path: Option<String>,
) -> Result<String, String> {
    let memo = memo_path
        .map(PathBuf::from)
        .or_else(default_memo_path)
        .ok_or_else(|| "memo path could not be resolved".to_string())?;

    let canonical_file = validate_task_path(&file_path, &memo)?;
    let content = fs::read_to_string(&canonical_file).map_err(|e| format!("read failed: {}", e))?;
    let (new_content, written) = replace_status(&content, &new_status, &expected_status)?;
    fs::write(&canonical_file, new_content).map_err(|e| format!("write failed: {}", e))?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_LF: &str =
        "---\nid: TASK-WM-007\n---\n\n## 단계\n\n- [x] 완료\n- [ ] 진행중\n";
    const SAMPLE_CRLF: &str =
        "---\r\nid: TASK-WM-007\r\n---\r\n\r\n## 단계\r\n\r\n- [x] 완료\r\n- [ ] 진행중\r\n";
    const SAMPLE_WITH_STATUS_LF: &str =
        "---\nid: TASK-KL-018\nstatus: seed\npriority: normal\n---\n\n## 본문\n";
    const SAMPLE_WITH_STATUS_CRLF: &str =
        "---\r\nid: TASK-KL-018\r\nstatus: seed\r\npriority: normal\r\n---\r\n\r\n## 본문\r\n";

    #[test]
    fn toggle_line_lf_unchecked_to_checked() {
        // 라인 8 = "- [ ] 진행중"
        let (new_content, new_done) = toggle_line(SAMPLE_LF, 8, "진행중").unwrap();
        assert!(new_done);
        assert!(new_content.contains("- [x] 진행중"));
        assert!(new_content.contains("- [x] 완료")); // 다른 라인 보존
    }

    #[test]
    fn toggle_line_lf_checked_to_unchecked() {
        // 라인 7 = "- [x] 완료"
        let (new_content, new_done) = toggle_line(SAMPLE_LF, 7, "완료").unwrap();
        assert!(!new_done);
        assert!(new_content.contains("- [ ] 완료"));
    }

    #[test]
    fn toggle_line_crlf_preserved() {
        let (new_content, _) = toggle_line(SAMPLE_CRLF, 8, "진행중").unwrap();
        assert!(new_content.contains("- [x] 진행중\r\n"));
        // CRLF 가 LF 로 깨지지 않았는지 확인
        assert!(new_content.contains("\r\n"));
    }

    #[test]
    fn toggle_line_uppercase_x_normalized() {
        let upper = "- [X] foo\n";
        let (new_content, new_done) = toggle_line(upper, 1, "foo").unwrap();
        assert!(!new_done);
        assert_eq!(new_content, "- [ ] foo\n");
    }

    #[test]
    fn toggle_line_text_mismatch_errors() {
        let res = toggle_line(SAMPLE_LF, 8, "다른 텍스트");
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("text mismatch"));
    }

    #[test]
    fn toggle_line_not_a_checkbox_errors() {
        // 라인 5 = "## 단계"
        let res = toggle_line(SAMPLE_LF, 5, "anything");
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("not a checkbox line"));
    }

    #[test]
    fn toggle_line_out_of_range_errors() {
        let res = toggle_line(SAMPLE_LF, 999, "anything");
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("out of range"));
    }

    #[test]
    fn toggle_line_zero_errors() {
        let res = toggle_line(SAMPLE_LF, 0, "anything");
        assert!(res.is_err());
    }

    #[test]
    fn toggle_line_preserves_leading_whitespace() {
        let indented = "  - [ ] indent\n";
        let (new_content, _) = toggle_line(indented, 1, "indent").unwrap();
        assert_eq!(new_content, "  - [x] indent\n");
    }

    #[test]
    fn replace_status_lf_seed_to_active() {
        let (new_content, written) =
            replace_status(SAMPLE_WITH_STATUS_LF, "active", "seed").unwrap();
        assert_eq!(written, "active");
        assert!(new_content.contains("status: active"));
        assert!(!new_content.contains("status: seed"));
        // 다른 frontmatter 보존
        assert!(new_content.contains("id: TASK-KL-018"));
        assert!(new_content.contains("priority: normal"));
    }

    #[test]
    fn replace_status_crlf_preserved() {
        let (new_content, _) =
            replace_status(SAMPLE_WITH_STATUS_CRLF, "hold", "seed").unwrap();
        assert!(new_content.contains("status: hold\r\n"));
        assert!(new_content.contains("\r\n"));
    }

    #[test]
    fn replace_status_mismatch_errors() {
        let res = replace_status(SAMPLE_WITH_STATUS_LF, "active", "active");
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("status mismatch"));
    }

    #[test]
    fn replace_status_invalid_status_errors() {
        let res = replace_status(SAMPLE_WITH_STATUS_LF, "exploding", "seed");
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("invalid status"));
    }

    #[test]
    fn replace_status_no_status_field_errors() {
        let no_status = "---\nid: TASK-WM-007\n---\n\n## 본문\n";
        let res = replace_status(no_status, "active", "seed");
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("no status field"));
    }

    #[test]
    fn replace_status_no_frontmatter_errors() {
        let no_fm = "## 본문\n그냥 본문\n";
        let res = replace_status(no_fm, "active", "seed");
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("no frontmatter"));
    }

    #[test]
    fn replace_status_all_six_allowed() {
        for status in &["seed", "ready", "active", "hold", "done", "sealed"] {
            let res = replace_status(SAMPLE_WITH_STATUS_LF, status, "seed");
            assert!(res.is_ok(), "status '{}' should be allowed", status);
        }
    }
}
