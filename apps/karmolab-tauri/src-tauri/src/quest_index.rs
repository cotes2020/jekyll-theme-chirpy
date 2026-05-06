// quest_index.rs
// QuestLog 위젯용 — memo TASK 파일을 런타임에 읽어 트리 데이터 산출.
//
// 데이터 소스 (모두 로컬, github.io 레포 외부):
//   - {memoPath}/wm/tasks/TASK-WM-*.md
//   - {memoPath}/projects/karmolab/tasks/TASK-KL-*.md
//   - {memoPath}/projects/yawnbot/tasks/TASK-YB-*.md
//   - {memoPath}/life/tasks/TASK-LIFE-*.md
//   - {memoPath}/hobby/tasks/TASK-HOBBY-*.md
//   - {memoPath}/learning/tasks/TASK-LEARN-*.md
//
// 클라이언트는 invoke('get_quest_tree', { memoPath }) 로 호출.
// 변경 사항 푸시는 Phase E (notify watcher) 에서 추가 예정.
//
// 정책 (dashboard `karmoddrine_state.rs` 와 동일):
//   - yaml crate 안 씀 — 라인 단위 직접 파싱 (frontmatter 스펙 단순)
//   - 부분 실패 허용 — 깨진 파일 1개 때문에 위젯 전체 죽지 않게, errors[] 에 보고

use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// 사용자 홈 디렉토리 (`USERPROFILE` Windows / `HOME` POSIX). dashboard `karmoddrine_state.rs` 와 동일 정책.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

/// 기본 memo 경로 — `~/repos/karmoddrine/memo`. 사용자 머신 표준 위치.
/// 다른 위치 사용 시 invoke 인자(`memoPath`) 로 명시.
fn default_memo_path() -> Option<PathBuf> {
    home_dir().map(|h| h.join("repos").join("karmoddrine").join("memo"))
}

#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CheckItem {
    pub text: String,
    pub done: bool,
    pub group: Option<String>,
    /// 1-base 파일 라인 번호. v2 write-back (toggle_quest_check) 가 식별자로 사용.
    pub line_number: u32,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TaskNode {
    pub id: String,
    pub status: String,
    pub priority: String,
    pub path: Vec<String>,
    pub parent: Option<String>,
    pub tags: Vec<String>,
    pub title: String,
    pub file_path: String,
    pub checks: Vec<CheckItem>,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TaskError {
    pub file_path: String,
    pub reason: String,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct QuestTree {
    pub tasks: Vec<TaskNode>,
    pub generated_at_unix: u64,
    pub memo_path: String,
    pub errors: Vec<TaskError>,
}

const DOMAIN_DIRS: &[&str] = &[
    "wm/tasks",
    "projects/karmolab/tasks",
    "projects/yawnbot/tasks",
    "life/tasks",
    "hobby/tasks",
    "learning/tasks",
];

/// frontmatter 파싱 — `---\n key: value\n ... \n---\n body`.
/// 반환: (key→value map, body, body_start_line).
/// `body_start_line` 은 1-base 파일 라인 번호 — v2 write-back 이 사용.
/// frontmatter 가 없거나 닫는 `---` 가 없으면 None.
fn parse_frontmatter(content: &str) -> Option<(HashMap<String, String>, String, u32)> {
    let normalized = content.replace("\r\n", "\n");
    let after_open = normalized.strip_prefix("---\n")?;
    let close_idx = after_open.find("\n---")?;
    let fm_text = &after_open[..close_idx];
    let body_with_close = &after_open[close_idx + "\n---".len()..];
    let body = body_with_close.strip_prefix('\n').unwrap_or(body_with_close);

    // body 시작 파일 라인 = 1 (open ---) + fm_lines + 1 (close ---) + 1 (다음 라인)
    let fm_lines = if fm_text.is_empty() {
        0
    } else {
        fm_text.matches('\n').count() + 1
    };
    let body_start_line = (fm_lines + 3) as u32;

    let mut map: HashMap<String, String> = HashMap::new();
    for raw_line in fm_text.split('\n') {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((key, value)) = line.split_once(':') {
            map.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    Some((map, body.to_string(), body_start_line))
}

/// `[a, b, c]` 형식 → `vec![a, b, c]`. 따옴표 제거. 그 외 형식이면 단일 요소.
fn parse_list(value: &str) -> Vec<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        return trimmed[1..trimmed.len() - 1]
            .split(',')
            .map(|cell| cell.trim().trim_matches('"').trim_matches('\'').to_string())
            .filter(|cell| !cell.is_empty())
            .collect();
    }
    vec![trimmed.to_string()]
}

/// 본문 체크박스 추출 — `- [ ]` / `- [x]` / `- [X]` 라인. 가장 가까운 위 h2/h3 헤더가 group.
/// `body_start_line` 은 body 의 첫 라인이 파일에서 몇 번째 라인인지 (1-base).
/// 각 CheckItem 의 `line_number` = 절대 파일 라인 번호.
fn extract_checks(body: &str, body_start_line: u32) -> Vec<CheckItem> {
    let mut checks: Vec<CheckItem> = Vec::new();
    let mut current_group: Option<String> = None;
    for (idx, raw_line) in body.split('\n').enumerate() {
        let line = raw_line.trim_end_matches('\r');
        let trimmed = line.trim_start();

        // 헤더 갱신 (h2 / h3). h1 은 거의 없음.
        if let Some(rest) = trimmed.strip_prefix("### ") {
            current_group = Some(rest.trim().to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("## ") {
            current_group = Some(rest.trim().to_string());
            continue;
        }

        // 체크박스 — 표시 4종 케이스
        let (rest, done) = if let Some(r) = trimmed.strip_prefix("- [ ]") {
            (r, false)
        } else if let Some(r) = trimmed.strip_prefix("- [x]") {
            (r, true)
        } else if let Some(r) = trimmed.strip_prefix("- [X]") {
            (r, true)
        } else {
            continue;
        };
        let text = rest.trim().to_string();
        if text.is_empty() {
            continue;
        }
        checks.push(CheckItem {
            text,
            done,
            group: current_group.clone(),
            line_number: body_start_line + idx as u32,
        });
    }
    checks
}

/// 파일명에서 `TASK-{PREFIX}-{NNN}[-X]-` 떼고 .md 떼고 하이픈을 공백으로 — 트리 표시용 사람-친화 제목.
/// sub-phase letter (`-A-`, `-B-` 등) 는 정확히 1글자 + ASCII 대문자일 때만 떼낸다 —
/// `Research-` 같은 단어 첫 글자가 우연히 대문자인 케이스를 보호.
/// prefix 매칭 실패 시 파일명 그대로 (확장자만 제거, 하이픈 → 공백).
fn extract_title(file_name: &str) -> String {
    let stem = file_name.strip_suffix(".md").unwrap_or(file_name);
    let Some(after_task) = stem.strip_prefix("TASK-") else {
        return stem.replace('-', " ");
    };
    let Some(after_prefix) = after_task.splitn(2, '-').nth(1) else {
        return stem.replace('-', " ");
    };
    let Some(after_nnn) = after_prefix.splitn(2, '-').nth(1) else {
        return stem.replace('-', " ");
    };
    let mut parts = after_nnn.splitn(2, '-');
    let first_segment = parts.next().unwrap_or("");
    let is_sub_phase_letter = first_segment.len() == 1
        && first_segment
            .chars()
            .next()
            .map_or(false, |c| c.is_ascii_uppercase());
    let rest = if is_sub_phase_letter {
        parts.next().unwrap_or("")
    } else {
        after_nnn
    };
    rest.replace('-', " ")
}

/// 단일 TASK 파일 → TaskNode. parse 실패는 Err — 호출자가 errors[] 에 모음.
fn parse_one_task(file_name: &str, rel_path: &str, content: &str) -> Result<TaskNode, String> {
    let (fm, body, body_start_line) =
        parse_frontmatter(content).ok_or_else(|| "frontmatter 없음".to_string())?;

    let id = fm
        .get("id")
        .cloned()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "id 필드 없음".to_string())?;

    let status = fm
        .get("status")
        .cloned()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "ready".to_string());

    let priority = fm
        .get("priority")
        .cloned()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "normal".to_string());

    let path = fm
        .get("path")
        .map(|value| parse_list(value))
        .unwrap_or_default();

    let parent = fm
        .get("parent")
        .cloned()
        .filter(|value| !value.is_empty() && value != "null");

    let tags = fm
        .get("tags")
        .map(|value| parse_list(value))
        .unwrap_or_default();

    let title = extract_title(file_name);
    let checks = extract_checks(&body, body_start_line);

    Ok(TaskNode {
        id,
        status,
        priority,
        path,
        parent,
        tags,
        title,
        file_path: rel_path.to_string(),
        checks,
    })
}

#[tauri::command]
pub fn get_quest_tree(memo_path: Option<String>) -> Result<QuestTree, String> {
    let memo = match memo_path.filter(|s| !s.is_empty()) {
        Some(value) => PathBuf::from(value),
        None => default_memo_path().ok_or_else(|| {
            "default memo path 결정 실패 (USERPROFILE / HOME 환경변수 없음)".to_string()
        })?,
    };
    if !memo.is_dir() {
        return Err(format!(
            "memo path 가 디렉토리가 아님: {}",
            memo.display()
        ));
    }
    let canonical_memo = memo
        .canonicalize()
        .map_err(|err| format!("canonicalize 실패: {}", err))?;

    let mut tasks: Vec<TaskNode> = Vec::new();
    let mut errors: Vec<TaskError> = Vec::new();

    for rel_dir in DOMAIN_DIRS {
        let full_dir = canonical_memo.join(rel_dir);
        if !full_dir.is_dir() {
            continue;
        }
        let entries = match fs::read_dir(&full_dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let file_name = match path.file_name().and_then(|n| n.to_str()) {
                Some(name) => name.to_string(),
                None => continue,
            };
            if !file_name.starts_with("TASK-") || !file_name.ends_with(".md") {
                continue;
            }

            let rel_path = path
                .strip_prefix(&canonical_memo)
                .map(|relative| relative.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| file_name.clone());

            let content = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(err) => {
                    errors.push(TaskError {
                        file_path: rel_path,
                        reason: format!("read 실패: {}", err),
                    });
                    continue;
                }
            };

            match parse_one_task(&file_name, &rel_path, &content) {
                Ok(task) => tasks.push(task),
                Err(reason) => errors.push(TaskError {
                    file_path: rel_path,
                    reason,
                }),
            }
        }
    }

    let generated_at_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);

    Ok(QuestTree {
        tasks,
        generated_at_unix,
        memo_path: canonical_memo.to_string_lossy().to_string(),
        errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "---\nid: TASK-WM-007\nstatus: ready\npriority: normal\npath: [wm]\ntags: [content, lumber]\n---\n\n## 목표\n\n나무 벌목.\n\n## 단계\n\n- [x] 완료된 거\n- [ ] 안 한 거\n\n### 하위\n\n- [X] 대문자 X 도\n- [ ] 마지막\n";

    const SAMPLE_NO_FM: &str = "## 목표\n\nfrontmatter 없는 파일\n";
    const SAMPLE_NO_ID: &str = "---\nstatus: ready\n---\n\n## 목표\n";

    #[test]
    fn parse_frontmatter_basic() {
        let (fm, body, body_start_line) = parse_frontmatter(SAMPLE).unwrap();
        assert_eq!(fm.get("id").map(String::as_str), Some("TASK-WM-007"));
        assert_eq!(fm.get("status").map(String::as_str), Some("ready"));
        assert!(body.starts_with("\n## 목표"));
        // SAMPLE: --- (1) + 5 fm lines (2~6) + --- (7) → body 첫 라인 = 8
        assert_eq!(body_start_line, 8);
    }

    #[test]
    fn parse_frontmatter_missing_returns_none() {
        assert!(parse_frontmatter(SAMPLE_NO_FM).is_none());
    }

    #[test]
    fn parse_list_array_form() {
        assert_eq!(parse_list("[wm]"), vec!["wm"]);
        assert_eq!(parse_list("[a, b, c]"), vec!["a", "b", "c"]);
        assert_eq!(parse_list(""), Vec::<String>::new());
    }

    #[test]
    fn parse_list_quoted() {
        assert_eq!(parse_list("[\"a\", 'b']"), vec!["a", "b"]);
    }

    #[test]
    fn extract_checks_grouped_and_mixed() {
        let (_, body, body_start_line) = parse_frontmatter(SAMPLE).unwrap();
        let checks = extract_checks(&body, body_start_line);
        assert_eq!(checks.len(), 4);
        assert_eq!(checks[0].text, "완료된 거");
        assert!(checks[0].done);
        assert_eq!(checks[0].group.as_deref(), Some("단계"));
        assert!(!checks[1].done);
        assert_eq!(checks[1].group.as_deref(), Some("단계"));
        assert!(checks[2].done); // 대문자 X
        assert_eq!(checks[2].group.as_deref(), Some("하위"));
        assert!(!checks[3].done);
    }

    #[test]
    fn extract_checks_line_numbers_match_file_lines() {
        // SAMPLE 파일의 체크박스는 라인 15, 16, 20, 21.
        let (_, body, body_start_line) = parse_frontmatter(SAMPLE).unwrap();
        let checks = extract_checks(&body, body_start_line);
        assert_eq!(checks[0].line_number, 15);
        assert_eq!(checks[1].line_number, 16);
        assert_eq!(checks[2].line_number, 20);
        assert_eq!(checks[3].line_number, 21);
    }

    #[test]
    fn extract_checks_empty_body() {
        assert!(extract_checks("", 1).is_empty());
        assert!(extract_checks("## 목표\n\n그냥 본문\n", 1).is_empty());
    }

    #[test]
    fn extract_title_from_filename() {
        assert_eq!(
            extract_title("TASK-WM-007-나무벌목-컨텐츠.md"),
            "나무벌목 컨텐츠"
        );
        assert_eq!(
            extract_title("TASK-KL-001-window-titlebar-custom.md"),
            "window titlebar custom"
        );
        assert_eq!(
            extract_title("TASK-WM-027-A-블록-청크-데이터모델.md"),
            "블록 청크 데이터모델"
        );
        // Research 같은 단어 첫 글자 대문자는 sub-phase letter 가 아님 — 보존
        assert_eq!(
            extract_title("TASK-WM-004-Research-시스템-연결.md"),
            "Research 시스템 연결"
        );
    }

    #[test]
    fn parse_one_task_full_path() {
        let task = parse_one_task("TASK-WM-007-나무벌목-컨텐츠.md", "wm/tasks/TASK-WM-007-나무벌목-컨텐츠.md", SAMPLE).unwrap();
        assert_eq!(task.id, "TASK-WM-007");
        assert_eq!(task.status, "ready");
        assert_eq!(task.path, vec!["wm".to_string()]);
        assert_eq!(task.tags, vec!["content".to_string(), "lumber".to_string()]);
        assert!(task.parent.is_none());
        assert_eq!(task.checks.len(), 4);
    }

    #[test]
    fn parse_one_task_missing_id_errors() {
        let res = parse_one_task("TASK-WM-001-x.md", "wm/tasks/TASK-WM-001-x.md", SAMPLE_NO_ID);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("id"));
    }

    #[test]
    fn parse_one_task_no_frontmatter_errors() {
        let res = parse_one_task("TASK-WM-001-x.md", "wm/tasks/TASK-WM-001-x.md", SAMPLE_NO_FM);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("frontmatter"));
    }

    #[test]
    fn parse_one_task_with_parent_chain() {
        let content = "---\nid: TASK-WM-027-A\nstatus: done\npriority: normal\npath: [wm, voxel]\nparent: TASK-WM-027\n---\n\n본문\n";
        let task = parse_one_task("TASK-WM-027-A-블록.md", "wm/tasks/TASK-WM-027-A-블록.md", content).unwrap();
        assert_eq!(task.parent.as_deref(), Some("TASK-WM-027"));
        assert_eq!(task.path, vec!["wm".to_string(), "voxel".to_string()]);
    }

    #[test]
    fn default_memo_path_returns_some_in_test_env() {
        // 테스트 환경에 USERPROFILE 또는 HOME 둘 중 하나는 있다는 일반적 가정.
        assert!(default_memo_path().is_some());
    }
}
