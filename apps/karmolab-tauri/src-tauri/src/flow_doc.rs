// flow_doc.rs
// karmoddrine-flows 위젯용 — `memo/flows/*.md` 흐름 문서를 사이드바 목록 + 본문으로.
//
// 데이터 소스 (모두 로컬, github.io 레포 외부):
//   - {memoPath}/flows/<slug>.md  (frontmatter: title, summary, category)
//
// 명령:
//   - list_flow_docs(memoPath?) -> FlowList
//   - read_flow_doc(memoPath?, slug) -> FlowDoc
//
// 정책 (quest_index.rs 와 동일):
//   - yaml crate 안 씀 — 라인 단위 직접 파싱
//   - 부분 실패 허용 — 깨진 파일 1개 때문에 list 전체 죽지 않게, errors[] 에 보고
//   - read 측 slug 검증 — `..` `/` `\` `null` 차단 + canonicalize 후 flows dir prefix 검증

use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const FLOWS_DIR: &str = "flows";

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

fn default_memo_path() -> Option<PathBuf> {
    home_dir().map(|h| h.join("repos").join("karmoddrine").join("memo"))
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FlowMeta {
    pub slug: String,
    pub title: String,
    pub summary: Option<String>,
    pub category: Option<String>,
    pub last_modified_unix: u64,
    pub file_path: String,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FlowDoc {
    pub meta: FlowMeta,
    pub body: String,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FlowError {
    pub file_path: String,
    pub reason: String,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct FlowList {
    pub flows: Vec<FlowMeta>,
    pub generated_at_unix: u64,
    pub memo_path: String,
    pub errors: Vec<FlowError>,
}

/// frontmatter 파싱 — `---\n key: value\n ... \n---\n body`. quest_index 와 동일 스펙.
fn parse_frontmatter(content: &str) -> Option<(HashMap<String, String>, String)> {
    let normalized = content.replace("\r\n", "\n");
    let after_open = normalized.strip_prefix("---\n")?;
    let close_idx = after_open.find("\n---")?;
    let fm_text = &after_open[..close_idx];
    let body_with_close = &after_open[close_idx + "\n---".len()..];
    let body = body_with_close.strip_prefix('\n').unwrap_or(body_with_close);

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
    Some((map, body.to_string()))
}

fn validate_slug(slug: &str) -> Result<(), String> {
    if slug.is_empty() {
        return Err("slug 비어있음".into());
    }
    if slug.contains("..") || slug.contains('/') || slug.contains('\\') || slug.contains('\0') {
        return Err("slug 에 .. / \\ NUL 사용 불가".into());
    }
    Ok(())
}

fn resolve_memo(memo_path: Option<String>) -> Result<PathBuf, String> {
    let memo = match memo_path.filter(|s| !s.is_empty()) {
        Some(value) => PathBuf::from(value),
        None => default_memo_path().ok_or_else(|| {
            "default memo path 결정 실패 (USERPROFILE / HOME 환경변수 없음)".to_string()
        })?,
    };
    if !memo.is_dir() {
        return Err(format!("memo path 가 디렉토리가 아님: {}", memo.display()));
    }
    memo.canonicalize()
        .map_err(|err| format!("canonicalize 실패: {}", err))
}

/// content + 외부에서 받은 slug/rel_path/mtime 으로 FlowMeta 빌드. fs 의존 분리해서 테스트하기 쉽게.
fn build_flow_meta(
    content: &str,
    slug: &str,
    rel_path: &str,
    last_modified_unix: u64,
) -> Result<FlowMeta, String> {
    let (fm, _body) =
        parse_frontmatter(content).ok_or_else(|| "frontmatter 없음".to_string())?;
    let title = fm
        .get("title")
        .cloned()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| slug.to_string());
    let summary = fm
        .get("summary")
        .cloned()
        .filter(|value| !value.is_empty());
    let category = fm
        .get("category")
        .cloned()
        .filter(|value| !value.is_empty());
    Ok(FlowMeta {
        slug: slug.to_string(),
        title,
        summary,
        category,
        last_modified_unix,
        file_path: rel_path.to_string(),
    })
}

fn mtime_unix(path: &Path) -> u64 {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[tauri::command]
pub fn list_flow_docs(memo_path: Option<String>) -> Result<FlowList, String> {
    let memo = resolve_memo(memo_path)?;
    let flows_dir = memo.join(FLOWS_DIR);

    let mut flows: Vec<FlowMeta> = Vec::new();
    let mut errors: Vec<FlowError> = Vec::new();

    if flows_dir.is_dir() {
        let entries = match fs::read_dir(&flows_dir) {
            Ok(entries) => entries,
            Err(err) => {
                return Err(format!("flows 디렉토리 read 실패: {}", err));
            }
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
            if !file_name.ends_with(".md") {
                continue;
            }
            // 숨김 파일만 스킵 — `_test.md` 같은 검증용 파일은 보임 (사용자가 검증할 때 필요).
            if file_name.starts_with('.') {
                continue;
            }
            let slug = file_name.trim_end_matches(".md").to_string();
            let rel_path = path
                .strip_prefix(&memo)
                .map(|relative| relative.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| format!("{}/{}", FLOWS_DIR, file_name));

            let content = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(err) => {
                    errors.push(FlowError {
                        file_path: rel_path,
                        reason: format!("read 실패: {}", err),
                    });
                    continue;
                }
            };

            match build_flow_meta(&content, &slug, &rel_path, mtime_unix(&path)) {
                Ok(meta) => flows.push(meta),
                Err(reason) => errors.push(FlowError {
                    file_path: rel_path,
                    reason,
                }),
            }
        }
    }

    flows.sort_by(|a, b| a.slug.cmp(&b.slug));

    let generated_at_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);

    Ok(FlowList {
        flows,
        generated_at_unix,
        memo_path: memo.to_string_lossy().to_string(),
        errors,
    })
}

#[tauri::command]
pub fn read_flow_doc(memo_path: Option<String>, slug: String) -> Result<FlowDoc, String> {
    validate_slug(&slug)?;
    let memo = resolve_memo(memo_path)?;
    let flows_dir = memo.join(FLOWS_DIR);

    if !flows_dir.is_dir() {
        return Err(format!(
            "flows 디렉토리가 없습니다: {}",
            flows_dir.display()
        ));
    }
    let canonical_flows = flows_dir
        .canonicalize()
        .map_err(|err| format!("flows canonicalize 실패: {}", err))?;

    let path = flows_dir.join(format!("{}.md", slug));
    if !path.is_file() {
        return Err(format!("flow 파일 없음: flows/{}.md", slug));
    }
    let canonical_path = path
        .canonicalize()
        .map_err(|err| format!("path canonicalize 실패: {}", err))?;
    if !canonical_path.starts_with(&canonical_flows) {
        return Err("path traversal 의심 — flows 디렉토리 밖".into());
    }

    let content = fs::read_to_string(&canonical_path)
        .map_err(|err| format!("read 실패: {}", err))?;
    let (_, body) =
        parse_frontmatter(&content).ok_or_else(|| "frontmatter 없음".to_string())?;

    let rel_path = canonical_path
        .strip_prefix(&memo)
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| format!("{}/{}.md", FLOWS_DIR, slug));

    let meta = build_flow_meta(&content, &slug, &rel_path, mtime_unix(&canonical_path))?;

    Ok(FlowDoc { meta, body })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "---\ntitle: 메타 루프\nsummary: 사용자→memo→Claude→commit\ncategory: umbrella\n---\n\n# 메타 루프\n\n```mermaid\nflowchart LR\nA --> B\n```\n";
    const SAMPLE_NO_FM: &str = "# 그냥 본문\n";
    const SAMPLE_FM_ONLY_REQUIRED: &str = "---\ntitle: 단일\n---\n본문\n";
    const SAMPLE_EMPTY_FIELDS: &str = "---\ntitle:\nsummary:\ncategory:\n---\n본문\n";

    #[test]
    fn parse_frontmatter_basic() {
        let (fm, body) = parse_frontmatter(SAMPLE).unwrap();
        assert_eq!(fm.get("title").map(String::as_str), Some("메타 루프"));
        assert_eq!(fm.get("category").map(String::as_str), Some("umbrella"));
        assert!(body.starts_with("\n# 메타 루프"));
    }

    #[test]
    fn parse_frontmatter_missing_returns_none() {
        assert!(parse_frontmatter(SAMPLE_NO_FM).is_none());
    }

    #[test]
    fn validate_slug_accepts_simple() {
        assert!(validate_slug("meta-loop").is_ok());
        assert!(validate_slug("wm_quest").is_ok());
        assert!(validate_slug("한글흐름").is_ok());
    }

    #[test]
    fn validate_slug_rejects_empty() {
        assert!(validate_slug("").is_err());
    }

    #[test]
    fn validate_slug_rejects_traversal() {
        assert!(validate_slug("..").is_err());
        assert!(validate_slug("a/b").is_err());
        assert!(validate_slug("a\\b").is_err());
        assert!(validate_slug("a..b").is_err());
        assert!(validate_slug("a\0b").is_err());
    }

    #[test]
    fn build_flow_meta_full_path() {
        let meta = build_flow_meta(SAMPLE, "meta-loop", "flows/meta-loop.md", 1700000000).unwrap();
        assert_eq!(meta.slug, "meta-loop");
        assert_eq!(meta.title, "메타 루프");
        assert_eq!(meta.summary.as_deref(), Some("사용자→memo→Claude→commit"));
        assert_eq!(meta.category.as_deref(), Some("umbrella"));
        assert_eq!(meta.file_path, "flows/meta-loop.md");
        assert_eq!(meta.last_modified_unix, 1700000000);
    }

    #[test]
    fn build_flow_meta_falls_back_to_slug_when_no_title() {
        let content = "---\nsummary: only summary\n---\n본문\n";
        let meta = build_flow_meta(content, "fallback-slug", "flows/fallback-slug.md", 0).unwrap();
        assert_eq!(meta.title, "fallback-slug");
        assert_eq!(meta.summary.as_deref(), Some("only summary"));
        assert!(meta.category.is_none());
    }

    #[test]
    fn build_flow_meta_optional_fields_minimal() {
        let meta = build_flow_meta(SAMPLE_FM_ONLY_REQUIRED, "단일", "flows/단일.md", 0).unwrap();
        assert_eq!(meta.title, "단일");
        assert!(meta.summary.is_none());
        assert!(meta.category.is_none());
    }

    #[test]
    fn build_flow_meta_treats_empty_fields_as_none() {
        let meta = build_flow_meta(SAMPLE_EMPTY_FIELDS, "x", "flows/x.md", 0).unwrap();
        // empty title → fallback to slug
        assert_eq!(meta.title, "x");
        assert!(meta.summary.is_none());
        assert!(meta.category.is_none());
    }

    #[test]
    fn build_flow_meta_no_frontmatter_errors() {
        let res = build_flow_meta(SAMPLE_NO_FM, "x", "flows/x.md", 0);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("frontmatter"));
    }

    #[test]
    fn default_memo_path_returns_some_in_test_env() {
        assert!(default_memo_path().is_some());
    }
}
