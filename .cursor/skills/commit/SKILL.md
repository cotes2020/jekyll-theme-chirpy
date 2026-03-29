---
name: commit
description: >-
  Git commit discipline: the active conversation topic defines what may be
  staged—never add _posts/drafts just because they appear in git status dumps.
  Default = tool-touched paths in THIS chat; explicit paths or directory
  permission override. Never git add -A for unrelated files. Split commits by
  topic. Never push unless asked. English in doc; match repo git log style.
---

# Commit skill (strict)

## Where this applies

- **This repository:** `.cursor/skills/commit/SKILL.md` — Cursor가 이 워크스페이스에서 커밋 요청을 처리할 때 참고할 수 있습니다.
- **All projects (same machine):** `~/.cursor/skills/commit/SKILL.md` — 전역 스킬로, 다른 저장소에서도 동일한 지침을 쓰려면 그쪽을 맞춰 두면 됩니다. (두 파일을 같은 정책으로 유지하는 것을 권장합니다.)

이 문서는 **특정 도구 이름에 묶이지 않습니다.** 아래의 “이번 대화에서 하던 일”은 린트·CI·앱 기능·리팩터·문서 중 **무엇이든** 해당합니다.

## Rule priority (read first — conflicts resolved here)

1. **Explicit user instructions win.** If the user lists paths, says “commit everything under `apps/foo/`”, “stage only KarmoLab”, “do **not** commit `_posts`”, “split this into N commits”, or “squash into one commit”, follow that even if it differs from the defaults below.
2. **Default when the user does *not* narrow or expand scope:** commit **only** paths the agent **created, modified, or deleted with tools** in **this conversation** (Write, StrReplace, Delete, EditNotebook, etc.). **Never** include files the agent only read or searched.
3. **Phrases like “related”, “sensible for this branch”, “fix the build”, “keep it consistent”** do **not** widen the default set. They are **not** permission to stage other dirty or untracked files.
4. **Never `git push`** unless the user clearly asks (e.g. “push”, “push to origin”). “Commit only”, “no push”, “do not push” ⇒ **no push**.
5. **Conversation topic binds the commit.** This thread is about a **specific task or change** ⇒ stage **only** paths that belong to that task. **Do not** tack on unrelated trees (e.g. `_posts/`, personal drafts, other apps) just because they appear in `git status` or a pasted “Currently unstaged files” block—that block is **diagnostic**, **not** a staging allowlist.
6. **`_posts/` and personal blog/diary** require **explicit** natural-language intent to commit (e.g. “`_posts` 포함해서”, “포스트도 커밋”, “commit my Jekyll drafts”). Listing unstaged paths that happen to include `_posts` is **not** that intent. When in doubt, **omit `_posts`** and ask once.

If anything is ambiguous, **ask once** with concrete options; do not guess by staging extra paths.

---

## Conversation topic binding (mandatory)

- Infer the **user’s active goal** from the **full thread**, not only the latest line. Commits must match that goal’s footprint (files this conversation actually changed or that the user **explicitly** scoped).
- **`_posts/**`**: **default exclude.** Include only when the user clearly asks to commit blog/site posts or explicitly names `_posts` paths as in-scope **for this commit**.
- **Mixed dirty tree** (task A done in-chat + unrelated local edits B): commit **only** what matches **task A**. **Never** bundle other areas “to be helpful” or “because they were listed.”
- If the user wants both, they must say so in plain language; otherwise **two separate commits** only after explicit confirmation—or **one topic only**.

---

## Two modes (pick one per request)

### Mode A — Session-only (default)

- **Eligible paths:** tool-touched paths in **this chat** only.
- **Topic split:** If those paths span **multiple unrelated concerns** (e.g. feature + unrelated docs + config), use **multiple commits**—one topic per commit. Exception: user says to use a **single** commit.
- **Forbidden:** `git add -A`, `git add .`, `git add some/dir/` **unless** that directory was **only** touched in-session **and** contains no extra untracked/unrelated files—or the user explicitly allowed the directory add.

### Mode B — User-directed

- **Eligible paths:** what the user specified **after filtering by conversation topic** (see Rule 5–6 and **Conversation topic binding**). A pasted `git status` / “Currently unstaged files” list is **not** an automatic allowlist—stage only paths tied to **this thread’s work** plus paths the user **explicitly** names for inclusion (e.g. “and `_posts/foo.md`”).
- **`_posts/`:** still needs **explicit** intent to commit (Rule 6); never add because it appeared alongside real targets in a file list.
- **Topic split:** If the user asks to “commit the right things for this branch” **without** listing files, infer **logical topics** (e.g. code vs content vs config—**whatever domains are actually separate**) and **separate commits per topic**. Do **not** ship one giant commit that mixes unrelated areas unless they ask for one commit.
- **User exclusions:** Permanent or stated exclusions (e.g. “never commit blog `_posts`”) **must** be respected: leave those paths unstaged and **do not** put them in any commit.

---

## Anti-patterns (do not do this)

- Staging “the whole app” or “everything under `apps/…`” to be helpful when only a few files were edited in-session.
- One commit titled like “refactor + lazy load + new widget + docs” when those are **separate** concerns—**split** unless the user asked for one commit.
- Interpreting “commit what’s relevant to the branch” as “add all unstaged files in the repo.”
- Treating a pasted **“Currently unstaged files”** dump as permission to stage **every** path (especially `_posts/`).
- Bundling **`_posts`/diary/personal content** with **unrelated** code or config changes without the user **explicitly** asking to include that content in this request.
- Running `git commit` without checking `git diff --cached` / `git status` for stray paths.
- Pushing after commit without an explicit push request.

---

## Procedure (execute in order)

1. **Classify the request**  
   Session-only (Mode A) vs user-directed (Mode B). If mixed, **B overrides** for scope; still apply topic splitting unless user wants one commit.

2. **Enumerate candidate paths**  
   Write the list explicitly (mental or brief note). Mode A: from chat tool edits only. Mode B: from user text only.

3. **`git status` and optional `git diff`**  
   Confirm candidates exist and match intent.

4. **Mixed-change / overlap check**  
   If a candidate file likely contains **non-session** edits and the user did not say “commit all changes in this file”, **stop** and ask: (A) omit file, (B) commit whole file, (C) user will split manually.

5. **Topic grouping (required if >1 topic)**  
   Group candidates by purpose. **Each commit** gets one group. Message must match **that group only**.

6. **Stage narrowly**  
   `git add -- <path1> <path2> …` per topic. After each `git add`, run `git status` or `git diff --cached --stat` and **remove** wrong paths with `git restore --staged -- <path>`.

7. **Pre-commit checklist (mandatory)**  
   - [ ] Staged paths ⊆ allowed set for this mode **and** match **this conversation’s topic**.  
   - [ ] No accidental `git add -A` / `git add .`.  
   - [ ] **`_posts/`** (and **analogous** private or content-only trees the user cares about) is **absent** unless the user **explicitly** asked to commit that material for this commit.  
   - [ ] User-excluded paths are **not** staged.  
   - [ ] Commit message matches **this** topic only.

8. **Commit message**  
   Short subject; match recent repo style (`git log -5 --oneline`). Body only if needed (one or two lines).

9. **Repeat** for remaining topic groups (clear staging between groups if needed).

10. **Do not push** unless asked.

---

## Shell: Windows / PowerShell

`cd … && git …` often **fails** in PowerShell. Chain with **`;`** instead, e.g. `cd path; git status`.

---

## Triggers

Apply this skill when the user asks to commit, `/commit`, stage, split commits, or “commit related / branch-appropriate” changes. **Those words do not expand Mode A** unless the user also gives paths or explicit directory permission. **They never imply `_posts/`** unless the user explicitly says to commit blog/draft content.
