---
name: commit
description: >-
  Git commit discipline: default = only paths edited via agent tools in THIS
  chat; user override = explicit paths, whole-directory permission, or branch
  curation when the user says so. Never git add -A / git add . for unrelated
  files. Split commits by topic (no mega-blobs). Never push unless explicitly
  asked. Prefer English in this doc; follow repo git log style for messages.
---

# Commit skill (strict)

## Rule priority (read first — conflicts resolved here)

1. **Explicit user instructions win.** If the user lists paths, says “commit everything under `apps/foo/`”, “stage only KarmoLab”, “do **not** commit `_posts`”, “split this into N commits”, or “squash into one commit”, follow that even if it differs from the defaults below.
2. **Default when the user does *not* narrow or expand scope:** commit **only** paths the agent **created, modified, or deleted with tools** in **this conversation** (Write, StrReplace, Delete, EditNotebook, etc.). **Never** include files the agent only read or searched.
3. **Phrases like “related”, “sensible for this branch”, “fix the build”, “keep it consistent”** do **not** widen the default set. They are **not** permission to stage other dirty or untracked files.
4. **Never `git push`** unless the user clearly asks (e.g. “push”, “push to origin”). “Commit only”, “no push”, “do not push” ⇒ **no push**.

If anything is ambiguous, **ask once** with concrete options; do not guess by staging extra paths.

---

## Two modes (pick one per request)

### Mode A — Session-only (default)

- **Eligible paths:** tool-touched paths in **this chat** only.
- **Topic split:** If those paths span **multiple unrelated concerns** (e.g. feature + unrelated docs + config), use **multiple commits**—one topic per commit. Exception: user says to use a **single** commit.
- **Forbidden:** `git add -A`, `git add .`, `git add some/dir/` **unless** that directory was **only** touched in-session **and** contains no extra untracked/unrelated files—or the user explicitly allowed the directory add.

### Mode B — User-directed

- **Eligible paths:** exactly what the user specified (listed paths, glob as given, or “entire directory X” when they say so).
- **Topic split:** If the user asks to “commit the right things for this branch” **without** listing files, infer **logical topics** (e.g. app vs blog vs tooling) and **separate commits per topic**. Do **not** ship one giant commit that mixes unrelated areas unless they ask for one commit.
- **User exclusions:** Permanent or stated exclusions (e.g. “never commit blog `_posts`”) **must** be respected: leave those paths unstaged and **do not** put them in any commit.

---

## Anti-patterns (do not do this)

- Staging “the whole app” or “everything under `apps/…`” to be helpful when only a few files were edited in-session.
- One commit titled like “refactor + lazy load + new widget + docs” when those are **separate** concerns—**split** unless the user asked for one commit.
- Interpreting “commit what’s relevant to the branch” as “add all unstaged files in the repo.”
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
   - [ ] Staged paths ⊆ allowed set for this mode.  
   - [ ] No accidental `git add -A` / `git add .`.  
   - [ ] User-excluded paths (e.g. `_posts`) are **not** staged.  
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

Apply this skill when the user asks to commit, `/commit`, stage, split commits, or “commit related / branch-appropriate” changes. **Those words do not expand Mode A** unless the user also gives paths or explicit directory permission.
