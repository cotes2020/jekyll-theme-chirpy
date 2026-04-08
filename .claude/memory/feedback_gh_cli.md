---
name: GitHub CLI usage on Windows
description: gh CLI is not in PATH on this machine; must use full path and --repo flag
type: feedback
---

Always invoke gh CLI with full path: `"/c/Program Files/GitHub CLI/gh.exe"`. It is not on the default bash PATH in this Windows environment.

**Why:** The standard `gh` command fails with "not found" in bash sessions.

**How to apply:** Any time a gh CLI command is needed (pr create, pr merge, issue list, etc.), use the full path. Also always pass `--repo mascari4615/mascari4615.github.io` explicitly to avoid ambiguity.
