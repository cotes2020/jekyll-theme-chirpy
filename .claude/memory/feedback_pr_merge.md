---
name: PR merge preference
description: User prefers squash merge with branch deletion
type: feedback
---

When merging PRs, use squash merge and delete the branch: `gh pr merge <number> --squash --delete-branch`.

**Why:** User confirmed this approach when asked about the first PR merge in this project.

**How to apply:** Default to `--squash --delete-branch` unless the user specifies otherwise.
