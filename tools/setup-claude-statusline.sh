#!/usr/bin/env bash
#
# Set up a Claude Code status line that shows context window, cost, and rate limits.
#
# Writes ~/.claude/statusline.sh and patches ~/.claude/settings.json.
# Requires: awk, sed, git (optional, for branch display)
#
# Usage:
#   bash tools/setup-claude-statusline.sh [options]

set -euo pipefail

CLAUDE_DIR="${HOME}/.claude"
STATUSLINE_SCRIPT="${CLAUDE_DIR}/statusline.sh"
SETTINGS_FILE="${CLAUDE_DIR}/settings.json"

help() {
  echo "Usage:"
  echo
  echo "   bash tools/setup-claude-statusline.sh [options]"
  echo
  echo "Options:"
  echo "     -h, --help    Print this help information"
  echo
  echo "Output (single line):"
  echo "   09:41 | 5h 16% (09:00) | 7d 30% (04/26 20:00) | ctx 33% | Sonnet 4.6 | master ↑1 +2~3"
  echo
  echo "  - Percent text is color-coded: green < 80%, yellow < 90%, red >= 90%"
  echo "  - Rate limits (5h/7d) come first with local reset time"
  echo "  - Git: branch, ahead/behind commits, staged/unstaged file counts"
}

while (($#)); do
  opt="$1"
  case $opt in
  -h | --help)
    help
    exit 0
    ;;
  *)
    echo "> Unknown option: '$opt'"
    echo
    help
    exit 1
    ;;
  esac
done

mkdir -p "$CLAUDE_DIR"

cat > "$STATUSLINE_SCRIPT" << 'EOF'
#!/bin/bash
input=$(cat)

grep_num() {
  echo "$input" | grep -oE "\"$1\":[0-9.]+" | head -1 | grep -oE '[0-9.]+'
}
grep_str() {
  echo "$input" | grep -oE "\"$1\":\"[^\"]*\"" | head -1 | sed "s/\"$1\":\"\([^\"]*\)\"/\1/"
}

# Context
PCT=$(grep_num "used_percentage"); PCT=${PCT:-0}; PCT=${PCT%.*}

# Rate limits + reset times
RATE_5H=$(echo "$input" | grep -oE '"five_hour":\{"used_percentage":[0-9]+' | grep -oE '[0-9]+$')
RATE_7D=$(echo "$input" | grep -oE '"seven_day":\{"used_percentage":[0-9]+' | grep -oE '[0-9]+$')
RESET_5H=$(echo "$input" | grep -oE '"five_hour":\{[^}]+' | grep -oE '"resets_at":[0-9]+' | grep -oE '[0-9]+')
RESET_7D=$(echo "$input" | grep -oE '"seven_day":\{[^}]+' | grep -oE '"resets_at":[0-9]+' | grep -oE '[0-9]+')
RATE_5H=${RATE_5H:-0}; RATE_7D=${RATE_7D:-0}

fmt_reset() {
  local ts=$1
  if [ -z "$ts" ] || [ "$ts" -eq 0 ]; then echo "?"; return; fi
  local now; now=$(date +%s)
  local diff=$((ts - now))
  if [ "$diff" -le 0 ]; then echo "곧"; return; fi
  if [ "$diff" -lt 86400 ]; then
    date -d "@$ts" +"%H:%M" 2>/dev/null || echo "${diff}s"
  else
    date -d "@$ts" +"%m/%d %H:%M" 2>/dev/null || echo "${diff}s"
  fi
}

RESET_5H_FMT=$(fmt_reset "${RESET_5H:-0}")
RESET_7D_FMT=$(fmt_reset "${RESET_7D:-0}")

# Model
MODEL=$(grep_str "display_name"); MODEL=${MODEL:-"Claude"}

# Git info
CWD=$(grep_str "cwd")
CWD_UNIX=$(echo "$CWD" | sed 's/\\/\//g; s/^C:/\/c/')
BRANCH=""
GIT_STAT=""
if cd "$CWD_UNIX" 2>/dev/null && git rev-parse --git-dir &>/dev/null 2>&1; then
  BRANCH=$(git branch --show-current 2>/dev/null || echo "")
  STAGED=$(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')
  UNSTAGED=$(git diff --name-only 2>/dev/null | wc -l | tr -d ' ')
  AHEAD=$(git rev-list --count "@{u}..HEAD" 2>/dev/null || echo 0)
  BEHIND=$(git rev-list --count "HEAD..@{u}" 2>/dev/null || echo 0)

  [ "$AHEAD"    -gt 0 ] && GIT_STAT="${GIT_STAT}↑${AHEAD}"
  [ "$BEHIND"   -gt 0 ] && GIT_STAT="${GIT_STAT}↓${BEHIND}"
  [ "$STAGED"   -gt 0 ] && GIT_STAT="${GIT_STAT} +${STAGED}"
  [ "$UNSTAGED" -gt 0 ] && GIT_STAT="${GIT_STAT} ~${UNSTAGED}"
  GIT_STAT="${GIT_STAT# }"
fi

BRANCH_PART=""
[ -n "$BRANCH" ] && BRANCH_PART=" | ${BRANCH}${GIT_STAT:+ ${GIT_STAT}}"

R='\033[0m'

# 색상: >=90 빨강, >=80 노랑, 그 외 초록
color_for() {
  local p=$1
  if   [ "$p" -ge 90 ]; then echo '\033[31m'
  elif [ "$p" -ge 80 ]; then echo '\033[33m'
  else                       echo '\033[32m'
  fi
}

C_5H=$(color_for "$RATE_5H")
C_7D=$(color_for "$RATE_7D")
C_CTX=$(color_for "$PCT")

NOW=$(date +"%H:%M")

# 한 줄: 시각 / 5h / 7d / ctx / 모델 / git
printf "%s | 5h ${C_5H}%s%%${R} (%s) | 7d ${C_7D}%s%%${R} (%s) | ctx ${C_CTX}%s%%${R} | ${MODEL}${BRANCH_PART}\n" \
  "$NOW" \
  "$RATE_5H" "$RESET_5H_FMT" \
  "$RATE_7D" "$RESET_7D_FMT" \
  "$PCT"
EOF

chmod +x "$STATUSLINE_SCRIPT"
echo "> Written: $STATUSLINE_SCRIPT"

# Patch settings.json
if [ ! -f "$SETTINGS_FILE" ]; then
  echo "{}" > "$SETTINGS_FILE"
fi

if command -v jq &>/dev/null; then
  PATCHED=$(jq --arg cmd "bash $STATUSLINE_SCRIPT" \
    '.statusLine = {"type": "command", "command": $cmd}' \
    "$SETTINGS_FILE")
  echo "$PATCHED" > "$SETTINGS_FILE"
else
  if grep -q '"statusLine"' "$SETTINGS_FILE"; then
    sed -i "s|\"statusLine\".*|\"statusLine\": {\"type\": \"command\", \"command\": \"bash $STATUSLINE_SCRIPT\"},|" "$SETTINGS_FILE"
  else
    sed -i "s|}$|,\n  \"statusLine\": {\"type\": \"command\", \"command\": \"bash $STATUSLINE_SCRIPT\"}\n}|" "$SETTINGS_FILE"
  fi
fi

echo "> Patched: $SETTINGS_FILE"
echo
echo "> Done. Restart Claude Code to see the status line."
