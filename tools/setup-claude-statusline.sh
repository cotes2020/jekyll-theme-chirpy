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
  echo "Output (2 lines):"
  echo "   Sonnet 4.6 | master ↑1 +2~3 | 세션 2m36s | Cost \$0.2699 (\$6.20/hr) | Cache 96% | 09:41"
  echo "   [███░░░░░░░] 65,847in/216out (33%) | 5h 16% (리셋 09:00) | 7d 30% (리셋 04/26 20:00)"
  echo
  echo "  - Progress bar is color-coded: green < 80%, yellow < 90%, red >= 90%"
  echo "  - Git: branch, ahead/behind commits, staged/unstaged file counts"
  echo "  - Cache hit rate: how much of context is served from cache (lower cost)"
  echo "  - Rate limits: 5h/7d usage % with local reset time"
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
PCT=$(grep_num "used_percentage");              PCT=${PCT:-0}; PCT=${PCT%.*}
MAX=$(grep_num "context_window_size");           MAX=${MAX:-200000}
T_INPUT=$(grep_num "input_tokens");              T_INPUT=${T_INPUT:-0}
T_CC=$(grep_num "cache_creation_input_tokens");  T_CC=${T_CC:-0}
T_CR=$(grep_num "cache_read_input_tokens");      T_CR=${T_CR:-0}
T_OUT=$(grep_num "output_tokens");               T_OUT=${T_OUT:-0}
USED=$((T_INPUT + T_CC + T_CR))

# Cache hit rate
CACHE_TOTAL=$((T_INPUT + T_CC + T_CR))
if [ "$CACHE_TOTAL" -gt 0 ]; then
  CACHE_HIT=$(awk "BEGIN{printf \"%d\", $T_CR/$CACHE_TOTAL*100}")
else
  CACHE_HIT=0
fi

# Cost & duration
COST=$(grep_num "total_cost_usd");               COST=${COST:-0}
DURATION_MS=$(grep_num "total_duration_ms");     DURATION_MS=${DURATION_MS:-0}

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

# Burn rate & elapsed
BURN=$(awk "BEGIN{if($DURATION_MS>0) printf \"%.4f\", $COST/($DURATION_MS/3600000); else print \"0.0000\"}")
ELAPSED_S=$((DURATION_MS / 1000))
ELAPSED_H=$((ELAPSED_S / 3600))
ELAPSED_M=$(( (ELAPSED_S % 3600) / 60 ))
ELAPSED_SEC=$((ELAPSED_S % 60))
if [ "$ELAPSED_H" -gt 0 ]; then
  ELAPSED=$(printf "%dh%02dm" "$ELAPSED_H" "$ELAPSED_M")
else
  ELAPSED=$(printf "%dm%02ds" "$ELAPSED_M" "$ELAPSED_SEC")
fi

# Color by context % (bar only)
if   [ "$PCT" -ge 90 ]; then C='\033[31m'
elif [ "$PCT" -ge 80 ]; then C='\033[33m'
else                          C='\033[32m'
fi
R='\033[0m'

# Progress bar
FILLED=$((PCT / 10)); EMPTY=$((10 - FILLED))
printf -v F "%${FILLED}s"; printf -v P "%${EMPTY}s"
BAR="${F// /█}${P// /░}"

fmt() {
  local n=$1
  if   [ "$n" -ge 1000000 ]; then awk "BEGIN{printf \"%.1fM\", $n/1000000}"
  elif [ "$n" -ge 1000 ];    then awk "BEGIN{printf \"%.1fK\", $n/1000}"
  else echo "$n"
  fi
}

COST_FMT=$(awk "BEGIN{printf \"%.4f\", $COST}")
NOW=$(date +"%H:%M")

# Line 1: 모델 / git / 세션 시간 / 비용 / 캐시 / 현재 시각
printf "${MODEL}${BRANCH_PART} | 세션 %s | Cost \$%s (\$%s/hr) | Cache %s%% | %s\n" \
  "$ELAPSED" "$COST_FMT" "$BURN" "$CACHE_HIT" "$NOW"

# Line 2: 컨텍스트 바 / 토큰(in+out) / 할당량 + 리셋 시각
printf "[${C}%s${R}] %s/%s in | %s out (%s%%) | 5h %s%% (리셋 %s) | 7d %s%% (리셋 %s)\n" \
  "$BAR" "$(fmt $USED)" "$(fmt $MAX)" "$(fmt $T_OUT)" "$PCT" \
  "$RATE_5H" "$RESET_5H_FMT" "$RATE_7D" "$RESET_7D_FMT"
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
