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
  echo "   Sonnet 4.6 (master) | 2m36s | Cost \$0.2699 (\$6.20/hr)"
  echo "   [███░░░░░░░] 65,847/200,000 (33%) | 5h 16% (3h51m후) | 7d 30% (4d19h후)"
  echo
  echo "  - Progress bar is color-coded: green < 80%, yellow < 90%, red >= 90%"
  echo "  - Git branch shown when running inside a git repo"
  echo "  - Rate limits: 5-hour and 7-day usage % with time until reset"
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
USED=$((T_INPUT + T_CC + T_CR))

# Cost & duration
COST=$(grep_num "total_cost_usd");               COST=${COST:-0}
DURATION_MS=$(grep_num "total_duration_ms");     DURATION_MS=${DURATION_MS:-0}

# Rate limits + reset times
RATE_5H=$(echo "$input" | grep -oE '"five_hour":\{"used_percentage":[0-9]+' | grep -oE '[0-9]+$')
RATE_7D=$(echo "$input" | grep -oE '"seven_day":\{"used_percentage":[0-9]+' | grep -oE '[0-9]+$')
RESET_5H=$(echo "$input" | grep -oE '"five_hour":\{[^}]+' | grep -oE '"resets_at":[0-9]+' | grep -oE '[0-9]+')
RESET_7D=$(echo "$input" | grep -oE '"seven_day":\{[^}]+' | grep -oE '"resets_at":[0-9]+' | grep -oE '[0-9]+')
RATE_5H=${RATE_5H:-0}; RATE_7D=${RATE_7D:-0}

fmt_remain() {
  local ts=$1
  if [ -z "$ts" ] || [ "$ts" -eq 0 ]; then echo "?"; return; fi
  local now; now=$(date +%s)
  local diff=$((ts - now))
  if [ "$diff" -le 0 ]; then echo "곧"; return; fi
  local d=$((diff / 86400))
  local h=$(( (diff % 86400) / 3600 ))
  local m=$(( (diff % 3600) / 60 ))
  if   [ "$d" -gt 0 ]; then printf "%dd%02dh후" "$d" "$h"
  elif [ "$h" -gt 0 ]; then printf "%dh%02dm후" "$h" "$m"
  else printf "%dm후" "$m"
  fi
}

REMAIN_5H=$(fmt_remain "${RESET_5H:-0}")
REMAIN_7D=$(fmt_remain "${RESET_7D:-0}")

# Model & git branch
MODEL=$(grep_str "display_name"); MODEL=${MODEL:-"Claude"}
CWD=$(grep_str "cwd")
CWD_UNIX=$(echo "$CWD" | sed 's/\\/\//g; s/^C:/\/c/')
BRANCH=$(cd "$CWD_UNIX" 2>/dev/null && git branch --show-current 2>/dev/null || echo "")

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

fmt() { printf "%d" "$1" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta'; }

BRANCH_PART=${BRANCH:+" ($BRANCH)"}
COST_FMT=$(awk "BEGIN{printf \"%.4f\", $COST}")

# Line 1: 모델 / 브랜치 / 경과 시간 / 비용
printf "${MODEL}${BRANCH_PART} | %s | Cost \$%s (\$%s/hr)\n" \
  "$ELAPSED" "$COST_FMT" "$BURN"

# Line 2: 컨텍스트 바 / 할당량 + 리셋
printf "[${C}%s${R}] %s/%s (%s%%) | 5h %s%% (%s) | 7d %s%% (%s)\n" \
  "$BAR" "$(fmt $USED)" "$(fmt $MAX)" "$PCT" \
  "$RATE_5H" "$REMAIN_5H" "$RATE_7D" "$REMAIN_7D"
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
