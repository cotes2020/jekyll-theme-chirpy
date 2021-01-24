#!/usr/bin/env bash
#
# Fetch Google Analytics Pageviews reporting cache
# and save as 'assets/data/pagevies.json'
#
# Requirement:
#   - jq
#   - wget

set -eu

WORK_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
URL_FILE="${WORK_DIR}/_config.yml"
PV_CACHE="${WORK_DIR}/assets/js/data/pageviews.json"

PROXY_URL="$(grep "proxy_endpoint:" "$URL_FILE" | sed "s/.*: '//g;s/'.*//")"

wget "$PROXY_URL" -O "$PV_CACHE"

echo "ls $PV_CACHE"
