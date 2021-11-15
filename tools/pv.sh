#!/bin/bash
#
# Fetch Google Analytics Pageviews reporting cache
# and save as 'assets/data/pagevies.json'
#
# Requirement:
#   - jq
#   - wget
#
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2019 Cotes Chung
# MIT Licensed


set -eu

WORK_DIR=$(dirname $(dirname $(realpath "$0")))
URL_FILE=${WORK_DIR}/assets/data/proxy.json
PV_CACHE=${WORK_DIR}/assets/data/pageviews.json


PROXY_URL=$(jq -r '.proxyUrl' $URL_FILE)

wget $PROXY_URL -O $PV_CACHE
