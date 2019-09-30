#!/bin/bash
#
# Fetch Google Analytics Pageviews reporting cache
# and save as 'assets/data/pagevies.json'
#
# Requirement:
#   - jq
#   - wget
#
# © 2019 Cotes Chung
# MIT Licensed


URL_FILE=assets/data/proxy.json
PV_CACHE=assets/data/pageviews.json

set -eu

PROXY_URL=$(jq -r '.proxyUrl' $URL_FILE)

wget $PROXY_URL -O $PV_CACHE
