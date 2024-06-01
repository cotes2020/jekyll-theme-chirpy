#!/usr/bin/env bash

setup_node() {
  bash -i -c "nvm install --lts && nvm install-latest-npm"

  [[ -f package-lock.json && -d node_modules ]] || npm i
}

setup_assets() {
  has_built_css=false
  has_built_js=false

  CSS_DIST="_sass/dist"
  JS_DIST="assets/js/dist"

  if [ -d "$CSS_DIST" ]; then
    [ -z "$(ls -A $CSS_DIST)" ] || has_built_css=true
  fi

  if [ -d "$JS_DIST" ]; then
    [ -z "$(ls -A $JS_DIST)" ] || has_built_js=true
  fi

  $has_built_css || npm run build:css
  $has_built_js || npm run build:js
}

if [ -f package.json ]; then
  setup_node
  setup_assets
fi

# Install dependencies for shfmt extension
curl -sS https://webi.sh/shfmt | sh &>/dev/null
