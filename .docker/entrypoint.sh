#!/usr/bin/env sh

bundle install
if [ "$1" = "serve" ]; then
  bash _scripts/sh/create_pages.sh
  bash _scripts/sh/dump_lastmod.sh
  eval "bundle exec jekyll serve --open-url --watch --livereload --incremental --host 0.0.0.0 --port $APP_PORT"
else
  eval "$@"
fi
