#!/bin/bash
set -e

GEMFILE="Gemfile"
GEM_LINE="gem 'jekyll-admin', group: :jekyll_plugins"

# Add the gem line if not already present
if ! grep -Fxq "$GEM_LINE" "$GEMFILE"; then
  echo "$GEM_LINE" >>"$GEMFILE"
fi

# Install gems
bundle install

# Serve the Jekyll site
bundle exec jekyll serve
