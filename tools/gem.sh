#!/usr/bin/env bash
#
# Create a ruby-gem pakcage with doc
#
# Required: rubygem, yard

set -eu

GEM_CONFIG="./jekyll-theme-chirpy.gemspec"

if [[ ! -f $GEM_CONFIG ]]; then
  echo "Error: .gemspec file not found!"
  exit -1
fi

if [[ -d doc ]]; then
  rm -rf doc
fi

yard doc

gem build "$GEM_CONFIG"
