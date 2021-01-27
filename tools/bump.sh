#!/usr/bin/env bash
#
# Bump latest version to
#   - _sass/jekyll-theme-chirpy.scss
#   - assets/js/_commons/_copyright.js
#   - assets/js/dist/*.js
#   - jekyll-theme-chirpy.gemspec
#
# Required: gulp

set -eu

ASSETS=(
  "_sass/jekyll-theme-chirpy.scss"
  "assets/js/.copyright"
)

GEM_SPEC="jekyll-theme-chirpy.gemspec"

NODE_META="package.json"

bump_assets() {
  _version="$1"
  for i in "${!ASSETS[@]}"; do
    sed -i "s/v[[:digit:]]\.[[:digit:]]\.[[:digit:]]/v$_version/" "${ASSETS[$i]}"
  done

  gulp
}

bump_gemspec() {
  sed -i "s/[[:digit:]]\.[[:digit:]]\.[[:digit:]]/$1/" "$GEM_SPEC"
}

bump_node() {
  sed -i \
    "s,[\"]version[\"]: [\"][[:digit:]]\.[[:digit:]]\.[[:digit:]][\"],\"version\": \"$1\"," \
    $NODE_META
}

bump() {
  bump_assets "$1"
  bump_gemspec "$1"
  bump_node "$1"

  if [[ -n $(git status . -s) ]]; then
    git add .
    git commit -m "Bump version to $1"
  fi
}

main() {
  if [[ -n $(git status . -s) ]]; then
    echo "Warning: commit unstaged files first, and then run this tool againt."
    exit -1
  fi

  _latest_tag="$(git describe --tags --abbrev=0)"

  echo "Input a version number (hint: latest version is ${_latest_tag:1})"

  read _version

  if [[ $_version =~ ^[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]$ ]]; then

    if git tag --list | egrep -q "^v$_version$"; then
      echo "Error: version '$_version' already exists"
      exit -1
    fi

    echo "Bump version to $_version"
    bump "$_version"

    echo "Create tag v$_version"
    git tag "v$_version"

  else

    echo "Error: Illegal version number: '$_version'"
  fi
}

main
