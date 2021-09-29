#!/usr/bin/env bash
#
#
# 1. Bump latest version number to the following files:
#
#   - _sass/jekyll-theme-chirpy.scss
#   - _javascript/copyright
#   - assets/js/dist/*.js (will be built by gulp later)
#   - jekyll-theme-chirpy.gemspec
#   - package.json
#
# 2. Then create a commit to automatically save the changes.
#
# Usage:
#
#   Run on the default branch or hotfix branch
#
# Requires: Git, Gulp

set -eu

ASSETS=(
  "_sass/jekyll-theme-chirpy.scss"
  "_javascript/copyright"
)

GEM_SPEC="jekyll-theme-chirpy.gemspec"

NODE_META="package.json"

_check_src() {
  if [[ ! -f $1 && ! -d $1 ]]; then
    echo -e "Error: Missing file \"$1\"!\n"
    exit -1
  fi
}

check() {
  if [[ -n $(git status . -s) ]]; then
    echo "Error: Commit unstaged files first, and then run this tool againt."
    exit -1
  fi

  for i in "${!ASSETS[@]}"; do
    _check_src "${ASSETS[$i]}"
  done

  _check_src "$NODE_META"
  _check_src "$GEM_SPEC"
}

_bump_assets() {
  for i in "${!ASSETS[@]}"; do
    sed -i "s/v[[:digit:]]\+\.[[:digit:]]\+\.[[:digit:]]\+/v$1/" "${ASSETS[$i]}"
  done

  gulp
}

_bump_gemspec() {
  sed -i "s/[[:digit:]]\+\.[[:digit:]]\+\.[[:digit:]]\+/$1/" "$GEM_SPEC"
}

_bump_node() {
  sed -i \
    "s,[\"]version[\"]: [\"][[:digit:]]\+\.[[:digit:]]\+\.[[:digit:]]\+[\"],\"version\": \"$1\"," \
    $NODE_META
}

bump() {
  _bump_assets "$1"
  _bump_gemspec "$1"
  _bump_node "$1"

  if [[ -n $(git status . -s) ]]; then
    git add .
    git commit -m "Bump version to $1"
  fi
}

main() {
  check

  _latest_tag="$(git describe --tags $(git rev-list --tags --max-count=1))"

  echo "Input a version number (hint: latest version is ${_latest_tag:1})"

  read _version

  if [[ $_version =~ ^[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]$ ]]; then

    if git tag --list | egrep -q "^v$_version$"; then
      echo "Error: version '$_version' already exists"
      exit -1
    fi

    echo -e "Bump version to $_version\n"
    bump "$_version"

  else

    echo "Error: Illegal version number: '$_version'"
  fi

}

main
