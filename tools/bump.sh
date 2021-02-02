#!/usr/bin/env bash
#
# 1. Bump latest version number to files:
#   - _sass/jekyll-theme-chirpy.scss
#   - assets/js/.copyright.js
#   - assets/js/dist/*.js (will be built by gulp later)
#   - jekyll-theme-chirpy.gemspec
#   - package.json
#
# 2. Create a git-tag
#
# 3. Build a rubygem package
#
# Requires: gulp, rubygem

set -eu

ASSETS=(
  "_sass/jekyll-theme-chirpy.scss"
  "assets/js/.copyright"
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

  # ensure the current branch is 'master'
  if [[ "$(git branch --show-current)" != "master" ]]; then
    echo "Error: This operation must be performed on the 'master' branch!"
    exit -1
  fi

  for i in "${!ASSETS[@]}"; do
    _check_src "${ASSETS[$i]}"
  done

  _check_src "$NODE_META"
  _check_src "$GEM_SPEC"
}

_bump_assets() {
  _version="$1"
  for i in "${!ASSETS[@]}"; do
    sed -i "s/v[[:digit:]]\.[[:digit:]]\.[[:digit:]]/v$_version/" "${ASSETS[$i]}"
  done

  gulp
}

_bump_gemspec() {
  sed -i "s/[[:digit:]]\.[[:digit:]]\.[[:digit:]]/$1/" "$GEM_SPEC"
}

_bump_node() {
  sed -i \
    "s,[\"]version[\"]: [\"][[:digit:]]\.[[:digit:]]\.[[:digit:]][\"],\"version\": \"$1\"," \
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

build_gem() {
  rm -f ./*.gem
  gem build "$GEM_SPEC"
}

release() {
  _version="$1"
  _major=""
  _minor=""

  IFS='.' read -r -a array <<< "$_version"

  for elem in "${array[@]}"; do
    if [[ -z $_major ]]; then
      _major="$elem"
    elif [[ -z $_minor ]]; then
      _minor="$elem"
    else
      break
    fi
  done

  _release_branch="$_major-$_minor-stable"

  if [[ -z $(git branch -v | grep "$_release_branch") ]]; then
    git checkout -b "$_release_branch"
  else
    git checkout "$_release_branch"
    # cherry-pick the latest 2 commit from master to release branch
    git cherry-pick "$(git rev-parse master~1)" "$(git rev-parse master)"
  fi

  echo -e "Create tag v$_version\n"
  git tag "v$_version"

  build_gem

  # head back to master branch
  git checkout master

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

    echo -e "Release to v$_version\n"
    release "$_version"

  else
    echo "Error: Illegal version number: '$_version'"
  fi

}

main
