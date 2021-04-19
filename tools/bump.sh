#!/usr/bin/env bash
#
# How does it work:
#
#   1. Bump latest version number to files:
#     - _sass/jekyll-theme-chirpy.scss
#     - _javascript/copyright
#     - assets/js/dist/*.js (will be built by gulp later)
#     - jekyll-theme-chirpy.gemspec
#     - package.json
#
#   2. Create a git-tag on release branch
#
#   3. Build a RubyGems package base on the latest git-tag
#
#
# Usage:
#
#   Switch to 'master' branch or 'X-Y-stable' branch with argument '-m',
#`  and then run this script.
#
#
# Requires: Git, Gulp, RubyGems

set -eu

manual_release=false

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

  # ensure the current branch is 'master' or running in 'manual' mode
  if [[ "$(git branch --show-current)" != "master" && $manual_release == "false" ]]; then
    echo "Error: This operation must be performed on the 'master' branch or '--manual' mode!"
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

  if $manual_release; then
    echo -e "Bump version to $_version (manual release)\n"
    bump "$_version"
    exit 0
  fi

  if [[ -z $(git branch -v | grep "$_release_branch") ]]; then
    git checkout -b "$_release_branch"
  else
    git checkout "$_release_branch"
    # cherry-pick the latest commit from master branch to release branch
    git cherry-pick "$(git rev-parse master)"
  fi

  echo -e "Bump version to $_version\n"
  bump "$_version"

  echo -e "Create tag v$_version\n"
  git tag "v$_version"

  echo -e "Build the gem pakcage for v$_version\n"
  build_gem

  # head back to master branch
  git checkout master
  # cherry-pick the latest commit from release branch to master branch
  git cherry-pick "$_release_branch" -x

}

help() {
  echo "Bump new version to Chirpy project"
  echo "Usage:"
  echo
  echo "   bash /path/to/bump.sh [options]"
  echo
  echo "Options:"
  echo "     -m, --manual         Manual relase, bump version only."
  echo "     -h, --help           Print this help information."
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

    release "$_version"

  else

    echo "Error: Illegal version number: '$_version'"
  fi

}

while (($#)); do
  opt="$1"
  case $opt in
    -m | --manual)
      manual_release=true
      shift
      ;;
    -h | --help)
      help
      exit 0
      ;;
    *)
      echo "unknown option '$opt'!"
      exit 1
      ;;
  esac
done

main
