#!/usr/bin/env bash
#
# Requires: Git, NPM and RubyGems

set -eu

opt_pre=false # option for bump gem version
opt_pkg=false # option for building gem package

MAIN_BRANCH="master"
RELEASE_BRANCH="production"

GEM_SPEC="jekyll-theme-chirpy.gemspec"
NODE_SPEC="package.json"
CHANGELOG="docs/CHANGELOG.md"
CONFIG="_config.yml"

CSS_DIST="_sass/dist"
JS_DIST="assets/js/dist"
PWA_DIST="_app"

FILES=(
  "$GEM_SPEC"
  "$NODE_SPEC"
  "$CHANGELOG"
  "$CONFIG"
)

TOOLS=(
  "git"
  "npm"
  "gem"
)

help() {
  echo -e "A tool to release new version Chirpy gem.\nThis tool will:"
  echo "  1. Build a new gem and publish it to RubyGems.org"
  echo "  2. Merge the release branch into the default branch"
  echo
  echo "Usage:"
  echo "  bash $0 [options]"
  echo
  echo "Options:"
  echo "  --prepare           Preparation for release"
  echo "  -p, --package       Build a gem package only, for local packaging in case of auto-publishing failure"
  echo "  -h, --help          Display this help message"
}

_check_cli() {
  for i in "${!TOOLS[@]}"; do
    cli="${TOOLS[$i]}"
    if ! command -v "$cli" &>/dev/null; then
      echo "> Command '$cli' not found!"
      exit 1
    fi
  done
}

_check_git() {
  $opt_pre || (
    # ensure that changes have been committed
    if [[ -n $(git status . -s) ]]; then
      echo "> Abort: Commit the staged files first, and then run this tool again."
      exit 1
    fi
  )

  $opt_pkg || (
    if [[ "$(git branch --show-current)" != "$RELEASE_BRANCH" ]]; then
      echo "> Abort: Please run the tool in the '$RELEASE_BRANCH' branch."
      exit 1
    fi
  )
}

_check_src() {
  for i in "${!FILES[@]}"; do
    _src="${FILES[$i]}"
    if [[ ! -f $_src && ! -d $_src ]]; then
      echo -e "> Error: Missing file \"$_src\"!\n"
      exit 1
    fi
  done
}

init() {
  _check_cli
  _check_git
  _check_src
  echo -e "> npm install\n"
  npm i
}

## Bump new version to gem-spec file
_bump_version() {
  _version="$(grep '"version":' "$NODE_SPEC" | sed 's/.*: "//;s/".*//')"
  sed -i "s/[[:digit:]]\+\.[[:digit:]]\+\.[[:digit:]]\+/$_version/" "$GEM_SPEC"
  echo "> Bump gem version to $_version"
}

_improve_changelog() {
  # Replace multiple empty lines with a single empty line
  sed -i '/^$/N;/^\n$/D' "$CHANGELOG"
  # Escape left angle brackets of HTML tag in the changelog as they break the markdown structure. e.g., '<hr>'
  sed -i -E 's/\s(<[a-z])/ \\\1/g' "$CHANGELOG"
}

prepare() {
  _bump_version
  _improve_changelog
}

## Build a Gem package
build_gem() {
  # Remove unnecessary theme settings
  sed -i -E "s/(^timezone:).*/\1/;s/(^cdn:).*/\1/;s/(^avatar:).*/\1/" $CONFIG
  rm -f ./*.gem

  npm run build
  # add CSS/JS distribution files to gem package
  git add "$CSS_DIST" "$JS_DIST" "$PWA_DIST" -f

  echo -e "\n> gem build $GEM_SPEC\n"
  gem build "$GEM_SPEC"

  echo -e "\n> Resume file changes ...\n"
  git reset
  git checkout .
}

# Push the gem to RubyGems.org (using $GEM_HOST_API_KEY)
push_gem() {
  gem push ./*.gem
}

## Merge the release branch into the default branch
merge() {
  git fetch origin "$MAIN_BRANCH"
  git checkout -b "$MAIN_BRANCH" origin/"$MAIN_BRANCH"

  git merge --no-ff --no-edit "$RELEASE_BRANCH" || (
    git merge --abort
    echo -e "\n> Conflict detected. Aborting merge.\n"
    exit 0
  )

  git push origin "$MAIN_BRANCH"
}

main() {
  init

  if $opt_pre; then
    prepare
    exit 0
  fi

  build_gem
  $opt_pkg && exit 0
  push_gem
  merge
}

while (($#)); do
  opt="$1"
  case $opt in
  --prepare)
    opt_pre=true
    shift
    ;;
  -p | --package)
    opt_pkg=true
    shift
    ;;
  -h | --help)
    help
    exit 0
    ;;
  *)
    # unknown option
    help
    exit 1
    ;;
  esac
done

main
