#!/usr/bin/env bash
#
# Release a new version to the GitLab flow production branch.
#
# For a new major/minor version, bump version on the main branch, and then merge into the production branch.
#
# For a patch version, bump the version number on the patch branch, then merge that branch into the main branch
# and production branch.
#
#
# Usage: run on the default, release or the patch branch
#
# Requires: Git, NPM and RubyGems

set -eu

opt_pre=false # preview mode option

working_branch="$(git branch --show-current)"

DEFAULT_BRANCH="$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@')"

PROD_BRANCH="production"

GEM_SPEC="jekyll-theme-chirpy.gemspec"
NODE_CONFIG="package.json"
CHANGE_LOG="docs/CHANGELOG.md"

JS_DIST="assets/js/dist"
BACKUP_PATH="$(mktemp -d)"

FILES=(
  "$GEM_SPEC"
  "$NODE_CONFIG"
)

TOOLS=(
  "git"
  "npm"
  "standard-version"
  "gem"
)

help() {
  echo "A tool to release new version Chirpy gem"
  echo
  echo "Usage:"
  echo
  echo "   bash ./tools/release [options]"
  echo
  echo "Options:"
  echo "     -p, --preview            Enable preview mode, only package, and will not modify the branches"
  echo "     -h, --help               Print this information."
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
  # ensure that changes have been committed
  if [[ -n $(git status . -s) ]]; then
    echo "> Abort: Commit the staged files first, and then run this tool again."
    exit 1
  fi

  if [[ $working_branch != "$DEFAULT_BRANCH" &&
    $working_branch != hotfix/* &&
    $working_branch != "$PROD_BRANCH" ]]; then
    echo "> Abort: Please run on the default, release or patch branch."
    exit 1
  fi
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

_check_node_packages() {
  if [[ ! -d node_modules || "$(du node_modules | awk '{print $1}')" == "0" ]]; then
    npm i
  fi
}

check() {
  _check_cli
  _check_git
  _check_src
  _check_node_packages
}

# Auto-generate a new version number to the file 'package.json'
bump_node() {
  bump="standard-version -i $CHANGE_LOG"

  if $opt_pre; then
    bump="$bump -p rc"
  fi

  eval "$bump"

  # Change heading of Patch version to heading level 2 (a bug from `standard-version`)
  sed -i "s/^### \[/## \[/g" "$CHANGE_LOG"
  # Replace multiple empty lines with a single empty line
  sed -i "/^$/N;/^\n$/D" "$CHANGE_LOG"
}

## Bump new version to gem config file
bump_gem() {
  _ver="$1"

  if $opt_pre; then
    _ver="${1/-/.}"
  fi

  sed -i "s/[[:digit:]]\+\.[[:digit:]]\+\.[[:digit:]]\+/$_ver/" "$GEM_SPEC"
}

# Creates a new tag on the production branch with the given version number.
# Also commits the changes and merges the production branch into the default branch.
branch() {
  _version="$1" # X.Y.Z

  git add .
  git commit -m "chore(release): $_version"

  # Create a new tag on production branch
  echo -e "> Create tag v$_version\n"
  git tag "v$_version"

  git checkout "$DEFAULT_BRANCH"
  git merge --no-ff --no-edit "$PROD_BRANCH"

  if [[ $working_branch == hotfix/* ]]; then
    # delete the patch branch
    git branch -D "$working_branch"
  fi
}

## Build a gem package
build_gem() {
  # Remove unnecessary theme settings
  sed -i "s/^img_cdn:.*/img_cdn:/;s/^avatar:.*/avatar:/" _config.yml
  rm -f ./*.gem

  npm run build
  git add "$JS_DIST" -f # add JS dist to gem
  gem build "$GEM_SPEC"
  cp "$JS_DIST"/* "$BACKUP_PATH"

  # Resume the settings
  git reset
  git checkout .

  # restore the dist files for future development
  mkdir -p "$JS_DIST" && cp "$BACKUP_PATH"/* "$JS_DIST"
}

main() {
  check

  if [[ $opt_pre = false && $working_branch != "$PROD_BRANCH" ]]; then
    git checkout "$PROD_BRANCH"
    git merge --no-ff --no-edit "$working_branch"
  fi

  bump_node

  _version="$(grep '"version":' "$NODE_CONFIG" | sed 's/.*: "//;s/".*//')"

  bump_gem "$_version"

  if [[ $opt_pre = false ]]; then
    branch "$_version"
  fi

  echo -e "> Build the gem package for v$_version\n"

  build_gem
}

while (($#)); do
  opt="$1"
  case $opt in
  -p | --preview)
    opt_pre=true
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
