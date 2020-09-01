#!/bin/bash
#
# Initial the Categories/Tags pages and Lastmod for posts and then push to remote
#
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2019 Cotes Chung
# Published under MIT License

set -eu

CATEGORIES=false
TAGS=false
LASTMOD=false

WORK_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

check_status() {
  local _change=$(git status . -s)

  if [[ ! -z $_change ]]; then
    echo "Warning: Commit the following changes first:"
    echo "$_change"
    exit 1
  fi
}

update_files() {
  bash _scripts/sh/create_pages.sh
  bash _scripts/sh/dump_lastmod.sh
}

commit() {
  msg="Updated"

  if [[ ! -z $(git status categories -s) ]]; then
    git add categories/
    msg+=" the Categories"
    CATEGORIES=true
  fi

  if [[ ! -z $(git status tags -s) ]]; then
    git add tags/
    if $CATEGORIES; then
      msg+=","
    else
      msg+=" the"
    fi
    msg+=" Tags"
    TAGS=true
  fi

  if [[ -n $(git status _data -s) ]]; then
    git add _data
    if $CATEGORIES || $TAGS; then
      msg+=","
    else
      msg+=" the"
    fi
    msg+=" Lastmod"
    LASTMOD=true
  fi

  if $CATEGORIES || $TAGS || $LASTMOD; then
    msg+=" for post(s)."
    git commit -m "[Automation] $msg" -q
  else
    msg="Nothing changed."
  fi

}

push() {
  git push origin master -q
  echo "[INFO] Published successfully!"
}

main() {

  cd "$WORK_DIR"

  check_status

  update_files

  commit

  push
}

main
