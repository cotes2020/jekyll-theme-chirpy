#!/bin/bash
#
# Initial the Categories/Tags pages and Lastmod for posts.
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2019 Cotes Chung
# Published under MIT License

set -eu

CATEGORIES=false
TAGS=false
LASTMOD=false

WORK_DIR=$(dirname $(dirname $(realpath "$0")))

check_status() {
  local _watching_dirs=(
    "_post"
    "_data")

  for i in "${!_watching_dirs[@]}"
  do
    local _dir=${_watching_dirs[${i}]}
    if [[ ! -z $(git status $_dir -s) ]]; then
      echo "Warning: Commit the changes of the directory '$_dir' first."
      git status -s | grep $_dir
      exit 1
    fi
  done
}


update_files() {
  bash _scripts/sh/create_pages.sh
  bash _scripts/sh/dump_lastmod.sh

  find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
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
    if [[ $CATEGORIES = true ]]; then
      msg+=","
    else
      msg+=" the"
    fi
    msg+=" Tags"
    TAGS=true
  fi

  if [[ ! -z $(git status _data -s) ]]; then
    git add _data
    if [[ $CATEGORIES = true || $TAGS = true ]]; then
      msg+=","
    else
      msg+=" the"
    fi
    msg+=" Lastmod"
    LASTMOD=true
  fi

  if [[ $CATEGORIES = true || $TAGS = true || $LASTMOD = true ]]; then
    msg+=" for post(s)."
    git commit -m "[Automation] $msg"
  else
    msg="Nothing changed."
  fi

  echo $msg
}


main() {

  cd $WORK_DIR

  check_status

  update_files

  commit
}

main
