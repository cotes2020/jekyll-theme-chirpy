#!/bin/bash
#
# Initial the categories/tags pages and for posts.
# © 2019 Cotes Chung
# Published under MIT License


CATEGORIES=false
TAGS=false
LASTMOD=false

set -eu

if [[ ! -z $(git status -s) ]]; then
  echo "Warning: Commit the changes of the repository first."
  git status -s
  exit 1
fi

python _scripts/py/init_all.py

msg="Updated"

if [[ ! -z $(git status categories -s) ]]; then
  git add categories/
  msg+=" Categories' pages"
  CATEGORIES=true
fi


if [[ ! -z $(git status tags -s) ]]; then
  git add tags/
  if [[ $CATEGORIES = true ]]; then
    msg+=" and"
  fi
  msg+=" Tags' pages"
  TAGS=true
fi

if [[ ! -z $(git status _posts -s) ]]; then
  git add _posts/
  if [[ $CATEGORIES = true || $TAGS = true ]]; then
    msg+=" and"
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
