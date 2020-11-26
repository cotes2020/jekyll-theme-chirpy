#!/usr/bin/env bash
#
# Bump latest version to file `_data/meta.yml`
#
# v2.5.1
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2020 Cotes Chung
# Published under MIT License

set -eu

META_FILE="_data/meta.yml"

bump() {
  _version_field="version: $1"

  if [[ ! -f $META_FILE ]]; then
    echo "name: Chirpy" > $META_FILE
    echo "$_version_field" >> $META_FILE
    echo "homepage: https://github.com/cotes2020/jekyll-theme-chirpy/" >> $META_FILE
  else
    sed -i "s/^version:.*/$_version_field/g" $META_FILE
  fi

  if [[ -n $(git status $META_FILE -s) ]]; then
    git add $META_FILE
    git commit -m "Bump version to $1"
  fi
}

_latest_tag="$(git describe --tags --abbrev=0)"

echo "Input a version number (hint: latest version is ${_latest_tag:1})"

read version

if [[ $version =~ ^[[:digit:]]+\.[[:digit:]]+(\.[[:digit:]]+)?$ ]]; then

  if git tag --list | egrep -q "^v$version$"; then
    echo "Error: version '$version' already exists"
    exit -1
  fi

  echo "Bump version to $version"
  bump "$version"

  echo "Create tag v$version"
  git tag "v$version"

else

  echo "Error: Illegal version number: '$version'"
fi
