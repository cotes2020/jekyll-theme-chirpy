#!/bin/bash

# Files sync monitor
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2019 Cotes Chung
# MIT Licensed

# $1 -> the origin file with absolute path.
# $2 -> the origin sync directory
# $3 -> the destination sync directory

# Omit the system temp file
if [[ ! -f $1 ]]; then
  exit 0
fi

src_dir="$(dirname "$(realpath "$1")")"

dir_prefix="$(realpath "$2")/"

related_dir="${src_dir:${#dir_prefix}}"

dest="$(realpath "$3")/${related_dir}"

if [[ ! -d $dest ]]; then
  mkdir -p "$dest"
fi

if [[ -f $1 ]]; then
  cp "$1" "$dest"
fi

if [[ $related_dir == "_posts" ]]; then
  bash "$3"/_scripts/sh/create_pages.sh
  bash "$3"/_scripts/sh/dump_lastmod.sh
fi
