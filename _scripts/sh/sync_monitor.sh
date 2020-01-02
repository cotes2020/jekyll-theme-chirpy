#!/bin/bash

# Files sync monitoer
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2019 Cotes Chung
# MIT Licensed

# $1 -> the origin file with absolute path.
# $2 -> the origin sync directory
# $3 -> the destination sync direcotry

# Omit the system temp file
if [[ ! -f $1 ]]; then
  exit 0
fi

src_dir=`dirname $(realpath $1)`

dir_prefix="$(realpath $2)/"

related_dir="${src_dir:${#dir_prefix}}"


dest="$(realpath $3)/${related_dir}"

if [[ ! -d "$dest" ]]; then
  mkdir -p "$dest"
fi

if [[ -f "$1" ]]; then
  cp $1 $dest
fi

if [[ $related_dir == "_posts" ]]; then
  python $3/_scripts/py/init_all.py
  python $3/_scripts/py/update_posts_lastmod.py -f "$dest/$(basename $1)" -t fs
fi
