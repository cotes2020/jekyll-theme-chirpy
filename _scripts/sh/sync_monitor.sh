#!/bin/bash

# Files sync monitoer
#
# © 2019 Cotes Chung
# MIT Licensed

# $1 -> the origin filen with absolute path.
# $2 -> the origin sync directory
# $3 -> the destination sync direcotry

# Pass the system temp file
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
  python $3/_scripts/py/pages_generator.py
fi
