#!/bin/bash
#
# A development tool that use yuicompressor to compress JS files.
#
#
# Requirement:
#   - JRE
#   - yuicompressor › https://github.com/yui/yuicompressor
#
#
# Usage: bash /path/to/js-compress.sh
#
# Process:
#     input: /path/to/js/source.js --> output: /path/to/js/dist/source.min.js
#
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2020 Cotes Chung
# MIT Licensed


set -eu

PROJ_HOME=$(dirname $(dirname $(realpath "$0")))

YUICOMPRESSOR=${PROJ_HOME}/tools/package/yuicompressor-2.4.8.jar
JS_ROOT=${PROJ_HOME}/assets/js/
JS_SRC=${JS_ROOT}_src    # JS source files
JS_DEST=${JS_ROOT}dist   # Compressed output directory
PREFIX_LEN=${#JS_ROOT}   # To beautify the log

function compress() {
  # $1 is the source dir
  # $2 is the destination dir
  # $3 is the sub dir of source dir, nullable
  if [[ -z ${3:+unset} ]]
  then
    sub_dir=""
  else
    sub_dir="$3/"
  fi

  for item in $(ls $1)
  do
    src="$1/$item"
    if [[ -d "$src" ]]; then
      compress $src $2 $item   # recursion
    else
      if [[ ! -d "$2/${sub_dir}" ]]; then
        mkdir -p $2/${sub_dir}
      fi
      output=$2/${sub_dir}${item%.*}.min.js
      echo "java -jar $(basename $YUICOMPRESSOR) ${src:$PREFIX_LEN} -o ${output:$PREFIX_LEN}"
      java -jar $YUICOMPRESSOR $src -o $output
    fi
  done

  sub_dir="" # clean up for next recursion.
}

compress $JS_SRC $JS_DEST
