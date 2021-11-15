#!/bin/bash

# Run jekyll site at http://127.0.0.1:4000
#
# Requirement:
#   Option '-r, --realtime' needs fswatch › http://emcrisostomo.github.io/fswatch/
#
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2019 Cotes Chung
# Published under MIT License

set -eu

WORK_DIR=$(dirname $(dirname $(realpath "$0")))

CONTAINER=.container
SYNC_TOOL=_scripts/sh/sync_monitor.sh

cmd="bundle exec jekyll s"
realtime=false


help() {
  echo "Usage:"
  echo
  echo "   bash run.sh [options]"
  echo
  echo "Options:"
  echo "     -H, --host    <HOST>    Host to bind to"
  echo "     -P, --port    <PORT>    Port to listen on"
  echo "     -b, --baseurl <URL>     The site relative url that start with slash, e.g. '/project'"
  echo "     -h, --help              Print the help information"
  echo "     -t, --trace             Show the full backtrace when an error occurs"
  echo "     -r, --realtime          Make the modified content updated in real time"
}


cleanup() {
  rm -rf ${WORK_DIR}/${CONTAINER}
  ps aux | grep fswatch | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1
}


init() {

  if [[ -d ${WORK_DIR}/${CONTAINER} ]]; then
    rm -rf ${WORK_DIR}/${CONTAINER}
  fi

  temp=$(mktemp -d)
  cp -r ${WORK_DIR}/* $temp
  cp -r ${WORK_DIR}/.git $temp
  mv $temp ${WORK_DIR}/${CONTAINER}

  trap cleanup INT
}


check_unset() {
  if [[ -z ${1:+unset} ]]; then
    help
    exit 1
  fi
}


check_command() {
  if [[ -z $(command -v $1) ]]; then
    echo "Error: command '$1' not found !"
    echo "Hint: Get '$1' on <$2>"
    exit 1
  fi
}


main() {
  init

  cd ${WORK_DIR}/${CONTAINER}
  python _scripts/py/init_all.py

  if [[ $realtime = true ]]; then
    fswatch -0 -e "\\$CONTAINER" -e "\.git" ${WORK_DIR} | xargs -0 -I {} bash ./${SYNC_TOOL} {} $WORK_DIR . &
  fi

  echo "\$ $cmd"
  eval $cmd
}


while (( $# ))
do
  opt="$1"
  case $opt in
    -H|--host)
      check_unset $2
      cmd+=" -H $2"
      shift # past argument
      shift # past value
      ;;
    -P|--port)
      check_unset $2
      cmd+=" -P $2"
      shift
      shift
      ;;
    -b|--baseurl)
      check_unset $2
      if [[ $2 == \/* ]]
      then
        cmd+=" -b $2"
      else
        help
        exit 1
      fi
      shift
      shift
      ;;
    -t|--trace)
      cmd+=" -t"
      shift
      ;;
    -r|--realtime)
      check_command fswatch 'http://emcrisostomo.github.io/fswatch/'
      realtime=true
      shift
      ;;
    -h|--help)
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
