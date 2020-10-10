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

WORK_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

CONTAINER="${WORK_DIR}/.container"
SYNC_TOOL=_scripts/sh/sync_monitor.sh

cmd="bundle exec jekyll s"
JEKYLL_DOCKER_HOME="/srv/jekyll"

realtime=false
docker=false

_help() {
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
  echo "         --docker            Run within docker"
}

_cleanup() {
  rm -rf "$CONTAINER"
  ps aux | grep fswatch | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1
}

_setup_docker() {
  # docker image `jekyll/jekyll` based on Alpine Linux
  echo "http://dl-cdn.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
  ## CN Apline sources mirror
  # sed -i 's/dl-cdn.alpinelinux.org/mirrors.ustc.edu.cn/g' /etc/apk/repositories
  apk update
  apk add yq
}

_init() {
  cd "$WORK_DIR"

  if [[ -f Gemfile.lock ]]; then
    rm -f Gemfile.lock
  fi

  if [[ -d $CONTAINER ]]; then
    rm -rf "$CONTAINER"
  fi

  mkdir "$CONTAINER"
  cp -r ./* "$CONTAINER"
  cp -r ./.git "$CONTAINER"

  if $docker; then
    local _image_user=$(stat -c "%U" "$JEKYLL_DOCKER_HOME"/.)

    if [[ $_image_user != $(whoami) ]]; then
      # under Docker for Linux
      chown -R "$(stat -c "%U:%G" "$JEKYLL_DOCKER_HOME"/.)" "$CONTAINER"
    fi

  fi

  trap _cleanup INT
}

_check_unset() {
  if [[ -z ${1:+unset} ]]; then
    _help
    exit 1
  fi
}

_check_command() {
  if [[ -z $(command -v "$1") ]]; then
    echo "Error: command '$1' not found !"
    echo "Hint: Get '$1' on <$2>"
    exit 1
  fi
}

_run() {
  cd "$CONTAINER"
  bash _scripts/sh/create_pages.sh
  bash _scripts/sh/dump_lastmod.sh

  if $realtime; then

    exclude_regex="\/\..*"

    if [[ $OSTYPE == "darwin"* ]]; then
      exclude_regex="/\..*" # darwin gcc treat regex '/' as character '/'
    fi

    fswatch -e "$exclude_regex" -0 -r \
      --event Created --event Removed \
      --event Updated --event Renamed \
      --event MovedFrom --event MovedTo \
      "$WORK_DIR" | xargs -0 -I {} bash "./${SYNC_TOOL}" {} "$WORK_DIR" . &
  fi

  if $docker; then
    cmd+=" -H 0.0.0.0"
  else
    cmd+=" -l -o"
  fi

  echo "\$ $cmd"
  eval "$cmd"
}

main() {
  if $docker; then
    _setup_docker
  fi

  _init
  _run
}

while (($#)); do
  opt="$1"
  case $opt in
    -H | --host)
      _check_unset "$2"
      cmd+=" -H $2"
      shift # past argument
      shift # past value
      ;;
    -P | --port)
      _check_unset "$2"
      cmd+=" -P $2"
      shift
      shift
      ;;
    -b | --baseurl)
      _check_unset "$2"
      if [[ $2 == \/* ]]; then
        cmd+=" -b $2"
      else
        _help
        exit 1
      fi
      shift
      shift
      ;;
    -t | --trace)
      cmd+=" -t"
      shift
      ;;
    -r | --realtime)
      _check_command fswatch "http://emcrisostomo.github.io/fswatch/"
      realtime=true
      shift
      ;;
    --docker)
      docker=true
      shift
      ;;
    -h | --help)
      _help
      exit 0
      ;;
    *)
      # unknown option
      _help
      exit 1
      ;;
  esac
done

main
