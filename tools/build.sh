#!/bin/bash
#
# Build jekyll site and store site files in ./_site
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2019 Cotes Chung
# Published under MIT License

set -eu

WORK_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

CONTAINER="${WORK_DIR}/.container"

dest="${WORK_DIR}/_site"

cmd="JEKYLL_ENV=production bundle exec jekyll b"

docker=false

config=""

_help() {
  echo "Usage:"
  echo
  echo "   bash build.sh [options]"
  echo
  echo "Options:"
  echo "  -b, --baseurl     <URL>                  The site relative url that start with slash, e.g. '/project'"
  echo "  -h, --help                               Print the help information"
  echo "  -d, --destination <DIR>                  destination directory (defaults to ./_site)"
  echo "      --docker                             Build site within docker"
  echo "      --config      <CONFIG_a[,CONFIG_b]>  Specify config files"
}

_install_tools() {
  # docker image `jekyll/jekyll` based on Alpine Linux
  echo "http://dl-cdn.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
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

  if [[ -d $dest ]]; then
    bundle exec jekyll clean
  fi

  local _temp="$(mktemp -d)"
  cp -r ./* "$_temp"
  cp -r ./.git "$_temp"
  mv "$_temp" "$CONTAINER"
}

_build() {
  cd "$CONTAINER"
  echo "$ cd $(pwd)"

  bash "_scripts/sh/create_pages.sh"
  bash "_scripts/sh/dump_lastmod.sh"

  cmd+=" -d $dest"

  if [[ -n $config ]]; then
    cmd+=" --config $config"
  fi

  echo "\$ $cmd"
  eval "$cmd"
  echo -e "\nBuild success, the site files have been placed in '${dest}'."

  if [[ -d "${dest}/.git" ]]; then
    if [[ -n $(git -C "$dest" status -s) ]]; then
      git -C "$dest" add .
      git -C "$dest" commit -m "[Automation] Update site files." -q
      echo -e "\nPlease push the changes of $dest to remote master branch.\n"
    fi
  fi

  cd .. && rm -rf "$CONTAINER"
}

_check_unset() {
  if [[ -z ${1:+unset} ]]; then
    _help
    exit 1
  fi
}

main() {
  while [[ $# -gt 0 ]]; do
    opt="$1"
    case $opt in
      -b | --baseurl)
        local _baseurl="$2"
        if [[ -z $_baseurl ]]; then
          _baseurl='""'
        fi
        cmd+=" -b $_baseurl"
        shift
        shift
        ;;
      -d | --destination)
        _check_unset "$2"
        dest="$(realpath "$2")"
        shift
        shift
        ;;
      --docker)
        docker=true
        shift
        ;;
      --config)
        _check_unset "$2"
        config="$(realpath "$2")"
        shift
        shift
        ;;
      -h | --help)
        _help
        exit 0
        ;;
      *) # unknown option
        _help
        exit 1
        ;;
    esac
  done

  if $docker; then
    _install_tools
  fi

  _init
  _build
}

main "$@"
