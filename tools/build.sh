#!/bin/bash
#
# Build jekyll site and store site files in ./_site
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2019 Cotes Chung
# Published under MIT License

set -eu

CMD="JEKYLL_ENV=production bundle exec jekyll b"

WORK_DIR=$(dirname $(dirname $(realpath "$0")))

CONTAINER=${WORK_DIR}/.container

DEST=${WORK_DIR}/_site


help() {
  echo "Usage:"
  echo
  echo "   bash build.sh [options]"
  echo
  echo "Options:"
  echo "   -b, --baseurl <URL>      The site relative url that start with slash, e.g. '/project'"
  echo "   -h, --help               Print the help information"
  echo "   -d, --destination <DIR>  Destination directory (defaults to ./_site)"
}


init() {
  cd $WORK_DIR

  if [[ -d $CONTAINER ]]; then
    rm -rf $CONTAINER
  fi

  if [[ -d _site ]]; then
    jekyll clean
  fi

  temp=$(mktemp -d)
  cp -r * $temp
  cp -r .git $temp
  mv $temp $CONTAINER
}


build() {
  cd $CONTAINER

  echo "$ cd $(pwd)"
  python _scripts/py/init_all.py

  CMD+=" -d ${DEST}"
  echo "\$ $CMD"
  eval $CMD
  echo -e "\nBuild success, the site files have been placed in '${DEST}'."

  if [[ -d ${DEST}/.git ]]; then
    if [[ ! -z $(git -C $DEST status -s) ]]; then
      git -C $DEST add .
      git -C $DEST commit -m "[Automation] Update site files." -q
      echo -e "\nPlease push the changes of $DEST to remote master branch.\n"
    fi
  fi

  cd .. && rm -rf $CONTAINER
}


check_unset() {
  if [[ -z ${1:+unset} ]]
  then
    help
    exit 1
  fi
}


main() {
  while [[ $# -gt 0 ]]
  do
    opt="$1"
    case $opt in
      -b|--baseurl)
        check_unset $2
        if [[ $2 == \/* ]]
        then
          CMD+=" -b $2"
        else
          help
          exit 1
        fi
        shift
        shift
        ;;
      -d|--destination)
        check_unset $2
        DEST=$(realpath $2)
        shift;
        shift;
        ;;
      -h|--help)
        help
        exit 0
        ;;
      *) # unknown option
        help
        exit 1
        ;;
    esac
  done

  init
  build
}

main "$@"
