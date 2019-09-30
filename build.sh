#!/bin/bash
#
# Build jekyll site and store site files in ./_site
# © 2019 Cotes Chung
# Published under MIT License


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
  set -eu

  if [[ -d .container ]]; then
    rm -rf .container
  fi

  if [[ -d _site ]]; then
    rm -rf _site
  fi

  if [[ -d ../.chirpy-cache ]]; then
    rm -rf ../.chirpy-cache
  fi

  mkdir ../.chirpy-cache
  cp -r *   ../.chirpy-cache
  cp -r .git ../.chirpy-cache

  mv ../.chirpy-cache .container
}


check_unset() {
  if [[ -z ${1:+unset} ]]
  then
    help
    exit 1
  fi
}


CMD="JEKYLL_ENV=production bundle exec jekyll b"
DEST=`realpath "./_site"`

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
      if [[ -d $2 ]]; then
        rm -rf $2
      fi
      DEST=$2
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

cd .container

echo "$ cd $(pwd)"
python _scripts/py/init_all.py

CMD+=" -d $DEST"
echo "\$ $CMD"
eval $CMD

echo "$(date) - Build success, the Site files placed in _site."

cd .. && rm -rf .container
