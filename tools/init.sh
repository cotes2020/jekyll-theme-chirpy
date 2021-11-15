#!/bin/bash
#
# Init the evrionment for new user.
#
# v2.5
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2020 Cotes Chung
# Published under MIT License

set -eu

ACTIONS_WORKFLOW=pages-deploy.yml

help() {
  echo "Usage:"
  echo
  echo "   bash /path/to/init.sh [options]"
  echo
  echo "Options:"
  echo "     --no-gh              Do not deploy to Github."
  echo "     -h, --help           Print this help information."
}

check_init() {
  local _has_inited=false

  if [[ ! -d docs ]]; then
    if [[ ! -d .github ]]; then
      _has_inited=true # --no-gh
    else
      if [[ -f .github/workflows/$ACTIONS_WORKFLOW ]]; then
        # on BSD, the `wc` could contains blank
        local _count="$(find .github/workflows/ -type f -name "*.yml" | wc -l)"
        if [[ ${_count//[[:blank:]]/} == 1 ]]; then
          _has_inited=true
        fi
      fi
    fi
  fi

  if $_has_inited; then
    echo "Already initialized."
    exit 0
  fi
}

init_files() {

  if $_no_gh; then
    rm -rf .github
  else
    mv .github/workflows/$ACTIONS_WORKFLOW.hook .
    rm -rf .github
    mkdir -p .github/workflows
    mv ./${ACTIONS_WORKFLOW}.hook .github/workflows/${ACTIONS_WORKFLOW}
  fi

  rm -f .travis.yml
  rm -rf _posts/* docs

  git add -A && git add .github -f
  git commit -m "[Automation] Initialize the environment." -q

  echo "[INFO] Initialization successful!"
}

check_init

_no_gh=false

while (($#)); do
  opt="$1"
  case $opt in
    --no-gh)
      _no_gh=true
      shift
      ;;
    -h | --help)
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

init_files
