#!/usr/bin/env bash
#
# Init the evrionment for new user.

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

check_status() {
  if [[ -n $(git status . -s) ]]; then
    echo "Error: Commit unstaged files first, and then run this tool againt."
    exit -1
  fi
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
    # change the files of `.github`
    mv .github/workflows/$ACTIONS_WORKFLOW.hook .
    rm -rf .github
    mkdir -p .github/workflows
    mv ./${ACTIONS_WORKFLOW}.hook .github/workflows/${ACTIONS_WORKFLOW}

    # ensure the gh-actions trigger branch

    _workflow=".github/workflows/${ACTIONS_WORKFLOW}"
    _default_branch="$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@')"
    _lineno="$(sed -n "/branches:/=" "$_workflow")"
    sed -i "$((_lineno + 1))s/- .*/- ${_default_branch}/" "$_workflow"

  fi

  # trace the gem lockfile on user-end
  sed -i "/Gemfile.lock/d" .gitignore

  # remove the other fies
  rm -f .travis.yml
  rm -rf _posts/* docs

  # save changes
  git add -A && git add .github -f
  git commit -m "[Automation] Initialize the environment." -q

  echo "[INFO] Initialization successful!"
}

check_status

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
