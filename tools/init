#!/usr/bin/env bash
#
# Init the environment for new user.

set -eu

# CLI Dependencies
CLI=("git" "npm")

ACTIONS_WORKFLOW=pages-deploy.yml

# temporary file suffixes that make `sed -i` compatible with BSD and Linux
TEMP_SUFFIX="to-delete"

_no_gh=false

help() {
  echo "Usage:"
  echo
  echo "   bash /path/to/init [options]"
  echo
  echo "Options:"
  echo "     --no-gh              Do not deploy to Github."
  echo "     -h, --help           Print this help information."
}

# BSD and GNU compatible sed
_sedi() {
  regex=$1
  file=$2
  sed -i.$TEMP_SUFFIX "$regex" "$file"
  rm -f "$file".$TEMP_SUFFIX
}

_check_cli() {
  for i in "${!CLI[@]}"; do
    cli="${CLI[$i]}"
    if ! command -v "$cli" &>/dev/null; then
      echo "Command '$cli' not found! Hint: you should install it."
      exit 1
    fi
  done
}

_check_status() {
  if [[ -n $(git status . -s) ]]; then
    echo "Error: Commit unstaged files first, and then run this tool again."
    exit 1
  fi
}

_check_init() {
  local _has_inited=false

  if [[ ! -d .github ]]; then # using option `--no-gh`
    _has_inited=true
  else
    if [[ -f .github/workflows/$ACTIONS_WORKFLOW ]]; then
      # on BSD, the `wc` could contains blank
      local _count
      _count=$(find .github/workflows/ -type f -name "*.yml" | wc -l)
      if [[ ${_count//[[:blank:]]/} == 1 ]]; then
        _has_inited=true
      fi
    fi
  fi

  if $_has_inited; then
    echo "Already initialized."
    exit 0
  fi
}

check_env() {
  _check_cli
  _check_status
  _check_init
}

checkout_latest_release() {
  hash=$(git log --grep="chore(release):" -1 --pretty="%H")
  git reset --hard "$hash"
}

init_files() {
  if $_no_gh; then
    rm -rf .github
  else
    ## Change the files of `.github`
    mv .github/workflows/$ACTIONS_WORKFLOW.hook .
    rm -rf .github
    mkdir -p .github/workflows
    mv ./${ACTIONS_WORKFLOW}.hook .github/workflows/${ACTIONS_WORKFLOW}

    ## Cleanup image settings in site config
    _sedi "s/^img_cdn:.*/img_cdn:/;s/^avatar:.*/avatar:/" _config.yml
  fi

  # remove the other files
  rm -rf _posts/*

  # build assets
  npm i && npm run build

  # track the js output
  _sedi "/^assets.*\/dist/d" .gitignore
}

commit() {
  git add -A
  git commit -m "chore: initialize the environment" -q
  echo -e "\n[INFO] Initialization successful!\n"
}

main() {
  check_env
  checkout_latest_release
  init_files
  commit
}

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

main
