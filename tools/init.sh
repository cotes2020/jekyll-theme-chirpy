#!/usr/bin/env bash
#
# Init the environment for new user.

set -eu

# CLI Dependencies
CLI=("git" "npm")

ACTIONS_WORKFLOW=pages-deploy.yml

RELEASE_HASH=$(git log --grep="chore(release):" -1 --pretty="%H")

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
  sed -i.$TEMP_SUFFIX -E "$regex" "$file"
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
  if [[ $(git rev-parse HEAD^1) == "$RELEASE_HASH" ]]; then
    echo "Already initialized."
    exit 0
  fi
}

check_env() {
  _check_cli
  _check_status
  _check_init
}

reset_latest() {
  git reset --hard "$RELEASE_HASH"
  git clean -fd
  git submodule update --init --recursive
}

init_files() {
  if $_no_gh; then
    rm -rf .github
  else
    ## Change the files of `.github/`
    temp="$(mktemp -d)"
    find .github/workflows -type f -name "*$ACTIONS_WORKFLOW*" -exec mv {} "$temp/$ACTIONS_WORKFLOW" \;
    rm -rf .github && mkdir -p .github/workflows
    mv "$temp/$ACTIONS_WORKFLOW" .github/workflows/"$ACTIONS_WORKFLOW"
    rm -rf "$temp"
  fi

  # Cleanup image settings in site config
  _sedi "s/(^timezone:).*/\1/;s/(^.*cdn:).*/\1/;s/(^avatar:).*/\1/" _config.yml

  # remove the other files
  rm -rf tools/init.sh tools/release.sh _posts/*

  # build assets
  npm i && npm run build

  # track the CSS/JS output
  _sedi "/.*\/dist$/d;/^_app$/d" .gitignore
}

commit() {
  git add -A
  git commit -m "chore: initialize the environment" -q
  echo -e "\n> Initialization successful!\n"
}

main() {
  check_env
  reset_latest
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
