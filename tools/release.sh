#!/usr/bin/env bash
#
# According to the GitLab flow release branching model,
# cherry-pick the last commit on the main branch to the release branch,
# and then create a tag and gem package on the release branch (naming format: 'release/<X.Y>').
#
#
# Usage:
#
#   It can be run on main branch, and it should be used after just finishing the last feature in the version plan,
#   or just after merging the hotfix to the main branch.
#
# Requires: Git, Gulp

set -eu

GEM_SPEC="jekyll-theme-chirpy.gemspec"

check() {
  if [[ -n $(git status . -s) ]]; then
    echo "Error: Commit unstaged files first, and then run this tool againt."
    exit -1
  fi

  if [[ ! -f $GEM_SPEC ]]; then
    echo -e "Error: Missing file \"$GEM_SPEC\"!\n"
    exit -1
  fi
}

## Remove unnecessary theme settings
cleanup_config() {
  cp _config.yml _config.yml.bak
  sed -i "s/^img_cdn:.*/img_cdn:/;s/^avatar:.*/avatar:/" _config.yml
}

resume_config() {
  mv _config.yml.bak _config.yml
}

release() {
  _default_branch="$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@')"
  _version="$(grep "spec.version" jekyll-theme-chirpy.gemspec | sed 's/.*= "//;s/".*//')" # X.Y.Z
  _release_branch="release/${_version%.*}"

  if [[ -z $(git branch -v | grep "$_release_branch") ]]; then
    # create a new release branch
    git checkout -b "$_release_branch"
  else
    # cherry-pick the latest commit from default branch to release branch
    _last_commit="$(git rev-parse "$_default_branch")"
    git checkout "$_release_branch"
    git cherry-pick "$_last_commit" -m 1
  fi

  # create new tag
  echo -e "Create tag v$_version\n"
  git tag "v$_version"

  # build a gem package
  echo -e "Build the gem pakcage for v$_version\n"
  cleanup_config
  rm -f ./*.gem
  gem build "$GEM_SPEC"
  resume_config
}

main() {
  check
  release
}

main
