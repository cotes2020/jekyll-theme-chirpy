#!/bin/bash

# Travis CI build jobs.
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2019 Cotes Chung
# Published under MIT License

if [[ $TRAVIS_PULL_REQUEST = "false" ]]; then # triggered by author

  git clone https://${GH_PAT}@github.com/${GH_USER}/${BUILDER_REPO}.git \
    ../${BUILDER_REPO} --depth=1 -q

  cp -r ../${BUILDER_REPO}/framework/* .
  bash _cibuild.sh

else # triggered by Pull Request

  bash tools/build.sh
  bash tools/test.sh

fi
