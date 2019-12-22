#!/bin/bash

# Travis CI build jobs.
#
# © 2019 Cotes Chung
# Published under MIT License

if [[ $TRAVIS_PULL_REQUEST = "false" ]]; then # triggered by author

  BUILDER=../blog-builder

  git clone https://${GH_TOKEN}@github.com/cotes2020/blog-builder.git \
    $BUILDER --depth=1 -q

  cp -r $BUILDER/framework/* .
  bash _cibuild.sh

else # triggered by Pull Request

  SAFE_DOMAIN=cdn.jsdelivr.net

  bundle install --path vendor/bundle --quiet

  python _scripts/py/init_all.py

  build_cmd="JEKYLL_ENV=production bundle exec jekyll build"

  echo "\$ $build_cmd"
  eval $build_cmd

  bundle exec htmlproofer --disable-external \
            --check-html \
            --empty_alt_ignore \
            --allow_hash_href \
            --url_ignore $SAFE_DOMAIN \
            _site/

fi
