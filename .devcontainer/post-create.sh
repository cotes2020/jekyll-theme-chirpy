#!/usr/bin/env bash

npm_buid() {
  bash -i -c "nvm install --lts"
  npm i && npm run build
}

[[ -d _sass/dist && -d assets/js/dist ]] || npm_buid

exec zsh
