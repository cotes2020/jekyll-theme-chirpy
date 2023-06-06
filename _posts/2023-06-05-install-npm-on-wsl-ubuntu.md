---
title: Install npm on WSL Ubuntu
date: 2023-06-05 02:04 +0900
category: [Environment Settings]
tag: [WSL Ubuntu, npm]
---

### 문제 상황

WSL의 Ubuntu환경에서, jekyll의 [chirpy테마](https://github.com/cotes2020/jekyll-theme-chirpy)를 이용하기 위해 Node.js를 설치해야 했다.

### 해결법

[WSL 공식문서](https://learn.microsoft.com/en-us/windows/dev-environment/javascript/nodejs-on-wsl#install-nvm-nodejs-and-npm)를 참고하였다.

`curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash`
: nvm 설치한다. Node enVironment Manager를 축약한 말으로 node를 설치하기 위해 필요하다.

`command -v nvm`
: nvm이 설치되어있는지 확인한다. nvm이라고 출력되지 않으면 설치되지 않은 것이다. 터미널을 껐다 켜면 nvm이 인식되는 경우가 있다.

`nvm install node`
: node를 설치한다. npm도 동시에 설치된다.

`nvm ls`
: 설치된 node버전을 확인한다. N/A라고 표시되어있으면 설치되지 않은 것이다.
