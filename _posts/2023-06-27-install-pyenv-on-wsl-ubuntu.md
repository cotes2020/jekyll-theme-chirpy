---
title: Install Pyenv on WSL Ubuntu
date: 2023-06-27 12:46 +0900
category: [Environment Settings]
tag: [WSL Ubuntu, Pyenv]
---

WSL에 Pyenv를 설치하는 방법이다. Pyenv는 다양한 파이썬 버전을 아나콘다 없이도 관리할 수 있어 개인적으로 선호하는 도구이다.

WSL환경에 아무것도 설치되어있지 않다고 가정한다.

아래 명령어를 입력하여 [여기](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)에서 제공하는 권장 환경을 설치한다.

```
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

아래 명령어를 입력하여 pyenv를 설치한다. [여기](https://github.com/pyenv/pyenv-installer)에 자세한 설명이 나와있다.

```
curl https://pyenv.run | bash
```

### Ref.

<https://github.com/pyenv/pyenv/wiki#suggested-build-environment>

<https://github.com/pyenv/pyenv-installer>