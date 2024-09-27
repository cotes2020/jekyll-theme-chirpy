---
title: Chirpy 개발환경 설정 방법 (Ubuntu)
author: jinwoo
date: 2024-09-27 12:54 +0900
categories: [Tutorial]
tags: [system]
---

이 튜토리얼은 우분투에서 `Chirpy` 개발환경을 설정하는 방법을 정리하였습니다.

버전 `7.1.0` 기준으로 작성하였습니다.

## Ruby 환경 설정

### rvm 설치

`Ruby Version Manager`의 약자로, 다양한 버전의 Ruby를 한 시스템에서 쉽게 관리할 수 있도록 도와주는 도구

```zsh
sudo apt-add-repository -y ppa:rael-gc/rvm
sudo apt-get update
sudo apt-get install rvm
```

RVM의 User 권한 설정

```zsh
sudo usermod -a -G rvm $USER

# bash일 경우
echo 'source "/etc/profile.d/rvm.sh"' >> ~/.bashrc

# zsh일 경우
echo 'source "/etc/profile.d/rvm.sh"' >> ~/.zshrc
```
> RVM에 권한 문제가 있을 경우 아래 명령을 이용
{: .prompt-tip }

```zsh
command curl -sSL https://rvm.io/mpapis.asc | sudo gpg2 --import -
command curl -sSL https://rvm.io/pkuczynski.asc | sudo gpg2 --import -
rvm fix-permissions user
rvm fix-permissions system
rvm reload
```

### Ruby(v3.1.1) 설치

```zsh
rvm install 3.1.1 --with-download-link=https://ftp.ruby-lang.org/pub/ruby/3.1/ruby-3.1.1.tar.gz
```

설치 완료 후, `default version` 설정

```zsh
## Version 확인
ruby -v

## default version 설정
rvm use ruby-3.1.1
```

## Jekyll 환경 설정

### 설치

```zsh
gem install jekyll bundler
```

### 빌드

```zsh
# workspace로 이동
cd <사이트 환경>

# bundle을 이용하여 빌드 진행
bundle install
```

### 사이트 실행

```zsh
bundle exec jekyll serve
```

> default로 <http://127.0.0.1:4000/> 설정이 되어있어, 해당 링크로 들어가면 빌드된 사이트를 볼 수 있다.
{: .prompt-tip }
