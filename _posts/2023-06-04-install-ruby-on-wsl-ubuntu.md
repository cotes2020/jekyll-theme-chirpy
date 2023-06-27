---
title: Install Ruby on WSL Ubuntu
date: 2023-06-04 23:47 +0800
category: [Environment Settings]
tag: [WSL Ubuntu, Ruby]
---

### 문제 상황

WSL Ubuntu환경에서, jekyll을 이용하기 위해 Ruby와 Bundler를 설치해야 했다.

[Ruby 공식문서](https://www.ruby-lang.org/en/documentation/installation/)를 참고하여 [Ruby enViornment Manager (RVM)](https://rvm.io/)으로 설치하다 실패.

`rvm implode`으로 말끔하게 지우고 다른 설치 방법을 찾았다.

### 해결법

`sudo apt-get install ruby-full`
: ruby의 모든 하위 패키지를 함께 설치한다. bundler도 함께 설치된다.

`sudo apt-get install ruby`
: 필요에 따라 ruby의 모든 하위 패키지가 필요하지 않은 경우 필수 요소만 설치할 수 있다.

`sudo apt-get remove <패키지명>`
: 패키지를 지울 수 있다. 잘못 설치한 경우 참고.

Jekyll 테마를 설치하기 위한 Ruby라면 아래 명령어로 해결할 수 있다.

```
sudo apt-get install ruby ruby-dev
sudo gem update
sudo gem install jekyll bundler
sudo bundle
```