---
title: GitHub Pages와 Jekyll로 깃허브 블로그 작성 방법
author: leewoojin
date: 2025-04-29 20:00:00 +0900
categories: [Blogging, Tutorial]
tags: [GitHub Pages, Jekyll, blog]
---

성실함을 기록으로 보여주기 위해, 기존에 사용하던 네이버 블로그를 정리하고 GitHub 블로그로 이전했습니다.  
잔디 심기를 통해 하루하루의 노력을 쌓아가는 모습을 남기고자 합니다.

처음 구축하는 과정에서 쥔장이 직접 많이 헤맸던 부분들만 간단하게 정리해두었습니다.

---

## 1. 테마를 GitHub Fork 기능으로 가져옵니다.

- GitHub에서 원하는 테마(저는 Chirpy)를 찾아 **Fork** 버튼을 누릅니다.
- 내 계정에 `username.github.io` 형식의 저장소를 새로 만듭니다.

**(주의: 저장소 이름이 반드시 `username.github.io` 형태여야 합니다.)**

---

## 2. 저장소를 로컬로 복제합니다.

```bash
git clone https://github.com/내아이디/내아이디.github.io.git
cd 내아이디.github.io
```

복제해온 폴더 안에서 블로그 작업을 진행합니다.

---

## 3. Ruby, Jekyll, Bundler 설치하기

### 3-1. Ruby 설치

```bash
brew install ruby
```

**환경변수 설정:** (필요할 경우)

```bash
echo 'export PATH="/usr/local/opt/ruby/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**설치 확인:**

```bash
ruby -v
```

### 3-2. Jekyll & Bundler 설치

```bash
gem install bundler jekyll
```

**설치 확인:**

```bash
jekyll -v
bundler -v
```

---

## 4. 프로젝트 의존성 설치하기

Fork한 테마 저장소에 있는 Gemfile을 기반으로 필요한 패키지를 설치합니다.

```bash
bundle install
```

만약 `package.json` 파일이 있다면:

```bash
npm install
```

---

## 5. `_config.yml` 수정하기

이 부분은 제가 참고한 사이트를 링크로 걸어두겠습니다.

👉 [Git 블로그 만들기 참고](https://wlqmffl0102.github.io/posts/Making-Git-blogs-for-beginners-1/)

---

## 6. 로컬 서버 실행하고 확인하기

```bash
bundle exec jekyll serve
```

브라우저에서 `http://127.0.0.1:4000` 접속하면 내 블로그가 로컬에서 어떻게 보이는지 확인할 수 있습니다.

**로컬에서 보이는 것 = 실제 GitHub Pages 배포 결과와 거의 같습니다.**

---

## 7. 블로그 수정 후 GitHub에 푸시하기

1. 변경사항 확인

```bash
git status
```

2. 변경된 파일 추가

```bash
git add .
```

3. 커밋 메시지 작성

```bash
git commit -m "블로그 글 추가: GitHub Pages와 Jekyll로 블로그 구축 방법"
```

4. 원격 저장소로 푸시

```bash
git push origin main
```

> 한 줄로 요약해서 입력할 수도 있습니다:

```bash
git add . && git commit -m "커밋 메시지" && git push origin main
```

---

# 마무리

쥔장은 아직도 깃허브 블로그가 많이 어렵습니다.  
익숙해지는 그날을 기다려봅니다.
