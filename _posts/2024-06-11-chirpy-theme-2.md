---
title: Jekyll Chirpy Theme (2)
author: Leesh
categories: [jekyll, chirpy] 
tags: [jekyll, chirpy, github-blog, github, blog]
date: '2024-06-11 14:00:00 +0900'
---
## 아바타 이미지 설정

---
* `assets/img/[avata_img.png]` 경로에 아바타 이미지 업로드
* `_config.yml` 파일의 avata 설정

```yaml
# the avatar on sidebar, support local or CORS resources
avatar: assets/img/crab.jpeg
```

## favicon.ico 설정

---
`assets/img/favicons/[favicon files].ico` 파일들을 교체

{:.prompt-tip}
> 1) [iconfinder](https://www.iconfinder.com/icons){:target="_blank"} 에서 무료 아이콘 찾기
> 2) [favicon.io](https://favicon.io/favicon-converter/){:target="_black"} 에서 favicon 변환

## 사이드바 타이틀/서브 타이틀 폰트 색상 설정

---
* `_sass/colors/typography-dark.scss` 에서 아래의 내용으로 수정
* 배경 색상이 dark라서 잘 보이도록, 타이틀/서브 타이틀 폰트를 화이트 색상으로 `(#ffffff;)` 수정

```css
  /* Sidebar */
  --site-title-color: #717070;
  /* --site-title-color: #717070; */
  --site-title-color: #ffffff;
  /* --site-subtitle-color: #868686; */
  --site-subtitle-color: #ffffff;
  --sidebar-bg: #1e1e1e;
  --sidebar-border-color: #292929;
```
{: .nolineno }

## 사이드바 메뉴 폰트 색상 변경

---
* `_sass/addon/commons.scss` 에 내용 추가
* `color: rgba(255, 255, 255, 0.99);` "화이트 색상"

![](/assets/img/2024-06-11-chirpy-theme-2_images/7f0ae595.png)

## 사이드바 배경 이미지 설정

---
* `assets/img/[sidebar_img.png]` 경로에 사이드바 이미지 업로드.
* `_sass/addon/commons.scss` 내용 추가 (이미지에 맞게 size, position 값 조정)
```css
  background: url('/assets/img/sidebar_back_img.jpeg');
  background-size: cover;
  background-repeat: no-repeat;
  background-position: 40%;
```
{: .nolineno }

## Footer 수정

---
`_data/locales/ko-KR.yml` 에서 아래 내용 삭제
![](/assets/img/2024-06-11-chirpy-theme-2_images/00264e9e.png)

## 참고

---

* 하얀눈길님 블로그 : https://www.irgroup.org/posts/jekyll-chirpy/
* JSDevBlog : https://jason9288.github.io/posts/github_blog_3/
