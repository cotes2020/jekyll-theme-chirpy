---
title: Jekyll 기술블로그에 flow chart와 수식 넣기 (Feat. Mermaid, MathJax)
author:
  name: Bean
  link: https://github.com/beanie00
date: 2022-05-05 09:13:00 +0800
categories: [etc, 기술블로그]
tags: []
---

## Mermaid로 flow chart 넣기
---

### Mermaid?
---

Mermaid는 스크립트로 각종 다이어그램을 그려주는 JavaScript이다.

```
graph LR
  A(랜딩페이지)-->B[자동로그인 확인]
  B-->C(로그인 페이지)
  B-->D(메인페이지)
```

이런식으로 직관적으로 작성된 스크립트를 다이어그램으로 변환시켜준다.

<div class="mermaid">
  graph LR
  A(랜딩페이지)-->B[자동로그인 확인]
  B-->C(로그인 페이지)
  B-->D(메인페이지)
</div>

&nbsp;

### Rendering Mermaid in jekyll
---

지금 보여지고 있는 기술블로그에서 사용중인 Jekyll Chirpy 테마를 사용중이다.
가이드 문서에는 포스팅 헤드에 다음을 추가하면 `'''mermaid` 로 mermaid를 사용할 수 있다고 되어 있다.

```yaml
---
mermaid: true
---
```

하지만 무슨 이유에선지 잘 되지 않아 다른 방법을 찾아보았다.

jekyll에서 Mermaid를 렌더링해야 하는 방법으로 검색하면 다음의 2가지 방법이 나온다.
* [jekyll-mermaid](https://github.com/jasonbellamy/jekyll-mermaid)
* [jekyll-spaceship](https://github.com/jeffreytse/jekyll-spaceship)

하지만 둘다 직접 적용해보니 잘 적용되지 않았다.

그래서 직접 Mermaid를 html 파일에 임베딩하기로 하였다.

#### Embedding MermaidPermalink
Mermaid-js에 들어가보면 해당 js file의 CDN이 존재한다.
각 html 문서 앞에 아래 항목을 공통적으로 집어 넣어준다. 이 블로그에서는 _includes\head.html에 넣어주었다. 그냥 html 문서에 공통으로 들어가는 부분에 추가해주면 된다.

```javascript
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true});</script>
```

그리고 블로그 포스팅 .md 파일 내에서 다음과 같이 사용해주면 된다.

```markdown
<div class="mermaid">
  graph LR
  A(랜딩페이지)-->B[자동로그인 확인]
  B-->C(로그인 페이지)
  B-->D(메인페이지)
</div>
```

## MathJax 수식 넣기
---
### MathJax?
---
MathJax는 MathML, LaTeX 및 ASCIIMathML 마크 업을 사용하여 웹 브라우저에 수학 표기법을 표시하는 크로스 브라우저 JavaScript 라이브러리이다. MathJax는 아파치 라이선스에 따라 오픈 소스 소프트웨어로 제공된다.

### Rendering MathJax in jekyll
---
flow chart와 비슷하게 _includes\head.html에 다음의 코드를 추가하면 사용할 수 있다.

```html
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      inlineMath: [['$','$']]
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
```
그러면 라텍스 문법으로 작성된 수식이 잘 렌더링된다.
