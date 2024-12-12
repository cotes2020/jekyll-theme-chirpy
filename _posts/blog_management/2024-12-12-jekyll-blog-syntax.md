---
title: "[github page] jekyll 테마 마크다운 문법 - 첨부파일 문법"
categories: [blog management TIP]
tags: [jekyll, github page]
---

## 첨부파일

### 이미지 첨부

- basic

```md
![Image]({{"/assets/img/posts/IMAGE-FILE-NAME" | relative_url }})
```

- 크기 변경

```md
![Image]({{"/assets/img/posts/IMAGE-FILE-NAME" | relative_url }}){: style="width: 500px;" }
```

- 이미지에 테두리 추가

```md
![Image]({{"/assets/img/posts/IMAGE-FILE-NAME" | relative_url }}){: style="border: 2px solid black;" }
```

- 이미지 정렬
  - 왼쪽 정렬
  ```md
  ![Image]({{"/assets/img/posts/IMAGE-FILE-NAME" | relative_url }}){: style="float: left; margin-right: 10px;" }
  ```
  - 오른쪽 정렬
  ```md
  ![Image]({{"/assets/img/posts/IMAGE-FILE-NAME" | relative_url }}){: style="float: right; margin-left: 10px;" }
  ```

### 동영상 첨부

```html
<iframe width="480" height="360" src="VIDEO-ADDRESS-URL" frameborder="0">
</iframe>
```

### 링크 첨부

- 기본 링크

```md
[링크 텍스트](URL)
```

- 새 창에서 열리도록 설정

```md
[링크 텍스트](URL){: target="\_blank" }
```
