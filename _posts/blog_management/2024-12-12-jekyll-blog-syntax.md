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

- 블로그 내 다른 포스트 링크

```md
[링크 텍스트]({{ site.baseurl }}{% link _posts/ai/langchain/2024-12-11-01_Langchain모듈-00-basic.md %})
```

- 기본 링크

```md
[링크 텍스트](URL)
```

- 새 창에서 열리도록 설정

```md
[링크 텍스트](URL){: target="\_blank" }
```

## Code block

### File path를 제목으로 주기

````md
```shell
# content
```

{: file="path/to/file" }
````

```shell
# content
```

{: file="path/to/file" }

### Special Block

- tip

```
<!-- prettier-ignore --> #프리티어 자동변환을 막기 위해, 이 설정을 추가해준다.
> This is an example of a Tip.
{: .prompt-tip }
```

<!-- prettier-ignore -->
> This is an example of a Tip.
{: .prompt-tip }

- info

```
<!-- prettier-ignore --> #프리티어 자동변환을 막기 위해, 이 설정을 추가해준다.
> This is an example of an Info block.
{: .prompt-info }
```

<!-- prettier-ignore -->
> This is an example of an Info block.
{: .prompt-info }

- warning

```
<!-- prettier-ignore --> #프리티어 자동변환을 막기 위해, 이 설정을 추가해준다.
> This is an example of a Warning block.
{: .prompt-warning }
```

<!-- prettier-ignore -->
> This is an example of a Warning block.
{: .prompt-warning }

- danger

```
<!-- prettier-ignore --> #프리티어 자동변환을 막기 위해, 이 설정을 추가해준다.
> This is an example of a Danger block.
{: .prompt-danger }
```

<!-- prettier-ignore -->
> This is an example of a Danger block.
{: .prompt-danger }

## 그 외 팁

- [블로그](https://jeeklee.github.io/posts/%EB%B2%88%EC%97%AD-Chirpy-%ED%8F%AC%EC%8A%A4%ED%8C%85-%EC%9E%91%EC%84%B1/)
