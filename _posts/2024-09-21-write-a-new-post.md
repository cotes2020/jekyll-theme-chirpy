---
title: 새 포스트 작성 방법 (Chirpy)
author: jinwoo
date: 2024-09-21 21:30:00 +0900
categories: [Tutorial]
tags: [writing]
render_with_liquid: false
---

이 튜토리얼은 _Chirpy_ 템플릿에서 게시물을 작성하는 방법을 안내합니다. 

한국어로 이해할 수 있도록 의역한 점이 있는 것을 참고하시기 바랍니다.

## Naming and Path

`YYYY-MM-DD-TITLE.EXTENSION`{: .filepath}라는 이름의 새 파일을 만들고 루트 디렉토리의 `_posts`{: .filepath}에 넣습니다.

`EXTENSION`{: .filepath}은 `md`{: .filepath}와 `markdown`{: .filepath} 중 하나여야 합니다.

파일 생성 시간을 절약하려면 플러그인 [`Jekyll-Compose`](https://github.com/jekyll/jekyll-compose)를 사용하여 이를 달성하는 것을 고려하세요.

## Front Matter

기본적으로 게시물 맨 위에 아래와 같이 [Front Matter](https://jekyllrb.com/docs/front-matter/)를 채워야 합니다:

```yaml
---
title: TITLE
date: YYYY-MM-DD HH:MM:SS +/-TTTT
categories: [TOP_CATEGORIE, SUB_CATEGORIE]
tags: [TAG]     # TAG names should always be lowercase
---
```

> 게시물의 _layout_은 기본적으로 `post`로 설정되었으므로, Front Matter 블록에 _layout_ 변수를 추가할 필요가 없습니다.
{: .prompt-tip }

### Timezone of Date

게시물의 릴리스 날짜를 정확하게 기록하려면 `_config.yml`{: .filepath}의 `timezone`을 설정할 뿐만 아니라 Front Matter 블록의 변수 `date`에 게시물의 시간대를 제공해야 합니다. 형식: `+/-TTTT`, 예: `+0900`(KST).

### Categories and Tags

각 게시물의 `categories`는 최대 두 개의 요소를 포함하도록 설계되었으며 `tags`의 요소 수는 0에서 무한대까지 가능합니다. 

예를 들면 아래와 같습니다.

```yaml
---
categories: [Animal, Insect] # 대분류, 소분류
tags: [bee]
---
```

### Author Information

게시물의 작성자 정보는 일반적으로 _Front Matter_ 에 채울 필요가 없으며, 기본적으로 설정 파일의 `social.name` 변수와 `social.links`의 첫 번째 항목에서 가져옵니다. 

`_data/authors.yml`에 저자 정보를 추가하여 재정의 할 수 있습니다.
(웹사이트에 이 파일이 없다면, 주저하지 말고 하나 만드세요).

```yaml
<author_id>:
  name: <full name>
  twitter: <twitter_of_author>
  url: <homepage_of_author>
```
{: file="_data/authors.yml" }

그리고 `author`를 사용하여 단일 항목을 지정하거나 `authors`를 사용하여 여러 항목을 지정합니다:

```yaml
---
author: <author_id>                     # for single entry
# or
authors: [<author1_id>, <author2_id>]   # for multiple entries
---
```

즉, 키 `author`를 사용하여 여러 항목을 식별할 수도 있습니다.

> `_data/authors.yml`{: .filepath } 파일에서 작성자 정보를 읽는 이점은 페이지에 `twitter:creator` 메타 태그가 있어서 [Twitter Cards](https://developer.twitter.com/en/docs/twitter-for-websites/cards/guides/getting-started#card-and-content-attribution)를 풍부하게 만들고 SEO에 좋다는 것입니다.
> 
{: .prompt-info }

### Post Description

게시물에 대한 자동 생성된 설명을 표시하지 않으려면 다음과 같이 _Front Matter_의 `description` 필드를 사용하여 사용자 정의할 수 있습니다.

설정되지 않으면 게시물의 첫 단어가 추가자료 섹션 및 RSS 피드의 XML에 표시되는데 사용됩니다. 

```yaml
---
description: Short summary of the post.
---
```

또한, `description` 텍스트는 게시물 페이지의 게시물 제목 아래에도 표시됩니다.

## Table of Contents

기본적으로 **T**able **o**f **C**ontents (TOC)**는 게시물의 오른쪽 패널에 표시됩니다. 

전역적으로 끄려면 `_config.yml`{: .filepath}로 이동하여 `toc` 변수의 값을 `false`로 설정합니다. 

특정 게시물의 TOC를 끄려면 게시물의 [Front Matter](https://jekyllrb.com/docs/front-matter/)에 다음을 추가합니다.

```yaml
---
toc: false
---
```

## Comments

전역적으로 댓글 활성화는 파일 `_config.yml`{: .filepath}의 변수 `comments.active`에 의해 정의됩니다.

특정 게시물에 대한 댓글을 닫으려면 게시물의 **Front matter**에 다음을 추가하세요.

```yaml
---
comments: false
---
```

## Media

_Chirpy_에서는 이미지, 오디오, 비디오를 미디어 리소스라고 합니다.

### URL Prefix

때때로 게시물에서 여러 리소스에 대해 중복된 URL 접두사를 정의해야 합니다. 

이는 지루한 작업이지만 두 개의 매개변수를 설정함으로써 피할 수 있습니다.

- CDN을 사용하여 미디어 파일을 호스팅하는 경우 `_config.yml`{: .filepath }에서 `cdn`을 지정할 수 있습니다. 
- 그런 다음 사이트 아바타 및 게시물의 미디어 리소스 URL에 CDN 도메인 이름이 접두사로 붙습니다.

  ```yaml
  cdn: https://cdn.com
  ```
  {: file='_config.yml' .nolineno }

- 현재 게시물/페이지 범위에 대한 리소스 경로 접두사를 지정하려면 게시물의 **front matter**에 `media_subpath`를 설정합니다.

  ```yaml
  ---
  media_subpath: /path/to/media/
  ---
  ```
  {: .nolineno }

`site.cdn` 옵션과 `page.media_subpath` 옵션은 개별적으로 또는 조합하여 최종 리소스 URL을 유연하게 구성할 수 있습니다: `[site.cdn/][page.media_subpath/]file.ext`

### Images

#### Caption

이미지의 다음 줄에 이탤릭체를 추가하면 해당 줄이 캡션이 되어 이미지 맨 아래에 나타납니다.

```markdown
![img-description](/path/to/image)
_Image Caption_
```
{: .nolineno}

#### Size

이미지가 로드될 때 페이지 콘텐츠 레이아웃이 이동하는 것을 방지하려면 각 이미지의 너비와 높이를 설정해야 합니다.

```markdown
![Desktop View](/assets/img/sample/mockup.png){: width="700" height="400" }
```
{: .nolineno}

> SVG의 경우 최소한 _width_ 를 지정해야 합니다. 그렇지 않으면 렌더링되지 않습니다.
{: .prompt-info }

_Chirpy v5.0.0_부터 `height`와 `width`는 약어를 지원합니다(`height` → `h`, `width` → `w`). 

다음 예는 위와 동일한 효과를 갖습니다.

```markdown
![Desktop View](/assets/img/sample/mockup.png){: w="700" h="400" }
```
{: .nolineno}

#### Position

기본적으로 이미지는 중앙에 배치되지만 `normal`, `left`, `right` 클래스 중 하나를 사용하여 위치를 지정할 수 있습니다.

> 위치가 지정되면 이미지 캡션을 추가하면 안 됩니다.
{: .prompt-warning }

- **Normal position**

  아래 샘플에서는 이미지가 왼쪽 정렬됩니다.

  ```markdown
  ![Desktop View](/assets/img/sample/mockup.png){: .normal }
  ```
  {: .nolineno}

- **Float to the left**

  ```markdown
  ![Desktop View](/assets/img/sample/mockup.png){: .left }
  ```
  {: .nolineno}

- **Float to the right**

  ```markdown
  ![Desktop View](/assets/img/sample/mockup.png){: .right }
  ```
  {: .nolineno}

#### Dark/Light mode

다크/라이트 모드에서 이미지가 테마 환경 설정을 따르도록 할 수 있습니다. 

이를 위해 다크 모드용 이미지 하나와 라이트 모드용 이미지 하나를 준비한 다음, 특정 클래스(`dark` 또는 `light`)를 지정해야 합니다.

```markdown
![Light mode only](/path/to/light-mode.png){: .light }
![Dark mode only](/path/to/dark-mode.png){: .dark }
```

#### Shadow

프로그램 창의 스크린샷은 `shadow effect`를 보여주는 것으로 볼 수 있습니다.

![Desktop View](/img/tutorials/mockup.png){: .shadow }

```markdown
![Desktop View](/img/tutorials/mockup.png){: .shadow }
```
{: .nolineno}

#### Preview Image

게시물 상단에 이미지를 추가하려면 해상도가 `1200 x 630`인 이미지를 제공해야합니다.

> 이미지 종횡비가 `1.91 : 1`에 맞지 않으면 이미지가 크기 조정되고 잘립니다.
{: .prompt-warning }

이러한 전제 조건을 알면 이미지의 속성을 설정할 수 있습니다.

```yaml
---
image:
  path: /path/to/image
  alt: image alternative text
---
```

[`media_subpath`](#url-prefix)도 미리보기 이미지에 전달될 수 있습니다. 즉, `path` 속성이 설정된 경우 이미지 파일 이름만 필요합니다.

간단하게 사용하려면 `image`를 사용하여 경로를 정의하면 됩니다.

```yml
---
image: /path/to/image
---
```

#### LQIP

미리보기 이미지의 경우:

```yaml
---
image:
  lqip: /path/to/lqip-file # or base64 URI
---
```

> 게시물 \"[텍스트, 타이포그래피 사용 방법 (Chirpy)](../text-and-typography/)\"의 미리보기 이미지에서 LQIP를 관찰할 수 있습니다.

일반 이미지의 경우:

```markdown
![Image description](/path/to/image){: lqip="/path/to/lqip-file" }
```
{: .nolineno }

### Video

#### Social Media Platform

다음 구문을 사용하여 소셜 미디어 플랫폼의 비디오를 내장할 수 있습니다.

```liquid
{% include embed/{Platform}.html id='{ID}' %}
```

여기서 `Platform`은 플랫폼 이름의 소문자이고, `ID`는 비디오 ID입니다.

다음 표는 주어진 비디오 URL에 필요한 두 가지 매개변수를 가져오는 방법을 보여줍니다. 

또한 현재 지원되는 비디오 플랫폼도 알 수 있습니다.

| Video URL                                                                                          | Platform   | ID             |
| -------------------------------------------------------------------------------------------------- | ---------- | :------------- |
| [https://www.**youtube**.com/watch?v=**H-B46URT4mg**](https://www.youtube.com/watch?v=H-B46URT4mg) | `youtube`  | `H-B46URT4mg`  |
| [https://www.**twitch**.tv/videos/**1634779211**](https://www.twitch.tv/videos/1634779211)         | `twitch`   | `1634779211`   |
| [https://www.**bilibili**.com/video/**BV1Q44y1B7Wf**](https://www.bilibili.com/video/BV1Q44y1B7Wf) | `bilibili` | `BV1Q44y1B7Wf` |

#### Video Files

비디오 파일을 직접 포함하려면 다음 구문을 사용하세요.

```liquid
{% include embed/video.html src='{URL}' %}
```

여기서 `URL`은 비디오 파일의 URL입니다(예시: `/path/to/sample/video.mp4`).

또한 내장된 비디오 파일에 대한 추가 속성을 지정할 수 있습니다. 

허용되는 속성의 전체 목록은 다음과 같습니다.

- `poster='/path/to/poster.png'` — 비디오 다운로드 중 표시되는 비디오의 포스터 이미지
- `title='Text'` — 비디오 아래에 나타나며 이미지와 동일하게 보이는 비디오 제목
- `autoplay=true` — 비디오는 가능한 빨리 자동 재생을 시작
- `loop=true` — 비디오 끝에 도달하면 자동으로 시작 부분으로 돌아감
- `muted=true` — 오디오는 처음에는 음소거
- `types` — 추가 비디오 형식의 확장자를 `|`로 구분하여 지정, 이러한 파일이 기본 비디오 파일과 같은 디렉토리에 있는지 확인필요

위의 모든 것을 사용한 예를 고려해 보겠습니다.

```liquid
{%
  include embed/video.html
  src='/path/to/video.mp4'
  types='ogg|mov'
  poster='poster.png'
  title='Demo video'
  autoplay=true
  loop=true
  muted=true
%}
```

### Audios

오디오 파일을 직접 포함하려면 다음 구문을 사용하세요.

```liquid
{% include embed/audio.html src='{URL}' %}
```

여기서 `URL`은 오디오 파일의 URL입니다(예시: `/path/to/audio.mp3`).

내장된 오디오 파일에 대한 추가 속성을 지정할 수도 있습니다.

허용되는 속성의 전체 목록은 다음과 같습니다.

- `title='Text'` — 오디오 아래에 나타나고 이미지와 동일하게 보이는 오디오의 제목
- `types` — 추가 오디오 형식의 확장자를 `|`로 구분하여 지정, 이러한 파일이 기본 오디오 파일과 같은 디렉토리에 있는지 확인필요

위의 모든 것을 사용한 예를 고려해 보겠습니다.

```liquid
{%
  include embed/audio.html
  src='/path/to/audio.mp3'
  types='ogg|wav|aac'
  title='Demo audio'
%}
```

## Pinned Posts

하나 이상의 게시물을 홈페이지 상단에 고정할 수 있으며, 고정된 게시물은 출시 날짜에 따라 역순으로 정렬됩니다.

_Front Matter_의 `pin` 필드를 사용하여 사용자 정의할 수 있습니다.

```yaml
---
pin: true
---
```
### Filepath Highlight

```md
`/path/to/a/file.extend`{: .filepath}
```
{: .nolineno }

### Code Block

마크다운 기호 ```` ``` ````를 사용하면 다음과 같이 쉽게 코드 블록을 만들 수 있습니다.

````md
```
This is a plaintext code snippet.
```
````

#### Specifying Language

```` ```{language} ````를 사용하면 구문 강조가 적용된 코드 블록을 얻을 수 있습니다.

````markdown
```yaml
key: value
```
````

> Jekyll 태그 `{% highlight %}`는 이 테마와 호환되지 않습니다.
{: .prompt-danger }

#### Line Number

기본적으로 `plaintext`, `console`, `terminal`을 제외한 모든 언어는 줄 번호를 표시합니다. 

코드 블록의 줄 번호를 숨기려면 `nolineno` 클래스를 추가합니다.

```shell
echo 'No more line numbers!'
```
{: .nolineno }

````markdown
```shell
echo 'No more line numbers!'
```
{: .nolineno }
````

#### Specifying the Filename

코드 블록의 맨 위에 코드 언어가 표시되는 것을 알아차렸을 것입니다.

파일 이름으로 대체하려면 `file` 속성을 추가하여 이를 달성할 수 있습니다.

````markdown
```shell
# content
```
{: file="path/to/file" }
````

#### Liquid Codes

**Liquid** 언어를 표시하려면 liquid 코드를 `{% raw %}`와 `{% endraw %}`로 둘러싸세요.

````markdown
{% raw %}
```liquid
{% if product.title contains 'Pack' %}
  This product's title contains the word Pack.
{% endif %}
```
{% endraw %}
````

또는 `render_with_liquid: false`를 게시물의 YAML 블록에 추가합니다(Jekyll 4.0 이상 필요).

## Mathematics

우리는 수학 공식을 생성하기 위해 [**MathJax**][mathjax]를 사용합니다.

웹사이트 성능상의 이유로 `math` 기능은 기본적으로 로드되지 않습니다. 

_Front Matter_의 `math` 필드를 사용하여 사용자 정의할 수 있습니다.

[mathjax]: https://www.mathjax.org/

```yaml
---
math: true
---
```

`math` 기능을 활성화한 후 다음 구문을 사용하여 수학 공식을 추가할 수 있습니다.

- **Block math**  `$$ math $$`로 추가해야 하며, `$$` 앞뒤에 **필수**로 빈 줄이 있어야 함
  - **Inserting equation numbering** `$$\begin{equation} math \end{equation}$$`로 추가해야 합니다.
  - **Referencing equation numbering** 공식 블록에서 `\label{eq:label_name}`로, 텍스트와 함께 인라인으로 `\eqref{eq:label_name}`로 수행해야 함(아래 예시 참조)
- **Inline math** (줄 단위) `$$ math $$`로 추가해야 하며, `$$` 앞뒤에 빈 줄이 없어야 함
- **Inline math** (목록 단위) `\$$ math $$`로 추가해야 함

```markdown
<!-- Block math, keep all blank lines -->

$$
LaTeX_math_expression
$$

<!-- Equation numbering, keep all blank lines  -->

$$
\begin{equation}
  LaTeX_math_expression
  \label{eq:label_name}
\end{equation}
$$

Can be referenced as \eqref{eq:label_name}.

<!-- Inline math in lines, NO blank lines -->

"Lorem ipsum dolor sit amet, $$ LaTeX_math_expression $$ consectetur adipiscing elit."

<!-- Inline math in lists, escape the first `$` -->

1. \$$ LaTeX_math_expression $$
2. \$$ LaTeX_math_expression $$
3. \$$ LaTeX_math_expression $$
```

> `v7.0.0`부터 ​​**MathJax** 구성 옵션이 `assets/js/data/mathjax.js`{: .filepath } 파일로 이동되었으며 [extensions][mathjax-exts]를 추가하는 등 필요에 따라 옵션을 변경할 수 있습니다.
> `chirpy-starter`를 통해 사이트를 빌드하는 경우, 해당 파일을 gem 설치 디렉토리에서 복사합니다(명령어 `bundle info --path jekyll-theme-chirpy`로 확인). 해당 파일을 저장소의 동일한 디렉토리로 복사합니다.
{: .prompt-tip }

[mathjax-exts]: https://docs.mathjax.org/en/latest/input/tex/extensions/index.html

## Mermaid

[**Mermaid**](https://github.com/mermaid-js/mermaid) 훌륭한 다이어그램 생성 도구입니다. 

_Front Matter_의 `mermaid` 필드를 사용하여 사용자 정의할 수 있습니다.

```yaml
---
mermaid: true
---
```

그러면 다른 마크다운 언어처럼 사용할 수 있습니다. 그래프 코드를 ```` ```mermaid ````와 ```` ``` ````로 묶습니다.

## Learn More

Jekyll 게시물에 대한 자세한 내용은 [Jekyll Docs: Posts](https://jekyllrb.com/docs/posts/)을 방문하세요.
