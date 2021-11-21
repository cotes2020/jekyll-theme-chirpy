---
title: 글쓰기 요령 ( 이미지 업로드, HTML, CSS, JS 작성 방법 )
author: Shin Hyun
date: 2020-11-24 21:50:00 +0900
categories: [Exhibition,Others]
tags: [post,shinhyun]     # TAG names should always be lowercase, 띄어쓰기도 금지
---

### 이미지 업로드 방법
```
/assets/img/post/ 에 (해당 글 제목)/(이미지이름) 으로 저장하자. 
이미지 이름은 원하는대로 해도 된다.
# Example 
/assets/img/post/2020-11-24-how-to-post/pull-request.PNG
```

글에서 이미지가 보이게 하는 방법은 다음과 같다. 

예시로 how-to-post의 이미지 코드는 다음과 같다. 

```html
<img src="/assets/img/post/2020-11-24-how-to-post/pull-request.PNG" width="90%"> 
```

<img src="/assets/img/post/2020-11-24-how-to-post/pull-request.PNG" width="90%"> 

### HTML 작성 방법 
글 내에 html를 활용해 코드를 작성하고 싶다면, 이런식으로 쓸 수도 있다.

깃헙에서 이 글의 markdown 코드를 확인하여 어떻게 작성되었는지 확인해보자! 

<div>
    누르면 숫자가 증가하는 버튼
    <button id="counter">0</button>
    <script>
        cnt = 0;
        document.getElementById("counter").addEventListener("click",function(e) {
            cnt++;
            this.innerText = cnt;
        });
    </script>
</div>

```html
<div>
    <button id="counter">0</button>
    <script>
        cnt = 0;
        document.getElementById("counter").addEventListener("click",function(e) {
            cnt++;
            this.innerText = cnt;
        });
    </script>
</div>
```
