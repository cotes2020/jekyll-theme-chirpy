---
title: 웹앱 클립보드 리스너 만들기
author: Beanie
date: 2022-04-28 16:32:00 +0800
categories: [Web frontend, React]
tags: [Catty]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover: assets/img/post_images/catty_cover2.png
---

Catty 웹앱에서는 아래 영상처럼 유저가 웹사이트 url을 복사하면 이를 화면에 띄워 바로 url을 추가할 수 있도록 하는 기능이 있다.

<div style="text-align: left">
   <video controls width="100%">
      <source src="/assets/img/post_images/clipboard_listener.mov" type="video/mp4"/>
   </video>
</div>

&nbsp;

Catty 서비스 말고도 네이버 등의 서비스에서도 클립보드에 복사해 둔 url을 가져와 그 주소로 바로 이동하게 해준다. 이런 기능은 어떻게 구현하는 걸까? 이 클립보드 리스너 기능은 생각보다 간단한데, 사실 아래 코드가 전부이다.


```javascript
export function getClipboardData(setClipboardText, setClipboardType) {
  let prevText = ""
  setInterval(async () => {
    try {
      const newText = await navigator.clipboard.readText()
      if (newText.startsWith('http') && prevText != newText && !newText.startsWith('https://catty-serverless-test')) {
        setClipboardText(newText)
        setClipboardType('url')
      }
      prevText = newText
    } catch (e) {
    }
  }, 1000)
}
```

이 코드에서는 `setInterval` 함수를 통해 1초마다 클립보드의 텍스트를 확인하고 이 데이터가 url이면 화면에 띄우도록 하였다.

브라우저에서 JavaScript를 사용하여 클립보드를 데이터를 쓰거나 읽으려면 ClipboardAPI를 사용하면 된다. ClipboardAPI는 Promise 기반으로 클립 보드 내용을 비동기식으로 접근할 수 있는 새로운 API이다. 하지만 비교적 최신 스펙으로 아직 지원되지 않는 브라우저가 많다.

<div style="text-align: left">
   <img src="/assets/img/post_images/clipboard_listener1.png" width="100%"/>
</div>

클립보드에 저장된 텍스트 내용은 `navigator.clipboard.readText()`로 불러올 수 있다. 텍스트 말고 `read()` 함수를 이용해 이미지 등의 데이터도 가져올 수 있는데, 이는 다음에 구현하는 걸로 남겨두었다.

참고로, 이번에 구현한 `readText()` 는 Chrome66 이상에서, 임의 데이터를 가져오는 `read()` 함수는 Chrome 76 이상에서 지원된다.