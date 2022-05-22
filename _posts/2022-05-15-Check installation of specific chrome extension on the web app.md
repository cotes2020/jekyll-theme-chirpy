---
title: 웹앱에서 특정 크롬 익스텐션 설치 여부 확인하기
author: Beanie
date: 2022-05-15 02:21:00 +0800
categories: [Web frontend, React]
tags: [Catty, extension]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover: assets/img/post_images/catty_cover2.png
---

Catty 서비스는 크롬 익스텐션과 함께 사용해야 가장 편리하고 효과적으로 사용할 수 있다. 따라서 웹앱에 들어온 사람들 중에 아직 크롬 익스텐션을 설치하지 않은 사람들에게 아래와 같이 크롬 익스텐션을 설치하라는 알림을 띄우고자 하였다.
<div style="text-align: left">
   <img src="/assets/img/post_images/extension.png" width="100%"/>
</div>

그러기 위해서는 먼저 웹앱에서 해당 유저의 Catty 크롬 익스텐션 설치 여부를 파악할 수 있어야 한다.

이 기능은 크롬 익스텐션에서 이미지 하나를 웹에서 접근가능하게 설정해두고, 웹에서 해당 이미지를 불러오는 것을 시도한 다음, 이 시도의 성공/실패 여부를 판단하는 것으로 구현할 수 있다.

먼저, 크롬 익스텐션의 `manifest.json`에서 체크하려고 하는 이미지를 `web_accessible_resources`에 추가해준다.
```javascript
"web_accessible_resources": [
    "/images/Jcrop.gif",
    "/images/pixel.png"
],
```

그런 다음 웹앱에 다음 함수를 추가한다.

```javascript
useEffect(() => {
    detectExtension(CHROME_EXTENSION_ID)
  }, [])

  function detectExtension(extensionId, callback) {
    var img;
    img = new Image();
    img.src = "chrome-extension://" + extensionId + "/images/pixel.png";

    img.onload = function() {
      setExtensionInstalled(true)
    };
    img.onerror = function() {
      setExtensionInstalled(false)
    };
  }
```

이렇게 설정해두면 해당 함수가 있는 컴포넌트가 mount 될 때, 확인하고 싶은 CHROME_EXTENSION_ID의 images/pixel.png 이미지 로드를 시도한다. 이 때 로드가 실패하면 CHROME_EXTENSION_ID가 존재하지 않는 다는 뜻이므로 설치를 하라는 알림을 띄우면 된다.