---
title: "유니티 x SOOP 채팅 연동"
# description: ""
categories: [작업물]
tags: [작업물, 유니티]
image: "/assets/img/background/20230112-151539.jpg"
hidden: true

date: 2024-04-25. 21:26
# last_modified_at: 2024-04-25. 21:26
last_modified_at: 2024-08-29. 21:45
---

## 머리말

---

### 참여 / 담당

### 사용한 툴

- Unity

## 기록

---

### Thread를 써보자

비동기 프로그래밍을 해볼 수 있는 기회가 생겼다.  

Dispose  

[참고: 멀티스레드(Multi-thread)](https://coderzero.tistory.com/entry/%EC%9C%A0%EB%8B%88%ED%8B%B0-C-%EA%B0%95%EC%A2%8C-23-%EB%A9%80%ED%8B%B0%EC%8A%A4%EB%A0%88%EB%93%9CMulti-thread)
[참고: Unity: 비동기와 코루틴을 혼동하지 않기](https://tistory.jeon.sh/59)

[참고: `ConcurrentQueue<T>` 클래스](https://learn.microsoft.com/ko-kr/dotnet/api/system.collections.concurrent.concurrentqueue-1?view=net-8.0)  
[참고: Queue VS ConcurrentQueue](https://karl27.tistory.com/66)  

[참고: C# Parallel](https://rito15.github.io/posts/cs-parallel/)  

### 이모티콘을 띄우자

TextMeshPro의 SpriteSheet 기능을 이용해서 이모티콘을 띄워보자.  
런타임에 이모티콘을 받아서 Sprite에 그려주고, 이를 TextMeshPro에서 띄워주면 된다.  

웹 요청을 보내면 `byte[]` 형식으로 이모티콘을 받아올 수 있다.  
이를 `Texture2D`로 변환하고, 이를 `Sprite`로 변환하면 된다.  

동적으로 Sprite에 그려내는 것은 위 링크를 참고하였다.  

문제는 움직이는 이모티콘과 OGQ 이모티콘을 어떻게 띄울지다.  
먼저 움직이는 이모티콘은 파일이 webp 형식이라, 이를 여러 장의 png로 변환해야 한다.  

[참고: 이모티콘 및 아이콘을 런타임에 동적으로 생성 또는 변경하기](https://doublsb.tistory.com/113)  
[참고: Texture2D Resize하기 (크기조정)](https://dallcom-forever2620.tistory.com/30)  
[참고: Converting a texture to a sprite](https://stackoverflow.com/questions/49241953/converting-a-texture-to-a-sprite)  
[참고: Is possible to insert animated sprites in TextMeshPro?](https://forum.unity.com/threads/is-possible-to-insert-animated-sprites-in-textmeshpro.1014472/)

### 최적화를 하자

부끄럽게도 프로파일러를 제대로 사용해본 적이 없다.  
