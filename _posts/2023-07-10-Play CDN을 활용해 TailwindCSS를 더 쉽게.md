---
title: Play CDN을 활용해 TailwindCSS를 더 쉽게
date: 2023-07-10 20:00:00 +0900
categories:
  - CSS
tags:
  - TailwindCSS
---

## TailwindCSS를 간단하게! Play CDN 소개 🎉

안녕하세요! 🎊 오늘은 대세 프론트엔드 프레임워크 TailwindCSS를 아주 쉽게 사용할 수 있는 방법, 바로 Play CDN에 대해 소개해 드릴게요.

## Play CDN은 뭐고 왜 필요한가요? 🤔

Play CDN은 TailwindCSS를 사용하고 싶지만, 설정 과정이 귀찮은 분들이나 초기 세팅에서 좀 막막해지시는 분들을 위한 최고의 솔루션이에요.🤩 이제 복잡한 설정 파일 만지작거리는 일은 그만! HTML 문서 안에서 `<script>` 태그 한 줄로 TailwindCSS를 사용할 수 있답니다!

```html
<head>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
```

## 잠깐만, 추가로 플러그인도 사용하고 싶다고요? 🤩

네, 그럼 쿼리 문자열을 이용해 `plugins` 파라미터를 명시하면 돼요! 이렇게요:

```html
<head>
  <script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>
</head>
```

## 꿀팁! 전역 설정은 어떻게 하나요? 🍯

이것도 쉬워요! 전역 속성에 설정 객체를 할당해주기만 하면 끝! 이렇게 해볼까요?

```html
<head>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            tomato: "tomato",
          },
        },
      },
    };
  </script>
</head>
```

## 추가 CSS 클래스가 필요하면? 🤔

이 부분도 너무 쉬워요. `<style>` 요소의 `type` 속성을 `text/tailwindcss`로 설정한 다음, `@layer` 지시문을 사용해 클래스를 추가하면 끝!

```html
<head>
  <style type="text/tailwindcss">
    @layer components {
      .btn {
        @apply rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-600;
      }
    }
  </style>
</head>
```

## 실습으로 배워보자! 👩‍💻👨‍💻

그런데 이것보다 더 쉬운 방법이 있어요! 바로 Tailwind Play라는 온라인 에디터를 사용하는 거에요. 설치 필요없이 간단하게 웹에서 테스트 해 볼 수 있답니다. TailwindCSS를 사용하는 데 복잡한 설정은 이제 안녕!😎
