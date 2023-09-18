---
title: React Intl로 내 사이트 국제화하기
date: 2023-08-05 20:00:00 +0900
categories:
  - React
tags:
  - Intl
---

## React Intl이란 무엇인가요? 🤷‍♀️
React Intl은 npm을 통해 설치할 수 있답니다! 이 라이브러리를 사용하면 다국어 서비스를 아주 쉽게 만들 수 있습니다. 근데 다국어라니, 사람들이 왜 이걸 원할까요? 🤔 글로벌한 사이트에는 다양한 국적의 사람들이 들어오고 이 때 사람들의 국가에 맞춘 사이트를 표시한다면 이용하기가 편해지기 때문입니다.

## locale: 이게 뭐에요? 🌏
`locale`이라는 건 간단히 말하자면 언어와 지역을 알려주는 고유 코드입니다. 한국어는 `ko`, 미국 영어는 `en-US` 같은 식으로요. 브라우저 설정에서 이 정보를 가져올 수 있고, `localStorage`에 저장해서 쓸 수도 있어요! 🍪

```javascript
const locale = navigator.language; // 또는 `navigator.languages[0]`
// 또는
const locale = localStorage.getItem("locale") ?? "ko";
```

## 메시지 관리는 어떻게 하나요? 💌
메시지 관리는 JSON 파일을 사용해서 아주 깔끔하게 할 수 있어요! 📁 여러 언어를 지원하려면, 해당 언어별로 JSON 파일을 만들어주면 되는 것이죠!

```json
{
  "title": "리액트 Intl",
  "info": "현재 언어는 {locale}입니다.",
  "label": "언어",
  "button": "저장"
}
```

이렇게 하면 여러 언어로 멋진 메시지를 뿌릴 수 있습니다! 🎉

## IntlProvider는 뭔가요? 🛡️
`<IntlProvider/>`는 전역적으로 다국어 지원을 하기 위해 필요한 요소입니다. 최상위 컴포넌트를 이것으로 감싸주면, 모든 하위 컴포넌트에서도 다국어를 쉽게 처리할 수 있답니다! 😇

```jsx
function App() {
  return (
    <IntlProvider locale={locale} messages={messages}>
      <Page />
    </IntlProvider>
  );
}
```

## FormattedMessage로 다국어 메시지 출력하기 📣
다국어 메시지 출력은 `<FormattedMessage/>` 컴포넌트로 아주 쉽게 할 수 있습니다. 이 컴포넌트에 `id` prop을 주면, 해당 메시지를 찾아서 보여줍니다!

```jsx
<FormattedMessage id="title" />
```

## 마무리 🌈
이렇게 React Intl을 활용하면 간편하고 쉽게 국제화가 가능합니다! 그럼 다음글에서 다시 만나요~ 🌟 🎉
