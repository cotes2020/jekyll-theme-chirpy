---
title: When SSR says that there is no window object
author: Beanie
date: 2022-04-09 16:32:00 +0800
categories: [Web frontend, React]
tags: [Catty, Next]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover: assets/img/post_images/next_cover.png
---

앞선 글에서 언급하였듯, 서버사이드에서는 브라우저 객체를 사용할 수 없다. 따라서 `window is not defined` 등의 에러를 종종 볼  수 있다. 이번 글에는 이렇게 서버사이드에서 `window is not defined` 류의 에러가 발생했을 때 해결할 수 있는 3가지 방법을 담았다.


## Use typeof
먼저, 서버사이드에서는 브라우저 객체가 없으므로 `typeof window`은 `undefined`이다.
따라서` typeof window !== undefined` 조건을 통해 서버사이드에서의 호출을 막을 수 있다.


```html
if(window){...} // window is not defined error occur

if(typeof window !== undefined) {...} // No error
```

&nbsp;
## Use useEffect

DOM 형성 후에 실행되는 hook이다. 따라서 useEffect 안의 코드는 서버 사이드가 아니라 브라우저에서 실행됨이 보장된다.

```html
useEffect(()=>{
	// Use window object inside useEffect
},[])
```

&nbsp;
## (if Next.js project) Use dynamic

마지막으로 Next.js를 사용한다면 Next.js 자체적으로 지원한느 방법인 dynamic 사용하여 해당 컴포넌트만 서버사이드 렌더링을 끄는 방법으로 해결할 수 있다.

```javascript
import dynamic from 'next/dynamic'

const ComponentsWithNoSSR = dynamic<{props:type;}>( // When passing props in typescript, an interface is defined.
  () => import('./components/Component'), // Import items to be used as components.
  { ssr: false } // Set ssr option as false.
);

const App = () => {
  return(
    <div>
    	<Components/> // Those components can be loaded as SSR.

    	<ComponentsWithNoSSR/> // Since that component is ssr:false, it doesn't do server-side rendering.
    </div>
  )
};
```