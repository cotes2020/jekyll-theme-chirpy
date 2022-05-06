---
title: SSR에서 window 객체가 없다고 할 때
author:
  name: Bean
  link: https://github.com/beanie00
date: 2022-04-09 16:32:00 +0800
categories: [Web frontend, React]
tags: [Catty, Next]
---

## typeof 사용
---
```html
if(window){...} // window is not definde 에러발생

if(typeof window !== undefined) {...} // 정의되지않은 window의 타입이기떄문에 undefied가 발생 -> 에러가 발생하지 않습니다.
```

&nbsp;
## useEffect 사용
---
```html
useEffect(()=>{
	// 안에서 window 객체를 사용
},[])
```

&nbsp;
## (Next.js라면) dynamic을 사용
---
```html
import dynamic from 'next/dynamic'

const ComponentsWithNoSSR = dynamic<{props:type;}>( // typescript에서 props를 전달할때 interface를 정의해줍니다.
  () => import('./components/Component'), // Component로 사용할 항목을 import합니다.
  { ssr: false } // ssr옵션을 false로 설정해줍니다.
);

const App = () => {
  return(
    <div>
    	<Components/> // 해당 컴포넌트는 SSR로 불러올 수 있습니다.

    	<ComponentsWithNoSSR/> // 해당 컴포넌트는 ssr:false이기 때문에 서버사이드 렌더를 하지않습니다.
    </div>
  )
};
```