---
title: When SSR says that there is no window object
author:
  name: Bean
  link: https://github.com/beanie00
date: 2022-04-09 16:32:00 +0800
categories: [Web frontend, React]
tags: [Catty, Next]
---

## Use typeof
---
```html
if(window){...} // window is not defined error occur

if(typeof window !== undefined) {...} // No error
```

&nbsp;
## Use useEffect
---
```html
useEffect(()=>{
	// Use window object inside useEffect
},[])
```

&nbsp;
## (if Next.js project) Use dynamic
---
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