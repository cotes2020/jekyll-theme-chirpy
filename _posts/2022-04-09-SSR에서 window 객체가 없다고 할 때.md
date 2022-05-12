---
title: When SSR says that there is no window object
author: Bean
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

## Use typeof

```html
if(window){...} // window is not defined error occur

if(typeof window !== undefined) {...} // No error
```

&nbsp;
## Use useEffect

```html
useEffect(()=>{
	// Use window object inside useEffect
},[])
```

&nbsp;
## (if Next.js project) Use dynamic

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