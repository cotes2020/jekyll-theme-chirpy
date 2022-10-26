---
title: "React로 element를 생성하는 법"
author: cotes
categories: [study, react]
tag: [JSX]
math: true
mermaid: true
---

> **_React JS:_**
> 어플리케이션이 interactive하도록 만들어주는 libary

> **_react-dom:_**
> 모든 React element들을 HTML body에 둘 수 있도록 해줌
> `ineteractive한 UI`

### `JSX 사용 안함`
```javascript
const Element = React.createElement(
  'h1', // html tag
  {className: 'greeting'}, //property
  'Hello, world!' // contents
); // span 생성
ReactDOM.render(Element, root); // element를 root에 배치
```

### `JSX 사용` 
```jsx
const Element = (
  <h1 className='greeting'>
  Hello, world!
  </h1>
);
ReactDOM.render(Element, root)
```

#### 필요 라이브러리 : **[Babel](https://babeljs.io/)** - JavaScript compiler(JSX를 컴파일 해주는 역할)
```html
<script type='text/babel'></script>
```