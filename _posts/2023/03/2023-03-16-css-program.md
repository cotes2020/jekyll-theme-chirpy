---
title: "CSS 工程化方案"
date: 2023-03-16
permalink: /2023-03-16-css-program/
---
## 方案1：纯CSS or Style or ClassName


介绍：最简单的做法，一些工程化方案原理上，都是往style或者className上转换。


优点：兼容性好


缺点：容易互相污染style


## 方案2：css in js（运行时）


### 介绍和demo


介绍：在JS中，编写CSS样式，写法上是JS，但是最终还是转换成了CSS。


使用demo：


```typescript
import styled from '@emotion/styled'

const ErrorMessageRed = styled.div`
  color: red;
  font-weight: bold;
`;

function App() {
  return (
   <div>
    <ErrorMessageRed>
      hello ErrorMessageRed !!
    </ErrorMessageRed>
   </div>
  );
}

export default App;
```


### 优缺点


优点：

- 无全局样式污染。上面的样式代码，都进行了转换，将其变成了一个hash后的className，或者变成style，并将其挂到dom节点上，来让样式生效。
- 方便模块化管理。因为写在js里。
- 可以利用js变量，以及根据react compoent props来动态渲染。

缺点：

- 在SSR场景下，适配麻烦。
- 包体积变大，调试debug困难，不如直接用chrome调试style来的方便。
- 性能问题，无解。**频繁的插入 CSS 样式规则会迫使浏览器做更多的工作。**React 团队核心成员&React Hooks 设计者 [Sebasian](https://link.juejin.cn/?target=https%3A%2F%2Fgithub.com%2Fsebmarkbage) 写了一篇关于 CSS-in-JS 库如何与 React 18 一起工作的文章。

	> 他特别说到：在concurrent 渲染模式下，React 可以在渲染之间让出浏览器的控制权。如果你为一个组件插入一个新的 CSS 规则，然后 React 让出控制权，浏览器会检查这个新的规则是否作用到了已有的树上。所以浏览器重新计算了样式规则。然后 React 渲染下一个组件，该组件发现一个新的规则，那么又会重新触发样式规则的计算


	**实际上 React 进行渲染的每一帧，所有 DOM 元素上的 CSS 规则都会重新计算**。这会非常非常的慢。


### 内外部样式序列化


样式序列化指的是 Emotion 将你的 CSS 字符串或者样式对象转化成可以插入文档的纯 CSS 字符串。Emotion 同时也会在序列化的过程中根据生成的存 CSS 字符串计算出相应的哈希值——这个哈希值就是你可以看到的动态生成的类名，比如 `css-an61r6`


下面是内外部写法的对比：


```typescript
function MyComponent() {
  return (
    <div
      css={{
        backgroundColor: 'blue',
        width: 100,
        height: 100,
      }}
    />
  );
}
```


```typescript
const myCss = css({
  backgroundColor: 'blue',
  width: 100,
  height: 100,
});

function MyComponent() {
  return <div css={myCss} />;
}
```


内部可以获取组件的props，发挥 css in js 的作用，但是性能有问题；外部性能没问题，但是没法获取props，无法两全。**所以性能无解。**


### 参考文档

- [https://juejin.cn/post/7158712727538499598](https://juejin.cn/post/7158712727538499598)
- [https://github.com/ascoders/weekly/blob/master/前沿技术/263.精读《我们为何弃用 css-in-js》.md](https://github.com/ascoders/weekly/blob/master/%E5%89%8D%E6%B2%BF%E6%8A%80%E6%9C%AF/263.%E7%B2%BE%E8%AF%BB%E3%80%8A%E6%88%91%E4%BB%AC%E4%B8%BA%E4%BD%95%E5%BC%83%E7%94%A8%20css-in-js%E3%80%8B.md)

## 方案3：css in js（编译时）


### 介绍和demo


vanilla-extract是编译时css-in-js框架。本身和 css modules 方案是一种东西。


写法demo：


```typescript
import { style } from '@vanilla-extract/css'

const demoStyle = style({
  display: 'block',
  padding: '10px'
})

const App = () => <div className={demoStyle}/>
```


### 优缺点


优点：

- 解决了运行时css-in-js方案的性能问题

缺点：

- 丢失了css-in-js运行时方案的灵活性

### 参考文档

- [https://zhuanlan.zhihu.com/p/546491378](https://zhuanlan.zhihu.com/p/546491378)
- [https://github.com/ascoders/weekly/blob/master/前沿技术/263.精读《我们为何弃用 css-in-js》.md](https://github.com/ascoders/weekly/blob/master/%E5%89%8D%E6%B2%BF%E6%8A%80%E6%9C%AF/263.%E7%B2%BE%E8%AF%BB%E3%80%8A%E6%88%91%E4%BB%AC%E4%B8%BA%E4%BD%95%E5%BC%83%E7%94%A8%20css-in-js%E3%80%8B.md)

## 方案4：css module


### 介绍和demo


这个是用的最多的css工程化解决方案。


写法demo（左边是sass代码，右边是icon效果图）：


![Untitled.jpeg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2023-03-16-css-program/d54fb0efa3bb37c6a1edf965b1e4159b.jpeg)


![Untitled.jpeg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2023-03-16-css-program/77e5a22b3f413cb282e00be719519d21.jpeg)


注意：`&` 就代表当前层级的className，这个className是被hash后的样式名。


除此之外，webpack 在配置时，默认只识别 `xxx.module.scss` 的css module文件。如果想识别任意后缀的css文件，需要自定义路径解析规则。


### 优缺点


优点：

- 性能没有问题
- 解决了css全局污染问题
- 基于sass提高css表达能力

缺点：

- 没法读取js代码

