---
layout: post
title: "为什么 CSS 这么难学？"
subtitle: "Why I dislike CSS as a programming language"
author: "Hux"
header-img: "img/post-bg-css.jpg"
header-img-credit: "@WebdesignerDepot"
header-img-credit-href: "medium.com/@WebdesignerDepot/poll-should-css-become-more-like-a-programming-language-c74eb26a4270"
header-mask: 0.4
tags:
  - Web
  - CSS
  - 知乎
---

> 这篇文章转载自[我在知乎上的回答](https://www.zhihu.com/question/66167982/answer/240434582)

对我来说，CSS 难学以及烦人是因为它**「出乎我意料之外的复杂」**且让我觉得**「定位矛盾」**。

[@方应杭](//www.zhihu.com/people/b90c7eb6d3d5a4e2ce453dd8ad377672) 老师的答案我赞了：CSS 的属性互不正交，大量的依赖与耦合难以记忆。

[@顾轶灵](//www.zhihu.com/people/596c0a5fdd9b36cea06bac348d418824) [@王成](//www.zhihu.com/people/c02ec74a44ee4a6784d002c33e293652) 说得也没错：CSS 的很多规则是贯彻整个体系的，而且都记在规范里了，是有规律的，你应该好好读文档而不是去瞎试。


「**CSS是一门正儿八经的编程语言，请拿出你学C++或者Java的态度对待它**」

但是问题就在这了，无论从我刚学习前端还是到现在，我都没有把 CSS 作为一门正儿八经的编程语言（**而且显然图灵不完全的它也不是**），CSS 在我眼里一直就是一个布局、定义视觉样式用的 DSL，与 HTML 一样就是一个标记语言。

写 CSS 很有趣，CSS 中像继承、类、伪类这样的设计确实非常迎合程序员的思路，各种排列组合带来了很多表达上的灵活性。但如果可以选择，在生产环境里我更愿意像 iOS/Android/Windows 开发那样，把这门 DSL 作为 IDE WYSIWYG 编辑器的编译目标就可以了，当然你可以直接编辑生成的代码，但我希望「对于同一种效果，有比较确定的 CSS 表达方式」

因为我并不在 CSS 里处理数据结构，写算法、业务逻辑啊，我就是希望我能很精确得表达我想要的视觉效果就可以了。如果我需要更复杂的灵活性和控制，你可以用真正的编程语言来给我暴露 API，而不是在 CSS 里给我更多的「表达能力」


**CSS 语言本身的表达能力对于布局 DSL 来说是过剩的**，所以你仅仅用 CSS 的一个很小的子集就可以在 React Native 里搞定 iOS/Android 的布局了。你会发现各个社区（典型如 React）、团队都要花很多时间去找自己项目适合的那个 CSS 子集（so called 最佳实践）。而且 CSS 的这种复杂度其实还挺严重得影响了浏览器的渲染性能，很多优化变得很难做。

**而 CSS 的表达能力对于编程语言来说又严重不够**，一是语言特性不够，所以社区才会青睐 Less、Sass 这些编译到 CSS 的语言，然后 CSS 自己也在加不痛不痒的 Variable。二是 API 不够，就算你把规范读了，你会发现底层 CSSOM 的 Layout、Rendering 的东西你都只能强行用声明式的方式去 hack（比如用 transform 开新的 composition layer）而没有真正的 API 可以用，所以 W3C 才会去搞 Houdini 出来。

这种不上不下的感觉就让我觉得很「矛盾」，你既没法把 CSS 当一个很简单的布局标记语言去使用，又没办法把它作为一个像样的编程语言去学习和使用。


在写 CSS 和 debug CSS 的时候我经常处在一种「MD 就这样吧反正下次还要改」和「MD 这里凭什么是这样的我要研究下」的精分状态，可是明明我写 CSS 最有成就感的时候是看到漂亮的 UI 啊。

以上。
