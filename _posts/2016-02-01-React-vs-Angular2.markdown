---
layout:     post
title:      "「译」React vs Angular 2：冰与火之歌"
subtitle:   "React versus Angular 2: There Will Be Blood"
date:       2016-02-01 12:00:00
author:     "Hux"
header-img: "img/post-bg-re-vs-ng2.jpg"
header-mask: 0.3
catalog:    true
tags:
  - Web
  - JavaScript
  - 译
---

> 这篇文章转载自[我在知乎专栏「前端外刊评论」上发表的文章](http://zhuanlan.zhihu.com/FrontendMagazine/20549104)。


[Angular 2](https://angular.io/) 已经发布 Beta 版，而且似乎很有信心在 2016 年成为热门框架。是时候进行一场巅峰对决了，我们来看看它如何与 [React](https://facebook.github.io/react/) 这个 2015 年的新宠抗衡。


**免责声明：**我之前很喜欢使用 Angular 1，不过在 2015 年转到了 React。最近我也在 Pluralsight 上发布了一门关于 [React 和 Flux 的课程](https://www.pluralsight.com/courses/react-flux-building-applications)（[免费试学](http://app.pluralsight.com/signup)）。所以，**是的，我本人是有偏见的，但我不会偏袒任何一方。**

好了，我们开始吧，这场对决将会非常血腥。

![](https://cdn-images-1.medium.com/max/800/1*MRPl_SNuRGJchb6eOAnkSA.jpeg)

图片来源：[@jwcarrol](https://twitter.com/jwcarroll)

## 两者根本不具有可比性！

是的是的，Angular 是框架，React 是类库。所以有人觉得比较这两者没有逻辑性可言。大错特错！

> 选择 Angular 还是 React 就像选择直接购买成品电脑还是买零件自己组装一样。

两者的优缺点本文都会提及，我会拿 React 语法和组件模型跟 Angular 的语法和组件模型做对比。这就像是拿成品电脑的 CPU 跟零售的 CPU 做对比，没有任何不妥。

## Angular 2 的优点

我们先看 Angular 相对 React 有哪些优势。

#### **无选择性疲劳**

Angular 是一个完整的框架，本身就提供了比 React 多得多的建议和功能。而要用 React，开发者通常还需要借助别的类库来打造一个真正的应用。比如你可能需要额外的库来处理路由、强制单向数据流、进行 API 调用、做测试以及管理依赖等等。要做的选择和决定太多了，让人很有压力。这也是为什么 React 有那么多的入门套件的原因（我自己就写了两个：[1](https://github.com/coryhouse/react-flux-starter-kit)、[2](https://github.com/coryhouse/react-slingshot)）。

Angular 自带了不少主张，所以能够帮助你更快开始，不至于因为要做很多决定而无所适从。这种强制的一致性也能帮助新人更快适应其开发模式，并使得开发者在不同团队间切换更具可行性。

Angular 核心团队让我非常欣赏的一点是，他们拥抱了 TypeScript，这就造成了另一个优势。

#### TypeScript = 阳关大道

没错，并非所有人都喜欢 TypeScript，但是 Angular 2 毅然决然地选择了它确实是个巨大的优势。反观 React，网上的各种示例应用令人沮丧地不一致——ES5 和 ES6 的项目基本上各占一半，而且目前存在[三种不同的组件声明方式](http://jamesknelson.com/should-i-use-react-createclass-es6-classes-or-stateless-functional-components/)。这无疑给初学者造成了困惑。（Angular 还拥抱了装饰器（decorator）而不是继承（extends）——很多人认为这也是个加分项）。

尽管 Angular 2 并不强制使用 TypeScript，但显然的是，Angular 的核心团队默认在文档中使用 TypeScript。这意味着相关的示例应用和开源项目更有可能保持一致性。Angular 已经提供了[非常清晰的关于如何使用 TypeScript 编译器的例子](https://angular.io/docs/ts/latest/quickstart.html)。（诚然，目前[并非所有人都在拥抱 TypeScript](http://angularjs.blogspot.com/2015/09/angular-2-survey-results.html)，但我有理由相信等到正式发布之后，TypeScript 会成为事实上的标准）。这种一致性应该会帮助初学者避免在学习 React 时遇到的疑惑和选择困难。

#### 极少的代码变动

2015 年是 [JavaScript 疲劳](https://medium.com/@ericclemmons/javascript-fatigue-48d4011b6fc4#.559iqxb39)元年，React 可以说是罪魁祸首。而且 React 尚未发布 1.0，所以未来还可能有很多变数。React 生态圈依旧在快速地变动着，尤其是[各种 Flux 变种](https://github.com/kriasoft/react-starter-kit/issues/22)和[路由](https://github.com/rackt/react-router)。也就是说，你今天用 React 写的所有东西，都有可能在 React 1.0 正式发布后过时，或者必须进行大量的改动。

相反，Angular 2 是一个对已经成熟完整框架（Angular 1）的重新发明，而且经过仔细、系统的设计。所以 Angular 不大可能在正式发布后要求已有项目进行痛苦的代码变动。Angular 作为一个完整的框架，你在选择它的时候，也会信任其开发团队，相信他们会认真决定框架的未来。而使用 React，一切都需要你自己负责，你要自己整合一大堆开源类库来打造一个完整的应用，类库之间互不相干且变动频繁。这是一个令人沮丧的耗时工作，而且永远没有尽头。

#### **广泛的工具支持**

后面我会说，我认为 React 的 JSX 是非常耀眼的亮点。然而要使用 JSX，你需要选择支持它的工具。尽管 React 已经足够流行，工具支持不再是什么问题，但诸如 IDE 和 lint 工具等新工具还不大可能很快得到支持。Angular 2 的模版是保存在一个字符串或独立的 HTML 文件中的，所以不要求特殊的工具支持（不过似乎 Angular 字符串模版的智能解析工具已经呼之欲出了）。

#### Web Components 友好

Angular 2 还拥抱了 Web Component 标准。唉，真尴尬我居然一开始忘记提到这点了——最近我还发布了一门关于[Web Components 课程](https://www.pluralsight.com/courses/web-components-shadow-dom)呢！简单来说，把 Angular 2 组件转换成原生 Web Components 应该会比 React 组件容易得多。固然 Web Components 的[浏览器支持度依然很弱](http://jonrimmer.github.io/are-we-componentized-yet/)，但长期来看，对 Web Components 友好是很大的优势。

Angular 的实现有其自身的局限和陷阱，这正好让我过渡到对 React 优势的讨论。

### React 的优点

现在，让我们看看是什么让 React 如此与众不同。

#### **JSX**

JSX 是一种类似 HTML 的语法，但它实际上会被编译成 JavaScript。将标签与代码混写在同一个文件中意味着输入一个组件的函数或者变量时你将享受到自动补全的福利。而 Angular 基于字符串的模版就相形见绌了：很多编辑器都不会高亮它们（只会显示单色）、只有有限的代码补全支持，并且一直到运行时才会报错。并且，通常你也只能得到很有限的错误提示。不过，Angular 的团队[造了一个自己的 HTML 解析器来解决这个问题](https://github.com/angular/angular/issues/4417)。（叼叼叼！）

如果你不喜欢 Angular 的字符串模版，你可以把模版移到一个单独的文件里去。不过这样你就回到了我认为的“老样子”：你需要在自己脑袋里记住这两个文件的关联，不但没有代码自动补全，也没有任何编译时检查来协助你。这听起来可能并不算什么……除非你已经爱上了与 React 相伴的日子。在同一个文件中组合组件还能享受编译时的检查，大概是 JSX 最与众不同的地方之一了。


![](http://p5.qhimg.com/d/inn/8a99f370/2.jpg)

对比 Angular 2 与 React 在标签忘记闭合时是如何表现的。

关于为什么 JSX 是一个巨大的优势，可以看看 [JSX：硬币的另一面（JSX: The Other Side of the Coin）](https://medium.com/@housecor/react-s-jsx-the-other-side-of-the-coin-2ace7ab62b98#.5007n49wq). （P.S. 这是作者写的另一篇文章，如果大家希望我们可以把这篇也翻了，欢迎在评论区举手）

#### React 报错清晰快速

当你在 React 的 JSX 中不小心手抖打错时，它并不会被编译。这是一件非常美妙的事情：无论你是忘记闭合了标签还是引用了一个不存在的属性（property），你都可以立刻知道到底是哪一行出错了。**JSX 编译器会指出你手抖的具体行号**，彻彻底底加速你的开发。

相反，当你在 Angular 2 中不小心敲错了一个变量时，鸦雀无声。**Angular 2 并不会在编译时做什么，它会等到运行时才静默报错。**它报错得*如此之慢*，我加载完整个应用然后奇怪为什么我的数据没有显示出来呢？这太不爽了。

#### React 以 JavaScript 为中心

终于来了。这才是 React 和 Angular 的根本区别。**很不幸，Angular 2 仍然是以 HTML 而非 JavaScript 为中心的。**Angular 2 并没有解决它设计上的根本问题：

> Angular 2 继续把 “JS” 放到 HTML 里。React 则把 “HTML” 放到 JS 里。

这种分歧带来的影响真是再怎么强调也不为过。它们从根本上影响着开发体验。Angular 以 HTML 为中心的设计留下了巨大的缺陷。正如我在 [JSX：硬币的另一面](https://medium.com/@housecor/react-s-jsx-the-other-side-of-the-coin-2ace7ab62b98#.jqh5kkxlk) 中所说的，JavaScript 远比 HTML 要强大。因此，**增强 JavaScript 让其支持标签要比增强 HTML 让其支持逻辑要合理得多**。无论如何，HTML 与 JavaScript 都需要某种方式以粘合在一起。React 以 JavaScript 为中心的思路从根本上优于 Angular、Ember、Knockout 这些以 HTML 为中心的思路。

让我们来看看为什么。

#### React 以 JavaScript 为中心的设计 = 简约

Angular 2 延续了 Angular 1 试图让 HTML 更加强大的老路子。所以即使是像循环或者条件判断这样的简单任务你也不得不使用 Angular 2 的独特语法来完成。例如，Angular 2 通过两种语法同时提供了单向数据绑定与双向数据绑定，可不幸的是它们实在差得有点多：

{% raw %}

```hbs
{{myVar}}        //单向数据绑定
ngModel="myVar"  //双向数据绑定
```

{% endraw %}

在 React 中，数据绑定语法不取决于数据流的单双向（数据绑定的单双向是在其他地方处理的，不得不说我觉得理应如此）。不管是单向还是双向数据流，绑定语法都是这样的：

```js
{myVar}
```

Angular 2 的内联母版（inline master templates）使用了这样的语法：

{% raw %}
```hbs
<ul>
  <li *ngFor="#hero of heroes">
    {{hero.name}}
  </li>
</ul>
```
{% endraw %}

上面这个代码片段遍历了一组 hero，而我比较关心的几点是：

- 通过星号来声明一个“母版”实在是太晦涩了
- `hero` 前的英镑符号（`#`）用于声明一个局部模版变量。这个概念感觉非常鸡肋（如果你偏好不使用 `#`，你也可以使用 `var-` 前缀写法）
-  为 HTML 加入了循环语义的HTML 特性（attribute）`ngFor` 是 Angular 特有的东西

相比上面 Angular 2 的语法，React 的语法可是纯净的 JavaScript （不过我得承认下面的属性 `key` 是个 React 的私货）

```hbs
<ul>
  { heroes.map(hero =>
    <li key={hero.id}>{hero.name}</li>
  )}
</ul>
```

鉴于 JS 原生支持循环，React JSX 利用 JS 的力量来做到这类事情简直易如反掌，配合 `map`、`filter` 能做的还远不止此。

去看看 [Angular 2 速查表](https://angular.io/docs/ts/latest/guide/cheatsheet.html)？那不是 HTML，也不是 JavaScript……这叫 **Angular**。

> **读懂 Angular：** 学一大堆 Angular 特有的语法
>
> 读懂 React： 学 JavaScript

React 因为语法和概念的简约而与众不同。我们不妨品味下当今流行的 JS 框架/库都是如何实现遍历的：

{% raw %}
```
Ember     : {{# each}}
Angular 1 : ng-repeat
Angular 2 : ngFor
Knockout  : data-bind="foreach"
React     : 直接用 JS 就好啦 :)
```
{% endraw %}

除了 React，所有其它框架都用自己的专有语法重新发明了一个我们在 JavaScript 常见得不能再常见的东西：**循环**。这大概就是 React 的美妙之处，利用 JavaScript 的力量来处理标签，而不是什么奇怪的新语法。

Angular 2 中的奇怪语法还有点击事件的绑定：

```javascript
(click)="onSelect(hero)"
```

相反，React 再一次使用了普通的 JavaScript：

```javascript
onClick={this.onSelect.bind(this, hero)}
```

并且，鉴于 React 内建了一个模拟的事件机制（Angular 2 也有），你并不需要去担心使用内联语法声明事件处理器所暗含的性能问题。

为什么要强迫自己满脑子都是一个框架的特殊语法呢？为什么不直接拥抱 JS 的力量？

#### 奢华的开发体验

JSX 具备的代码自动补全、编译时检查与丰富的错误提示已经创造了非常棒的开发体验，既为我们减少了输入，与节约了时间。而配合上热替换（hot reloading）与时间旅行（time travel），你将获得前所未有的开发体验，效率高到飞起。

原文这里链了个 Youtube 上的视频：[Dan Abramov - Live React: Hot Reloading with Time Travel at react-europe 2015](https://www.youtube.com/watch?v=xsSnOQynTHs&feature=youtu.be)，大家自备梯子。

#### 担心框架的大小？

这里是一些常见框架/库压缩后的大小（[来源](https://gist.github.com/Restuta/cda69e50a853aa64912d)）：

- **Angular 2:** 566k (766k with RxJS)
- **Ember:** 435k
- [**Angular 1**](https://ajax.googleapis.com/ajax/libs/angularjs/1.4.8/angular.min.js)**:** 143k
- **React + Redux:** 139k

列出的都是框架级的、用于浏览器且压缩后的大小（但并未 gzip）。需要补充的是，Angular 2 的尺寸在最终版本发布时应该会有所减小。

为了做一个更真实的对比，我将 Angular 2 [官方教程](https://angular.io/docs/ts/latest/tutorial/)中的 Tour of Heroes 应用用 Angular 2 和 React（还用上了新的 [React Slingshot](https://github.com/coryhouse/react-slingshot) 入门套件）都实现了一遍，结果如何呢？


- [**Angular 2**](https://github.com/coryhouse/angular-2-tour-of-heroes/tree/master)**:** 764k 压缩后
- [**React + Redux**](https://github.com/coryhouse/react-tour-of-heroes)**:** 151k 压缩后

可以看到，**做一个差不多的东西，Angular 2 目前的尺寸是 React + Redux 的五倍还多**。重要的事情再说一遍，Angular 2 的最终版本应该会减重。

不过，我承认关于框架大小的担忧可能被夸大了：

> 大型应用往往至少有几百 KB 的代码，经常还更多，不管它们是不是使用了框架。开发者需要做很多的抽象来构建一个复杂的软件。无论这些抽象是来自框架的还是自己手写的，它都会对应用的加载性能造成负面影响。  
>
>  就算你完全杜绝框架的使用，许多应用仍然是几百 KB 的 JavaScript 在那。 — Tom Dale [JavaScript Frameworks and Mobile Performance](http://tomdale.net/2015/11/javascript-frameworks-and-mobile-performance/)

Tom 的观点是对的。像 Angular、Ember 这样的框架之所以更大是因为它们自带了更多的功能。

但是，我关心的点在于：很多应用其实用不到这种大型框架提供的所有功能。在这个越来越拥抱微服务、微应用、[单一职责模块（single-responsibility packages）](http://www.npmjs.com)的时代，**React 通过让你自己挑选必要模块，让你的应用大小真正做到量身定做**。在这个有着 200,000 个 npm 模块的世界里，这点非常强大。

#### React 信奉[Unix 哲学](https://en.wikipedia.org/wiki/Unix_philosophy).

React 是一个类库。它的哲学与 Angular、Ember 这些大而全的框架恰恰相反。你可以根据场景挑选各种时髦的类库，搭配出你的最佳组合。JavaScript 世界在飞速发展，React 允许你不断用更好的类库去迭代你应用中的每个小部分，而不是傻等着你选择的框架自己升级。

Unix 久经沙场屹立不倒，原因就是：

> 小而美、可组合、目的单一，这种哲学永远不会过时。

React 作为一个专注、可组合并且目的单一的工具，已经被[全世界的各大网站们](https://github.com/facebook/react/wiki/Sites-Using-React)使用，预示着它的前途光明（当然，Angular 也被用于[许多大牌网站](https://www.madewithangular.com/#/)）。

#### 谢幕之战

Angular 2 相比第一代有着长足的进步。新的组件模型比第一代的指令（directives）易学许多；新增了对于同构／服务器端渲染的支持；使用虚拟 DOM 提供了 3-10 倍的性能提升。这些改进使得 Angular 2 与 React 旗鼓相当。不可否认，它功能齐全、观点鲜明，能够显著减少 “JavaScript 疲劳” 。

不过，Angular 2 的大小和语法都让我望而却步。Angular 致力的 HTML 中心设计比 React 的 JavaScript 中心模型要复杂太多。在 React 中，你并不需要学习 `ng-什么什么` 这种框架特有的 HTML 补丁（shim），你只要写 JavaScript 就好了。这才是我相信的未来。

### 著作权声明

本文译自 [Angular 2 versus React: There Will Be Blood](https://medium.freecodecamp.com/angular-2-versus-react-there-will-be-blood-66595faafd51#.v4y4euy1r)，其实[之前有人翻译过](http://www.w3ctech.com/topic/1675?from=timeline&isappinstalled=0)，但是翻得水平有一点不忍直视，我们不希望浪费这篇好文章。  
本文由 [@李凌豪](https://www.zhihu.com/people/li-ling-hao) [@黄玄](https://www.zhihu.com/people/huxpro) 联合翻译，首次发布于[前端外刊评论 · 知乎专栏](http://zhuanlan.zhihu.com/FrontendMagazine)，转载请保留原文链接 ;)
