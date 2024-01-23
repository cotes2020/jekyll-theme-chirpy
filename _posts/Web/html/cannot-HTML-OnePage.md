---
title: HTML - HTML5 OnePage
date: 2019-08-29 11:11:11 -0400
description: Learning Path
categories: [Web, HTML]
img: /assets/img/sample/rabbit.png
tags: [OnePage]
---

# HTML5 OnePage

[toc]

---

## introduction

HTML
- Hyper Text Markup Language
- HTML describes the structure of a Web page
- HTML consists of a series of elements
- HTML elements tell the browser how to display the content
- HTML elements are represented by tags
- HTML tags label pieces of content such as "heading", "paragraph", "table", and so on
- Browsers do not display the HTML tags, but use them to render the content of the page

![Screen Shot 2020-05-12 at 20.24.54](https://i.imgur.com/kPTQ6qH.png)

使用 HTML 来建立自己的 WEB 站点，HTML 运行在浏览器上，由浏览器来解析

- `<!DOCTYPE html>` : 声明为 HTML5 文档
- **Metadata elements**
  - `<html>` : HTML页面的根元素
    - contains all other elements
    - tells a browser that it should use the HTML standard in displaying the web page
  - `<head>` :
    - contains general information about the page.
    - includes the title Information about scripts and information about displaying the page using CSS.
    - 包含了文档的元（meta）数据，
    - 如 `<meta charset="utf-8">` 定义网页编码格式为 utf-8。
  - `<title>` :
    - 描述了文档的标题
    - title for the page
    - must be nested between the `<head>``/<head>`.
  - `<base>` :	定义了页面链接标签的默认链接地址
  - `<link>` :	定义了一个文档和外部资源之间的关系
  - `<meta>` :	定义了HTML文档中的元数据
  - `<script>` :	定义了客户端的脚本文件
  - `<style>` :	定义了HTML文档的样式文件
- **sectioning elements**
- `<body>` : define the body of the whole webpage,只有 body 区域才会在浏览器中显示。
- `<h1>` : header
- `<div>`:
  - section or division of a web page.
  - for grouping elements together to use CSS styling.
- **text content**
  - `<p>` : 定义一个段落
  - `<hr>`	: 定义水平线
  - `<!--...-->`	: 定义注释
  - `<ul> <ol> <dl> <li>`

`<tagname>content goes here...</tagname>`

```
<html>
    <head>
        <title>Page title</title>
    </head>

    <body>
        <h1>This is a heading</h1>
        <p>This is a paragraph.</p>
        <p>This is another paragraph.</p>
    </body>
</html>
```

---

### HTML文档的后缀名
`.html`
`.htm`


### HTML
HTML 是用来描述网页的一种语言。
- 指的是超文本标记语言: HyperText Markup Language
- 不是一种编程语言，而是一种标记语言
- 标记语言是一套`标记标签 (markup tag)`
- 使用`标记标签`来描述网页
- 文档包含了`HTML` `标签`及`文本内容`
- `HTML文档`也叫做 `web 页面`


### The `<!DOCTYPE>` Declaration
- declaration represents the document type, and helps browsers to display web pages correctly.
- only appear once, at the top of the page (before any HTML tags).
- not case sensitive.
- The `<!DOCTYPE>` declaration for HTML5 is: `<!DOCTYPE html>`

#### 通用声明

HTML5 `<!DOCTYPE html>`
HTML 4.01
`<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">`
XHTML 1.0
`<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">`


### Empty HTML Elements
HTML elements with no content are called empty elements.
- `<br>` is an empty element without a closing tag, line break:
- `<p> a <br> paragraph with a line break.</p>`


### inlines vs block level Elements

![Screen Shot 2020-05-12 at 19.50.31](https://i.imgur.com/7rOUAfh.png)


### Web 浏览器
- Web浏览器用于读取HTML文件，并将其作为网页显示。
- 浏览器并不是直接显示的HTML标签，但可以使用标签来决定如何展现HTML页面的内容给用户：

---

## HTML 基础

### HTML Heading
`<h1> - <h6>`

```
<h1>这是一个标题</h1>
<h2>这是一个标题</h2>
<h3>这是一个标题</h3>
```

---


### HTML 段落 `<p>`

```
<p>这是一个段落。</p>
<p>这是另外一个段落。</p>
浏览器会自动地在段落的前后添加空行。（</p> 是块级元素）
```


### HTML Horizontal Rules `<hr>`

```
创建水平线。

<p>这是一个段落。</p>
<hr>
<p>这是一个段落。</p>
<hr>
<p>这是一个段落。</p>
```

### Line Breaks `<br>`


### The Poem Problem `<pre> </pre>`
This poem will display on a single line:

```
<pre>
  My Bonnie lies over the ocean.

  My Bonnie lies over the sea.
</pre>
```

---



### HTML 链接 `<a href="url"> </a>`

`<a href="https://www.runoob.com"> 这是一个链接 </a>`

---

### HTML 图像 `<img>`
`<img alt="pic" src="/images/logo.png" width="258" height="39" />`

### Example

1. `*.html` 文件跟 `*.jpg` 文件(f盘)在不同目录下：
`<img alt="pic" src="file:///f:/*.jpg" width="300" height="120"/>`

2. `*.html` 文件跟 `*.jpg` 图片在相同目录下：
`<img alt="pic" src="*.jpg" width="300" height="120"/> `

3. `*.html` 文件跟 `*.jpg` 图片在不同目录下：

    - 图片 `*.jpg` 在 image 文件夹中，`*.html` 跟 image 在同一目录下：
  `<img alt="pic" src="image/*.jpg/"width="300" height="120"/>`

    - 图片 `*.jpg` 在 image 文件夹中，`*.html` 在 connage 文件夹中，image 跟 connage 在同一目录下：
  `<img alt="pic" src="../image/*.jpg/"width="300" height="120"/>`

4. 如果图片来源于网络，那么写绝对路径：
`<img alt="pic" src="http://static.runoob.com/images/runoob-logo.png" width="300" height="120"/>`

#### HTML 中 href、src 区别

`href` Hypertext Reference 超文本引用。
- 用来建立当前元素和文档之间的链接。
- 常用的有：link、a。
- 例如：
- `<link href="reset.css" rel=”stylesheet“/>`

浏览器会识别该文档为 css 文档，并行下载该文档，并且不会停止对当前文档的处理。
这也是建议使用 link，而不采用 `@import` 加载 css 的原因。

`src` source 的缩写，src 的内容是页面必不可少的一部分，是引入。
- src 指向的内容会嵌入到文档中当前标签所在的位置。
- 常用的有：img、script、iframe。
- 例如:
- `<script src="script.js"></script>`
- 当浏览器解析到该元素时，会暂停浏览器的渲染，直到该资源加载完毕。
- 这也是将js脚本放在底部而不是头部得原因。

简而言之，src 用于 *替换当前元素*；href 用于 *在当前文档和引用资源之间建立联系*。

---

### HTML 元素语法
- 以开始标签起始
- 以结束标签终止
- 元素的内容是开始标签与结束标签之间的内容
- 某些 HTML 元素具有空内容（empty content）
- 空元素在开始标签中进行关闭（以开始标签的结束而结束）
- 大多数 HTML 元素可拥有属性

### 嵌套的 HTML 元素
HTML 文档由嵌套的 HTML 元素构成。

HTML 文档实例

```
<!DOCTYPE html>
<html>
  <body>
    <p>这是第一个段落。</p>
  </body>
</html>
```

`<p>` 元素: `<p>这是第一个段落。</p> `
- 这个元素定义了 HTML 文档中的一个段落。
- 这个元素拥有一个开始标签 `<p>` 以及一个结束标签 `</p>.`
- 元素内容是: 这是第一个段落。

`<body>` 元素:
- `<body>` 元素定义了 HTML 文档的主体。
- 这个元素拥有一个开始标签 `<body>` 以及一个结束标签 `</body>`。
- 元素内容是另一个 HTML 元素（p 元素）。

`<html>` 元素：
- `<html> `元素定义了整个 HTML 文档。
- 这个元素拥有一个开始标签` <html>` ，以及一个结束标签 `</html>`.
- 元素内容是另一个 HTML 元素（body 元素）。

### 不要忘记结束标签
即使您忘记了使用结束标签，大多数浏览器也会正确地显示 HTML：
`<p>这是一个段落`
以上实例在浏览器中也能正常显示，因为关闭标签是可选的。

但不要依赖这种做法。忘记使用结束标签会产生不可预料的结果或错误。

### HTML 空元素
没有内容的 HTML 元素被称为空元素。空元素是在开始标签中关闭的。

`<br>` 就是没有关闭标签的空元素（`<br>` 标签定义换行）。

在 XHTML、XML 以及未来版本的 HTML 中，所有元素都必须被关闭。

在开始标签中添加斜杠，比如 `<br />`，是关闭空元素的正确方法，HTML、XHTML 和 XML 都接受这种方式。

即使 `<br>` 在所有浏览器中都是有效的，但使用 `<br /> `其实是更长远的保障。

### 使用小写标签
HTML 标签对大小写不敏感：`<P>` 等同于 `<p>`。许多网站都使用大写的 HTML 标签。

---

## HTML 属性
属性是 HTML 元素提供的附加信息。
- HTML 元素可以设置属性
- 属性可以在元素中添加附加信息
- 属性一般描述于开始标签
- 属性总是以名称/值对的形式出现，比如：`name="value"`。

HTML 链接由 `<a>` 标签定义。链接的地址在 `href 属性`中指定：
`<a href="http://www.runoob.com">这是一个链接</a>`

### HTML 属性常用引用属性值
- 属性值应该始终被包括在`引号`内。
- 双引号是最常用的，不过使用`单引号`也没有问题。
- Remark提示: 在某些个别的情况下，比如属性值本身就含有双引号，那么您必须使用单引号，例如：`name='John "ShotGun" Nelson'`

### 使用小写属性
属性和属性值对大小写不敏感。

### HTML 属性参考手册

| 属性  | 描述                                                          |
| ----- | ------------------------------------------------------------- |
| class | 为html元素定义一个或多个类名（classname）(类名从样式文件引入) |
| id    | 定义元素的唯一id                                              |
| style | 规定元素的行内样式（inline style）                            |
| title | 描述了元素的额外信息 (作为工具条使用)                         |


### HTML 注释 `<!-- 这是一个注释 -->`
可以将注释插入 HTML 代码中，这样可以提高其可读性，使代码更易被人理解。浏览器会忽略注释，也不会显示它们。
`<!-- 这是一个注释 -->`


#### HTML 折行 `<p>这个<br>段落<br>演示了分行的效果</p>`
如果您希望在不产生一个新段落的情况下进行换行（新行），请使用 <br> 标签：
`<p>这个<br>段落<br>演示了分行的效果</p>`

`<br />` 元素是一个空的 HTML 元素。由于关闭标签没有任何意义，因此它没有结束标签。


### HTML 文本格式化

HTML 使用标签 `<b>("bold")` 与 `<i>("italic")` 对输出的文本进行格式, 如：粗体 or 斜体

这些HTML标签被称为格式化标签。

通常标签 `<strong>` 替换加粗标签 `<b>` 来使用, `<em>` 替换 `<i>`标签使用。
- 然而，这些标签的含义是不同的：
- `<b>` 与`<i>` 定义粗体或斜体文本。
- `<strong>` 或者 `<em>`意味着你要呈现的文本是重要的，所以要突出显示。
- 现今所有主要浏览器都能渲染各种效果的字体。不过，未来浏览器可能会支持更好的渲染效果。

HTML 文本格式化标签
| 标签       | 描述         |
| ---------- | ------------ |
| `<b>`      | 定义粗体文本 |
| `<em>`     | 定义着重文字 |
| `<i>`      | 定义斜体字   |
| `<small>`  | 定义小号字   |
| `<strong>` | 定义加重语气 |
| `<sub>`    | 定义下标字   |
| `<sup>`    | 定义上标字   |
| `<ins>`    | 定义插入字   |
| `<del>`    | 定义删除字   |

HTML "计算机输出" 标签
| 标签     | 描述               |
| -------- | ------------------ |
| `<code>` | 定义计算机代码     |
| `<kbd>`  | 定义键盘码         |
| `<samp>` | 定义计算机代码样本 |
| `<var>`  | 定义变量           |
| `<pre>`  | 定义预格式文本     |


HTML 引文, 引用, 及标签定义
| 标签           | 描述               |
| -------------- | ------------------ |
| `<abbr>`       | 定义缩写           |
| `<address>`    | 定义地址           |
| `<bdo>`        | 定义文字方向       |
| `<blockquote>` | 定义长的引用       |
| `<q>`          | 定义短的引用语     |
| `<cite>`       | 定义引用、引证     |
| `<dfn>`        | 定义一个定义项目。 |


### HTML 链接 `<a href="url">链接文本</a>`
HTML 使用超级链接与网络上的另一个文档相连。几乎可以在所有的网页中找到链接。点击链接可以从一张页面跳转到另一张页面。
HTML使用标签 `<a>`来设置超文本链接。
- "链接文本" 不必一定是文本。图片或其他 HTML 元素都可以成为链接。超链接可以是一个字，一个词，或者一组词，也可以是一幅图像，您可以点击这些内容来跳转到新的文档或者当前文档中的某个部分。
- 当把鼠标指针移动到网页中的某个链接上时，箭头会变为一只小手。
- 在标签 `<a>` 中使用了 `href属性` 来描述链接的地址。
- 默认情况下，链接将以以下形式出现在浏览器中：
  - 一个未访问过的链接显示为蓝色字体并带有下划线。
  - 访问过的链接显示为紫色并带有下划线。
  - 点击链接时，链接显示为红色并带有下划线。
注意：如果为这些超链接设置了 CSS 样式，展示样式会根据 CSS 的设定而显示。

`<a href="url">链接文本</a>`

#### HTML 链接 - target 属性
使用 target 属性，你可以定义被链接的文档在何处显示。

`target="_blank"` : 在新窗口打开文档

`<a href="https://www.runoob.com/" target="_blank" rel="noopener noreferrer"> 访问菜鸟教程! </a>`

#### HTML 链接- id 属性
id属性可用于创建在一个HTML文档书签标记。

提示: 书签是不以任何特殊的方式显示，在HTML文档中是不显示的，所以对于读者来说是隐藏的。

在HTML文档中插入ID:
`<a id="tips"> 有用的提示部分 </a>`

在HTML文档中创建一个链接到"有用的提示部分(id="tips"）"：
`<a href="#tips"> 访问有用的提示部分 </a>`

或者，从另一个页面创建一个链接到"有用的提示部分(id="tips"）"：
`<a href="https://www.runoob.com/html/html-links.html#tips"> 访问有用的提示部分 </a>`

基本的注意事项 - 有用的提示
注释： 请始终将正斜杠添加到子文件夹。
假如这样书写链接：`href="https://www.runoob.com/html"`，就会向服务器产生两次 HTTP 请求。
这是因为服务器会添加正斜杠到这个地址，然后创建一个新的请求，就像这样：`href="https://www.runoob.com/html/"`。

### HTML `<head>`
`<head>` 元素包含了所有的头部标签元素。在 `<head>` 元素中你可以插入脚本（scripts）, 样式文件（CSS），及各种meta信息。
- 可以添加在头部区域的元素标签为: `<title>, <style>, <meta>, <link>, <script>, <noscript>, and <base>`.

`<title>` 标签定义了不同文档的标题。
- `<title>` 在 HTML/XHTML 文档中是必须的。
- `<title>` 元素:
  - 定义了浏览器工具栏的标题
  - 当网页添加到收藏夹时，显示在收藏夹中的标题
  - 显示在搜索引擎结果页面的标题

#### HTML `<base>` 元素
描述了`基本的链接地址/链接目标`，该标签作为HTML文档中所有的`链接标签`的默认链接:

```
<head>
<base href="http://www.runoob.com/images/" target="_blank">
</head>
```

#### HTML `<link>` 元素
定义了文档与外部资源之间的关系。
通常用于链接到样式表:

```
<head>
<link rel="stylesheet" type="text/css" href="mystyle.css">
</head>
```

#### HTML `<style>` 元素
定义了HTML文档的样式文件引用地址.
在`<style>` 元素中你也可以直接添加样式来渲染 HTML 文档:

```
<head>
  <style type="text/css">
    body {background-color:yellow}
    p {color:blue}
  </style>
</head>
```

#### HTML `<meta>` 元素
- 提供了元数据.
- 元数据不显示在页面上，但会被浏览器解析。
- 通常用于指定网页的描述，关键词，文件的最后修改时间，作者，和其他元数据。
- 可以使用于浏览器（如何显示内容或重新加载页面），搜索引擎（关键词），或其他Web服务。
- 一般放置于 `<head>` 区域

为搜索引擎 定义关键词:
`<meta name="keywords" content="HTML, CSS, XML, XHTML, JavaScript">`

为网页 定义描述内容:
`<meta name="description" content="免费 Web & 编程 教程">`

定义网页作者:
`<meta name="author" content="Runoob">`

每30秒钟刷新当前页面:
`<meta http-equiv="refresh" content="30">`


#### HTML`<title>`元素不仅可以显示文本，也可以在左侧显示logo等图片。
显示时，要将`<link>`标签放入`<head>`里。

```
<!doctype HTML>
<html>
<head>
<link rel="shortcut icon" href="图片url">
<title>这是一个带图片的标签</title>
</head>
<body>
……
……
……
</body>
</html>
```

## HTML 样式- CSS
CSS (Cascading Style Sheets) 用于渲染HTML元素标签的样式.

```
<body>

<div style="opacity:0.5;position:absolute;left:50px;width:300px;height:150px;background-color:#40B3DF"></div>

<div style="font-family:verdana;padding:20px;border-radius:10px;border:10px solid #EE872A;">

  <div style="opacity:0.3;position:absolute;left:120px;width:100px;height:200px;background-color:#8AC007"></div>
  <h3>Look! Styles and colors</h3>
  <div style="letter-spacing:12px;">Manipulate Text</div>
      <div style="color:#40B3DF;">Colors
      <span style="background-color:#B4009E;color:#ffffff;">Boxes</span>
      </div>
  <div style="color:#000000;">and more...</div>

</div>

</body>
```

```
<body>
<div style="opacity:0.5;position:absolute;left:50px;width:300px;height:150px;background-color:#40B3DF"></div>
<div style="font-family:verdana;padding:20px;border-radius:10px;border:10px solid #EE872A;">
<div style="opacity:0.3;position:absolute;left:120px;width:100px;height:200px;background-color:#8AC007"></div>
<h3>Look! Styles and colors</h3>
<div style="letter-spacing:12px;">Manipulate Text</div>
<div style="color:#40B3DF;">Colors
<span style="background-color:#B4009E;color:#ffffff;">Boxes</span>
</div>
<div style="color:#000000;">and more...</div>
</div>
</body>
```





















.
