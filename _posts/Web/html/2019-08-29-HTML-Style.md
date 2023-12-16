---
title: HTML - HTML Style
date: 2019-08-29 11:11:11 -0400
description: Learning Path
categories: [Web, HTML]
img: /assets/img/sample/rabbit.png
tags: [HTML]
---


# HTML - HTML Style

[toc]

---

## HTML Attributes
- ecommends lowercase in HTML,
- demands lowercase for stricter document types like XHTML.

**The `href` Attribute**： HTML links

`<a href="https://www.w3schools.com">This is a link</a>`


**The `src` Attribute**: The filename of the image source is specified in the src attribute:

`<img src="img_girl.jpg">`


**The `width` and `height` Attributes**

`<img src="img_girl.jpg" width="500" height="600">`


**The `alt` Attribute**:
- an alternative text to be used, useful if the image cannot be displayed or does not exist.
- The value of the alt attribute can be read by screen readers. This way, someone "listening" to the webpage, e.g. a vision impaired person, can "hear" the element.

`<img src="img_girl.jpg" alt="Girl with a jacket">`


**The `style` Attribute**
- specify the styling of an element, like color, font, size etc.

`<p style="color:red">This is a red paragraph.</p>`


**The `lang` Attribute**
- The language of the document can be declared in the `<html>` tag
- important for *accessibility applications (screen readers) and search engines*:

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<!DOCTYPE html>
<html lang="en-US">
<body>

</body>
</html>
```

**The `title` Attribute**
- title attribute is added to the `<p>` element.
- will be displayed as a tooltip when you mouse over the paragraph:

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<h2 title="I'm a header">The title Attribute</h2>
<p title="I'm a tooltip">Mouse over this paragraph, to display the title attribute as a tooltip.</p>
```

---

## HTML Styles

`<tagname style="property:value;">`


### The HTML `Style` Attribute

### CSS property

#### `background-color`
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<body style="background-color:powderblue;">

<h1>This is a heading</h1>
<p>This is a paragraph.</p>

</body>
```

---

## HTML Styles - `CSS` Cascading Style Sheets

CSS can be added to HTML elements in 3 ways:
- Inline - using the `style attribute` in HTML elements
- Internal - by using a `<style>` element in the `<head>` section
- External - by using an external CSS file

---

### 1. Inline CSS
- used to apply a unique style to a *single HTML element*.
- uses the `style` attribute of an HTML element.
- **avoid this**
  - keep the presentation the functionality and the styling completely separate or as much as possible.
  - inline css mix the presentation with Styling
  - not professional, not scalable, not practical.


<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<h1 style="color:blue;">This is a Blue Heading</h1>
```


---


### 2. Internal CSS `<head> <style> body {} h1 {} p {} </style> </head>`
- define a style for a *single HTML page*.
- defined in the `<head>` section of an HTML page, within a `<style>` element:

- **avoid**
  - only for that single html file, also fattens up the HTML file.

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<head>

  <style>
    body {background-color: powderblue;}

    h1 {
      color: blue;
      font-family: verdana;
      font-size: 300%;
    }
  </style>

</head>
```



---

### 3. External CSS `<head> <link rel="stylesheet" href="name.css"> </head>`
- define the style for *many HTML pages*.
- With an external style sheet, change entire web site, by changing one file!
- add a `<link>` to it in the `<head>` section of the HTML page:

![Screen Shot 2020-05-12 at 22.35.19](https://i.imgur.com/GT3XrCC.png)

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<head>
  <link rel="stylesheet" href="styles.css">
</head>
```

<<<<<<< HEAD
**crate a separate css file**: the `"styles.css"`:

```html
=======
**create a separate css file**: the `"styles.css"`:

```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
body {
  background-color: powderblue;
}
h1 {
  color: blue;
}
/p {
  color: red;
}
```

---

#### 4. External References
External style sheets can be referenced with a full URL or with a path relative to the current web page.

full URL to link to a style sheet:
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<link rel="stylesheet" href="https://www.w3schools.com/html/styles.css">
```

style sheet located in the html folder on the current web site:
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<link rel="stylesheet" href="/html/styles.css">
```

style sheet located in the same folder as the current page:
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<link rel="stylesheet" href="styles.css">
```

---

<<<<<<< HEAD
#### CSS: Classe and ID
=======
#### CSS: Classes and ID
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28

**id** is *unique*, wont's use it for other element.
**class** is may use for other element.

- re-use as needed
- across pages



1. The `id` Attribute
  - define a specific style for **one special element**,
  - add an `id` attribute to the element:
  - then define a style for the element with the specific `id`:

<<<<<<< HEAD
    ```html
=======
    ```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
    <p id="p01">I am different</p>
    <img src="cake,jpg" id="cakeimg" />

    #p01 {
      color: blue;
    }

    #cakeimg {
      flost: right;
    }
    ```


2. The `class` Attribute
   - define a style for **special types of elements**,
   - add a `class` attribute to the element:
   - then define a style for the elements with the specific `class`:

<<<<<<< HEAD
    ```html
=======
    ```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
    <p class="error">I am different</p>
    <li class="foodLi"> choco </li>

    p.error {
      color: red;
    }

    .foodLi {
      coolor: green;
    }
    ```


---


### CSS `Border` `Padding`

The CSS `border` property defines a border around an HTML element:

The CSS `padding` property defines a padding (space) between the text and the border:

The CSS `margin` property defines a margin (space) outside the border:

```
p {
  border: 1px solid powderblue;
  padding: 30px;  内圈框
  margin: 50px;   外圈框
}
```


#### `color`
- the text color for an HTML element:
  - 140 color name,
  - RGB `rgb(138,42,33)`,
  - Hex value `#8A2BE2`
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<h1 style="color:blue;">This is a heading</h1>
<p style="color:red;">This is a paragraph.</p>
```

#### `font-family` 字体
defines the font to be used for an HTML element:
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<h1 style="font-family:verdana;">This is a heading</h1>
<p style="font-family:courier;">This is a paragraph.</p>
```

#### `font-size` 字号
defines the text size for an HTML element:
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<h1 style="font-size:300%;">This is a heading</h1>
<p style="font-size:160%;">This is a paragraph.</p>
```

#### `text-align` 居中 text alignment
defines the horizontal text alignment for an HTML element:
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<h1 style="text-align:center;">Centered Heading</h1>
<p style="text-align:center;">Centered paragraph.</p>
```

---


## HTML Text Formatting

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28

<b> - Bold text
<strong> - Important text

<i> Italic text </i>
<em> Emphasized text </em>


<small> Smaller text </small>

<mark> Marked text 荧光笔标注 </mark>
<del> - Deleted text 划去
<ins> - Inserted text 下横线
<sub> - Subscript text 脚注的
<sup> - Superscript text 上标
```

---

## HTML Quotation and Citation Elements

### HTML `<q>` for Short Quotations
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<p>WWF's goal is to: <q>Build a future where people live in harmony with nature.</q></p>
```

### HTML `<blockquote>` for Quotations
defines a section that is quoted from another source.
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<p>Here is a quote from WWF's website:</p>
<blockquote cite="http://www.worldwildlife.org/who/index.html">
For 50 years, WWF has been protecting the future of nature.
The world's leading conservation organization,
WWF works in 100 countries and is supported by
1.2 million members in the United States and
close to 5 million globally.
</blockquote>

// 并不会断行，是一整句话
```


### HTML `<abbr>` for Abbreviations
defines an abbreviation or an acronym.

Marking abbreviations can *give useful information to browsers, translation systems and search-engines*.

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<p>The <abbr title="World Health Organization">WHO</abbr> was founded in 1948.</p>
```


### HTML `<address>` for Contact Information
defines contact information (author/owner) of a document or an article.

The `<address>` element is usually displayed in italic. Most browsers will add a line break before and after the element.

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<address>
Written by John Doe.<br>
Visit us at:<br>
Example.com<br>
Box 564, Disneyland<br>
USA
</address>
```

### HTML `<cite>` for Work Title
defines the title of a work.

Browsers usually display `<cite>` elements in italic.
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<p><cite>The Scream</cite> by Edvard Munch. Painted in 1893.</p>
```

### HTML `<bdo>` for Bi-Directional Override
defines bi-directional override. override the current text direction:
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<bdo>This text will be written from right to left</bdo>
正方形
<br>
<bdo dir="rtl">This text will be written from right to left</bdo>
反方向
```

---

## HTML Comments

###HTML Comment Tags

exclamation point (!) in the opening tag, but not in the closing tag.

not displayed by the browser, but can help document HTML source code.

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<!-- This is a comment -->
<p>This is a paragraph.</p>
<!-- Remember to add more information here -->
```

---

## HTML Colors

- color names
- html5 color names
- hexadecimal
- RGB

### `background-color` and `color`

<<<<<<< HEAD
```HTML
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28

<body style="background-color:powderblue;">

<h1 style="background-color:DodgerBlue;">Hello World</h1>
<p style="background-color:Tomato;">Lorem ipsum...</p>

<h1 style="color:Tomato;">Hello World</h1>
<p style="color:DodgerBlue;">Lorem ipsum...</p>
<p style="color:MediumSeaGreen;">Ut wisi enim...</p>

</body>

```


### `Border` Border Color

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<h1 style="border:2px solid Tomato;">Hello World</h1>
<h1 style="border:2px solid DodgerBlue;">Hello World</h1>
<h1 style="border:2px solid Violet;">Hello World</h1>
```

### color values

`rgba(255, 99, 71, 0.5)`

`#ff6347`

`hsl(hue, saturation, lightness)`

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<h1 style="background-color:rgb(255, 99, 71);">...</h1>
<h1 style="background-color:#ff6347;">...</h1>
<h1 style="background-color:hsl(9, 100%, 64%);">...</h1>

// 50% transparent
<h1 style="background-color:rgba(255, 99, 71, 0.5);">...</h1>
<h1 style="background-color:hsla(9, 100%, 64%, 0.5);">...</h1>
```


---
---

## HTML Links

### HTML Links - Syntax
an absolute URL (a full web address).
`<a href="https://www.w3schools.com/html/">Visit our HTML tutorial</a>`


### link pages Paths
External pages can be referenced with a full URL or with a path relative to the current web page.

full URL to link to a web page:
`<a href="https://www.w3schools.com/html/default.asp">HTML tutorial</a>`

a page located in the html folder on the current web site:
`<a href="/html/default.asp">HTML tutorial</a>`

a page located in the same folder as the current page:
`<a href="default.asp">HTML tutorial</a>`


### The `target` Attribute
The target attribute specifies where to open the linked document.

The target attribute can have one of the following values:

- `_blank` - Opens the linked document in a new window or tab
- `_self` - Opens the linked document in the same window/tab as it was clicked (this is default)
- `_parent` - Opens the linked document in the parent frame
- `_top` - Opens the linked document in the full body of the window 替代当前窗口
- `framename` - Opens the linked document in a named frame


### Image as a Link
It is common to use images as links:
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<a href="default.asp">
  <img src="smiley.gif" alt="HTML tutorial" style="width:42px;height:42px;border:0;">
</a>
```

### Button as a Link
To use an HTML button as a link, you have to add some JavaScript code.

JavaScript allows you to specify what happens at certain events, such as a click of a button:

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<button>HTML Tutorial</button>

<button onclick="document.location = 'default.asp'">HTML Tutorial</button>
```


### `title` attribute
The `title` attribute specifies extra information about an element. The information is most often shown as a tooltip text when the mouse moves over the element.

<a href="https://www.w3schools.com/html/" title="Go to W3Schools HTML section">Visit our HTML Tutorial</a>



### HTML Links - Different Colors

By default, a link will appear like this (in all browsers):

- An unvisited link is underlined and blue
- A visited link is underlined and purple
- An active link is underlined and red

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<style>
a:link {
  color: green;
  background-color: transparent;
  text-decoration: none;
}

a:visited {
  color: pink;
  background-color: transparent;
  text-decoration: none;
}

a:hover {
  color: red;
  background-color: transparent;
  text-decoration: underline;
}   移动到他的时候

a:active {
  color: yellow;
  background-color: transparent;
  text-decoration: underline;
}
</style>
```


#### Link Buttons
A link can also be styled as a button, by using CSS:

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<style>
a:link, a:visited {
  background-color: #f44336;        设置内圈颜色
  color: white;
  padding: 15px 25px;               把内圈做大
  text-align: center;
  text-decoration: none;
  display: inline-block;
}

a:hover, a:active {        移动到他的时候
  background-color: red;
}
</style>
```

---

## HTML Links - Create Bookmarks

### Create a Bookmark in HTML
Bookmarks can be useful if a web page is very long.

To create a bookmark - first create the bookmark, then add a link to it.

When the link is clicked, the page will scroll down or up to the location with the bookmark.

create a bookmark with the id attribute:
`<h2 id="C4">Chapter 4</h2>`

add a link to the bookmark ("Jump to Chapter 4"), from within the same page:
`<a href="#C4">Jump to Chapter 4</a>`

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<body>
<p><a href="#C4">Jump to Chapter 4</a></p>
<p><a href="#C10">Jump to Chapter 10</a></p>

<h2>Chapter 1</h2>
<p>This chapter explains ba bla bla</p>

<h2>Chapter 2</h2>
<p>This chapter explains ba bla bla</p>

<h2>Chapter 3</h2>
<p>This chapter explains ba bla bla</p>

<h2 id="C4">Chapter 4</h2>
<p>This chapter explains ba bla bla</p>
</body>
```

---

## HTML Images

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28

<img src="img_chania.jpg" alt="Flowers in Chania" width="460" height="345">

```

### Images

#### Width and Height, or Style?
The `width`, `height`, and `style` attributes are valid in HTML.

suggest using the `style` attribute. It prevents styles sheets from changing the size of images:


<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<!DOCTYPE html>
<html>

<head>
    <style>
    /* This stylesheet sets the width of all images to 100%: */
        img {width: 100%;}
    </style>
</head>

<body>
<img src="html5.gif" alt="HTML5 Icon" width="128" height="128">
<img src="html5.gif" alt="HTML5 Icon" style="width:128px;height:128px;">
</body>

</html>
```


#### location

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
Images in Another Folder
<img src="/images/html5.gif" alt="HTML5 Icon">

Images on Another Server:
<img src="https://www.w3schools.com/images/w3schools_green.jpg" alt="W3Schools.com">
```


#### Animated Images
HTML allows animated GIFs:
<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<img src="programming.gif" alt="Computer Man">
```


#### Image as a Link `<a href="url"> <img src="url" style="border:0;"> </a>`
use an image as a link, put the `<img>` tag inside the `<a>` tag:
- `border:0;` is added to prevent IE9 (and earlier) from displaying a border around the image (when the image is a link).

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<a href="default.asp">
  <img src="smiley.gif" alt="HTML tutorial" style="border:0;">
</a>
```


#### Image Floating `<img src="url" style="float:right;">`
let the image float to the right or to the left of a text:

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
float to the right of the text:
<p>
  <img src="smiley.gif" alt="Smiley face" style="float:right;">
  Hello
</p>

float to the left
<p>
  <img src="smiley.gif" alt="Smiley face" style="float:left">
  Hello
</p>
```


### Image Maps `<img src="url" alt="tag" usemap="#workmap">`
defines an image-map
- an image with clickable areas.
- may insert the `<map>` element anywhere,.

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<img src="workplace.jpg" alt="Workplace" usemap="#workmap">

<map name="workmap">
 <area shape="rect" coords="34,44,270,350" alt="Computer" href="computer.htm">
 <area shape="rect" coords="290,172,333,250" alt="Phone" href="phone.htm">
 <area shape="circle" coords="337,300,44" alt="Coffee" href="coffee.htm">
</map>
```

- `<img>`: add a usemap attribute:
- `<map>`: element is used to create an image map, and linked to the image by name attribute:
- `<area>` element: defined clickable area
    - define the shape of the area, and you can choose one of these values:
    - `rect` - defines a rectangular region
    - `circle` - defines a circular region
    - `poly` - defines a polygonal region
    - `default` - defines the entire region


#### Image Map and JavaScript
- A clickable area: a `link to another page`, or trigger a `JavaScript function`.
- Add a click event on the `<area>` element to execute a JavaScript function:

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<!DOCTYPE html>
<html>
<body>

<img src="workplace.jpg" alt="Workplace" usemap="#workmap" width="400" height="379">

<map name="workmap">
  <area shape="circle" coords="337,300,44" onclick="myFunction()">
</map>


<script>
function myFunction() {
  alert("You clicked the coffee cup!");
}
</script>

</body>
</html>

```


### HTML Background Images

1. text background color
<<<<<<< HEAD
```html
=======

```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<body>
    <div style="background-image: url('img_girl.jpg');">
      You can specify background images.
    </div>
</body>

or

<style>
    div {background-image: url('img_girl.jpg');}
</style>
```

<<<<<<< HEAD
2. 网页页面背景：
```html
=======
1. 网页页面背景：

```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<style>
    body {
      background-image: url('img_girl.jpg');
      background-repeat: no-repeat;            // avoid the repeating
      background-attachment: fixed;            // 拉伸至全屏
      background-size: cover;
      background-size: 100% 100%               // stretch to fit the entire image
    }
</style>
```

#### Background Image
- add a background image **on an HTML element**,
- **use the HTML style attribute and the CSS background-image property**

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<body>
    <div style="background-image: url('img_girl.jpg');">
      hello.         // 只有字段大小的图片显示出来
    </div>
</body>
```

Specify the background image **in the style element:**
<<<<<<< HEAD
```html
=======

```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<style>
    div {background-image: url('img_girl.jpg');}
</style>
```

**the entire page to have a background image**
specify the background image on the `<body>` element:
<<<<<<< HEAD
```html
=======

```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<style>
    body {background-image: url('img_girl.jpg');}
</style>
```


#### Background Repeat
- If the background image is smaller than the element,
<<<<<<< HEAD
- defalut, the image will repeat itself, horizontally and vertically, until it reaches the end of the element:
=======
- default, the image will repeat itself, horizontally and vertically, until it reaches the end of the element:
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
- avoid the repeating: `background-repeat: no-repeat;`


#### Background Cover
- want the background image cover the entire element, set the property `background-size: cover`. 拉伸至全屏
- make sure the entire element is always covered, set the property `background-attachment: fixed`:


#### Background Stretch
- want the background image stretch to fit the entire image in the element, you can set the property `background-size: 100% 100%`:


### HTML Picture Element
- display different pictures for different devices or screen sizes.
- the browser can choose the image that best fits the current view and/or device.

<<<<<<< HEAD
```html
=======
```js
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
<picture>
  <source media="(min-width: 650px)" srcset="img_food.jpg">
  <source media="(min-width: 465px)" srcset="img_car.jpg">
  <img src="img_girl.jpg">    这个在第一，则只显示他
</picture>
```

**When to use the Picture Element**
2 main purposes for the `<picture>` element:

1. Bandwidth
    - If you have a small screen or device, it is not necessary to load a large image file. The browser will use the first <source> element with matching attribute values, and ignore any of the following elements.

2. Format Support
    - Some browsers or devices may not support all image formats. By using the `<picture>`, add images of all formats, and the browser will use the first format it recognizes and ignore any of the following.


---



























.
