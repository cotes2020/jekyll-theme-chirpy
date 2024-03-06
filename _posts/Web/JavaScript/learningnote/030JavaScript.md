
# Guides

[toc]

---

## DOM

DOM, user interface.
- page is represented by the DOM


**DOM object**
- **document** - the root of the tree:
  - document(.URL.hright.links.bgColor)
- **element** - node of the tree
  - returned by a member of the API
- **nodelist** - an array of the element
  - document.getElementByTagName('p')
  - document.getElementByClassName(class)
  - document.getElementById(id)
  - element.innerHTML
  - element.style
  - element.setAttribute(attribute,value)
  - element.removeAttribute(attribute)
- **attribute**
  - a node in the DOM, rarely use,
  - another way to manipulate/change the document

JavaScript
-  use the **APIs**.
-  read and write HTML elements:
-  works so well with **the structure used to create HTML documents** DOM.
   -  use the DOM to interact with the document.
   -  Every webpage is a **mathematical tree structure - the Document Object Model (DOM)**. the parent.
   -  Each **HTML tag** is a **node** in the tree. the child.
   -  nodes have all types of different **attributes**, such as text, background color, width, etc.
- JavaScript can go through web page and search and find certain elements. It can add new elements. It can delete other elements from the DOM.
- react to mouse clicks, page reloads, and other actions that the user have.

HTML5 an CSS3 are not really temporary changed
- react by user action.


API application programming interface
- An API is a way for someone else to write code, and then you to **use it, to interact with their programs**.
- no matter which javascript, the API always the same.

---

## JavaScript

- a **Dynamic client-side scripting and programming language**
    - adds interactivity to website
    - allows to implement complex things on web pages. `displaying timely content updates, interactive maps, animated 2D/3D graphics, scrolling video jukeboxes, or more`
    - Build very interactive user interfaces with frameworks like React.

- A **high-level interpreted language**, dont need to deal with the memory management like the lower language C or C++.

- Run directlly, no compiler.
- Conforms to the ECMAScript specification
- Multi-paradigm: object-oriented code or functional code.

- Run on the client/browser as well as on the server(Node.js)
    - It is the programming language of the browser. If to do client-side programming, code runs on client machine.
    - Python, Java, PHP, c-sharp all run on server-side.

- Used in building
    - very fast server side and full stack applictaions.(Node.js)
    - Mobile development (React Native, NativeScript, Ionic)
    - Desktop application development (Electron JS)


**js output**:
- js dont have a build in print function.
- data is typically displayed via:
  - alerts box: `window.alert();`
    - say hi
  - prompts: `window.prompt();`
    - ask for input
  - HTML output: `document.write()`
    - write directly to the page
    - but may overwrite other thing
    - `document.write(<h1>hello</h1>)`
    - `document.write("<h1>"+name+"</h1>")`
  - HTML output: `ObjectName.innertHTML=text`
    - change the contents of the DOM, conbined innerHTML with element want to change.
    - `document.getElementById('test).innerHTML - "hello";`
      - getElementById catch the first one.
      - id should always be unique
  - browser: `console.log()`
    - to see the execution of the program
    - left notice in browser, web wont see.
    - `console.log('hi');`





---

**HTML**: markup language to structure and give meaning to web content:
- defining paragraphs, headings, and data tables, embedding images and videos in the page.

```
<p>Player 1: Chris</p>
```

**CSS** : a language of style rules to apply styling to HTML content
- for example setting background colors and fonts, and laying out our content in multiple columns.

```js
p {
  font-family: 'helvetica neue';
  letter-spacing: 1px;
}
```

**JavaScript**: a scripting language to create `dynamically updating content`, control multimedia, animate images, and pretty much everything else.

```js
const para = document.querySelector('p');

para.addEventListener('click', updateName);

function updateName() {
  let name = prompt('Enter a new name');
  para.textContent = 'Player 1: ' + name;
}
```

---

### API

The core client-side JavaScript language consists of some common programming features that allow you to do things like:
- Store useful values inside variables.
- Operations on pieces of text.
- Running code in response to certain events occurring on a web page: `click event`
- ...

**Application Programming Interfaces (APIs)**
- functionality built on top of the client-side JavaScript language.
- APIs are `ready-made sets of code building blocks` that allow a developer to implement programs that would otherwise be hard or impossible to implement.



<img alt="pic" src="https://www.redhat.com/cms/managed-files/styles/wysiwyg_full_width/s3/API-page-graphic.png?itok=5zMemph9">

图书发行公司：
- 提供一个成本高昂应用，供书店店员查看书的库存情况。
- 提供一个 API 来查询库存情况。
- 或许会有第三方使用某个公共 API 来开发应用，以便人们直接从该发行商处（而非书店）购书。这样就能为该图书发行商打开新的收入渠道。与特定合作伙伴或全世界共享 API 能带来积极的影响。公开技术可以带来意外之喜。有时，这些惊喜更会颠覆整个行业。

既开放资源访问权限，又能确保IT安全性, 继续握有控制权,如何以及向谁开放访问权限。

**Browser APIs** are built into web browser, and able to expose data from the surrounding computer environment, or do useful complex things. For example:

- - The **DOM (Document Object Model) API** allows you to` manipulate HTML and CSS`, creating, removing and changing HTML, dynamically applying new styles to your page, popup window appear on page, new content displayed for example, that's the DOM in action.
- - The **Geolocation API** `retrieves geographical information`. This is how Google Maps is able to find your location and plot it on a map.
- - The **Canvas and WebGL APIs** allow you to create animated 2D and 3D graphics.
- **Audio and Video APIs** like `HTMLMediaElement` and `WebRTC` allow you to do really interesting things with multimedia, such as play audio and video right in a web page, or grab video from your web camera and display it on someone else's computer (Snapshot demo).

**Third party APIs** are not built into the browser by default, and you generally have to grab their code and information from somewhere on the Web. For example:
- The **Twitter API** allows you to do things like displaying your latest tweets on your website.
- The **Google Maps API** and **OpenStreetMap API** allows you to embed custom maps into your website, and other such functionality.


### js on  pages

When load a web page in browser, it is running code (the HTML, CSS, and JavaScript) inside an `execution environment` (the browser tab).

<img alt="pic" src="https://mdn.mozillademos.org/files/13504/execution.png">

- A very common use of JavaScript is to `dynamically modify HTML and CSS` to `update a user interface`, via the **Document Object Model API**.
- Note that the code in your web documents is generally loaded and executed in the order it appears on the page. If the JavaScript loads and tries to run before the HTML and CSS it is affecting has been loaded, errors can occur.


### Browser security
- Each browser tab has its own separate bucket for running code in (`execution environments`)
- means that in most cases the code in each tab is run completely separately, and `the code in one tab cannot directly affect the code in another tab` — or on another website.
- This is a good security measure — if this were not the case, then pirates could start writing code to steal information from other websites, and other such bad things.


### Server-side versus client-side code
**Client-side code**: codes run on the user's computer
- when a web page is viewed, the page's `client-side code is downloaded, then run and displayed by the browser`.

**Server-side code**: run on the server, then its `results are downloaded and displayed in the browser`.


### Dynamic versus static code
**dynamic**: the ability to update the display of a web page/app to show different things in different circumstances, generating new content as required.
- `Server-side code dynamically generates new content on the server`, e.g. pulling data from a database,
- `client-side JavaScript dynamically generates new content inside the browser on the client`, e.g. creating a new HTML table, filling it with data requested from the server, then displaying the table in a web page shown to the user.


**static**: just shows the same content all the time. with no dynamically updating content


---

## add JavaScript to page

### Internal JavaScript

use the `script` tag.

```js
<head>
    <script>
        JavaScript...
    </script>
</head>
```

```html
<head>
  <script>
    function message(){alert('hi')};
  </script>
</head>
<body>
  <h1></h1>
  <script>
    message();
  </script>
</body>
```


### External JavaScript
```js
<head>
<script src="main.js" defer></script>
</head>
```

### Inline JavaScript handlers
- JavaScript code living inside HTML.
```js
function createParagraph() {
  let para = document.createElement('p');
  para.textContent = 'You clicked the button!';
  document.body.appendChild(para);
}
<button onclick="createParagraph()">Click me!</button>
```

---

### Script loading strategies

all the HTML on a page is loaded in the order in which it appears.
- If using JavaScript to manipulate elements on the page (the Document Object Model), code won't work if the JavaScript is loaded and parsed before the HTML you are trying to do something to.

the **internal and external JavaScript**, JavaScript is loaded and run in `the head of the document`, before the HTML body is parsed. could cause error.
- The JavaScript will not run until after that event is fired, therefore the error is avoided.


#### **external JavaScript**

`defer` attribute: tells the browser to continue downloading the HTML content once the `<script>` tag element has been reached.

```html
<script src="script.js" defer></script>
```

#### **old-fashioned solution**
put script at the bottom of the body
- it would load after all the HTML has been parsed.
- The problem: loading/parsing of the script is completely blocked until the HTML DOM has been loaded. On larger sites with lots of JavaScript, this can cause a major performance issue, slowing down your site.


### **Scripts loaded using async and defer**

async and defer both instruct the browser to `download the script(s) in a separate thread, while the rest of the page (the DOM, etc.) is downloading`, so the page loading is not blocked by the scripts.
-  `async`: script that should be run immediately and don't have any dependencies, then use .
- `defer`: need to wait for parsing and depend on other scripts and/or the DOM being in place, load them using defer and put their corresponding `<script>` elements in the order you want the browser to execute them.
Comments

1. `async` attribute: download the script without blocking rendering the page and will execute it as soon as the script finishes downloading.
```js
<script async src="js/vendor/jquery.js"></script>
<script async src="js/script2.js"></script>
<script async src="js/script3.js"></script>
```
- but no guarantee that scripts will run in order, only that they will not stop the rest of the page from displaying.

- use when have a bunch of background scripts to load in, and just want them in place asap.
    - when the scripts in the page run independently from each other and depend on no other script on the page.
    - For example, maybe you have some game data files to load, which will be needed when the game actually begins, but for now you just want to get on with showing the game intro, titles, and lobby, without them being blocked by script loading.

2. `defer` attribute: run in the order they appear in the page and execute them as soon as the script and content are downloaded:
```js
<script defer src="js/vendor/jquery.js"></script>
<script defer src="js/script2.js"></script>
<script defer src="js/script3.js"></script>
```
- load in the order they appear on the page.  sure that jquery.js will load before script2.js and script3.js and that script2.js will load before script3.js.
- They won't run until the page content has all loaded, which is useful if your scripts depend on the DOM being in place (e.g. they modify one of more elements on the page).


---

## Troubleshooting JavaScript

**Syntax errors**: These are spelling errors in your code that actually cause the program not to run at all, or stop working part way through

**Logic errors**: These are errors where the syntax is actually correct but the code is not what you intended it to be, meaning that `program runs successfully but gives incorrect results`. These are often harder to fix than syntax errors, as there usually isn't an error message to direct you to the source of the error.

---
