---
title: HTML - JavaScript OnePage
date: 2020-08-29 11:11:11 -0400
description: Learning Path
categories: [Web, JavaScript]
img: /assets/img/sample/rabbit.png
tags: [OnePage, JavaScript]
---


# HTML - JavaScript OnePage

[toc]

---

## Variables
Variables are containers store values in.

`let myVariable = 0;`

- end semicolon indicates statement ends;
- name restrictions
  - JavaScript is **case sensitive**
  - no digit begin.

assign variable:
- var nm = `prompt('what your name?')`;
- document.write(nm);
- var date = `Date()`;
- document.write(date);
- var location = `window.location`;
- document.write(location);

---

### different data types:

| Variable  | Explanation                                             | Example                                                                                     |
| --------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `String`  | A sequence of text, enclose in quote.                   | let myvar = 'Bob';                                                                          |
| `Number`  | A number. no quotes.                                    | let myvar = 10;                                                                             |
| `Boolean` | A True/False value. no quotes.                          | let myvar = true;                                                                           |
| `Array`   | A structure to store multiple values                    | let myvar = [1,'Bob','Steve',10];  myVariable[0], myVariable[1], etc.                       |
| `Object`  | Everything in JSt is object, can be stored in variable. | let myvar = document.querySelector('h1'); let dog = { name : 'Spot', breed : 'Dalmatian' }; |

difference between `var` and `let`

- hoisting
  - can declare a variable with var after you initialize it and it will still work.
  - Hoisting no longer works with let.


```js
// var, let, const
// var: not good, may have conflic, var with same name
// const: constant, cannot be dirrectlly resign value, cannot be changed
// always use const, unless you know you will resign it, makes it more robust and secure. less prone to errors.


var myName = 'Chris';
var myName = 'Bob';

let myName = 'Chris';
myName = 'Bob';

const age = 30;
age = 10   // error


// data type:
// strings, number, boolean, null, unerfined, symbol

const name = 'John';
const age = 30;     // all number
const rating = 4.5; // all number
const isCool = true;
const x = null;
const y = undefined;
let z;

console.log(typeof rating)
```

---

### Constants in JavaScript

constant — a value that once declared can't be changed.

reasons to do this
- from security (if a third party script changed such values it could cause problems)
- to debugging and code comprehension (it is harder to `accidentally change values` that shouldn't be changed and mess things up).

```js
const daysInWeek = 7;
const hoursInDay = 24;

daysInWeek = 8; /* error*/
```


---

### number

Converting to number data types

```js
let myNumber = '74';
myNumber + 3;           // result 743
/* myNumber is actually defined as a string */

Number(myNumber) + 3;  // result 77
```

---

### Strings as objects

```js
let strs = 'mozilla';
strs.length;

strs[0];               // 第一个字母
strs[strs.length-1];   // 最后一个字母
s.substring(0,5)


strs.indexOf('zilla');
2
strs.indexOf('vanilla');
-1    // not found in the main string.
if(strs.indexOf('mozilla') !== -1) {
  // do stuff with the string
}

strs.slice(0,3);
"moz"


strs.toLowerCase();
strs.toUpperCase();


strs.replace('moz','van');
"vanilla"      // 但是不改变原本的 browserType


// Concatenation
console.log('My name is ' + name + 'and I am' + age)
// Template String
const hello = `My name is ${name} and I am ${age}`;
console.log(hello)


const s = 'Hello, World';
s.split(', ')    // result is a array

```

---

### array

```js

const numbers = new Array(1,2,3,4,5)

const fruits = []

let shopping = ['bread', 'milk', 'cheese', 'hummus', 'noodles', 10, true]; // can have different type

shopping[0] = 'tahini';

/* an array inside an array  */
random[2][2];

shopping.length;

// check is item exist
shopping.isArray('meat')

// check index
shopping.indexof('bread')  // return number


let sequence = [1, 1, 2, 3, 5, 8, 13];
for (let i = 0; i < sequence.length; i++) {
  console.log(sequence[i]);
}

let myData = 'Manchester,London,Liverpool,Birmingham,Leeds,Carlisle';
let myArray = myData.split(',');
let myNewString = myArray.join(',');
myArray;
(6) ["Manchester", "London", "Liverpool", "Birmingham", "Leeds", "Carlisle"]
myNewString
"Manchester,London,Liverpool,Birmingham,Leeds,Carlisle"

myArray[myArray.length-1];
"Carlisle"     // the last item in the array

let dogNames = ['Rocket','Flash','Bella','Slugger'];
dogNames.toString(); // Rocket,Flash,Bella,Slugger


// add or remove at the end of your array
myArray.push('Cardiff');                   // return a number!!!
let newLength = myArray.push('Bristol');   // 8, a number

let removedItem = myArray.pop();           // return the popout item
removedItem
"Bristol"

// add orremove at the begin of your array
myArray.unshift('Edinburgh');   // return a number!!!
let removedItem = myArray.shift();

myArray.sort();

```


---

## Comments
Comments are, essentially, short snippets of text that can be added in-between code which is ignored by the browser.
You can put comments into JavaScript code, just as you can in CSS:

```js
/*
Everything in between is a comment.
*/

// This is a comment

```

---

## object literals

```js
const person = {
  firstname: 'John',
  lastname: 'Doe',
  age: 30
  hobbie: ['music', 'movies', 'sports']
  address: {
    street: '50 main st',
    city: 'Boston'
  }
}

person.hobbie[1]      // movies
person.address.city   // boston


// pull things out
const {firstname, lastname, address:{ciyt}} = person;
firstname    // John
city         // Boston

person.email = 'john@xx.com'


const todos = [
  {
    id: 1,
    text: 'Take out trash',
    isCompleted: true
  },
    {
    id: 2,
    text: 'Meeting with boss',
    isCompleted: true
  },
    {
    id: 3,
    text: 'Dentist appt',
    isCompleted: false
  },
]

todos[1].text    // Meeting with boss

// convert to JSON
const todoJSON = JSON.stringify(todos)

// loop through array
for(let i = 0; i < todos.length; i++) {
  console.log(todos[i].text)     // Meeting with boss
}

for(let todo of todos) {
  console.log(todos[i].text)     // Meeting with boss
}


// forEach, map, filter

todos.forEach(function(todo) {
  console.log(todo.text);
});
todos.forEach((todo) => console.log(todo))


const todoText = todos.map(function(todo) {     // map return a arry
  return todo.text;
});

const todoCompleted = todos.filter(function(todo) {
  return todo.isCompleted === true;             // retun a array, filter item
});

const todoCompleted = todos.filter(function(todo) {
  return todo.isCompleted === true;
}).map(function(todo){
  return todo.text;                              // retun a array with only text data
})

```

---

## Operators
An operator is a mathematical symbol
- produces a result based on two values (or variables).


| Operator                                      | Explanation	Symbol(s)                                                                  | Example                                                                     |
| --------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Addition    +                                 | add numbers or glue strings together                                                   | 6 + 9; 'Hello ' + 'world!';                                                 |
| Subtraction, Multiplication, Division -, *, / | basic math                                                                             | 9 - 3; 8 * 2; 9 / 3;                                                        |
| Assignment	=                                  | assigns value to variable                                                              | let myVariable = 'Bob';                                                     |
| Equality	===                                  | test see if two values are equal, returns Boolean result                               | myVariable === 4; [!(myVariable === 3); returns false because we negate it] |
| Not, Does-not-equal	!==                       | Returns the logically opposite value of what it precedes; it turns a true into a false | let myVariable = 3; myVariable !== 3;                                       |

```js
let name = 'Bingo';
name;
let hello = ' says hello!';
hello;
let greeting = name + hello;
greeting;

name += ' says hello!';
```

---

## link

<link rel='stylesheet' href='yrl'>;
<script src="url/main.js"></script>
<img src="pic.png">
backgroud: url('url');

---

## loop

### for loop

`for( assign iterator/variable/initializer; exit-condition; increment )`

```js
const cats = ['Bill', 'Jeff', 'Pete', 'Biggles', 'Jasmin'];
let info = 'My cats are called ';
const para = document.querySelector('p');

for (let i = 0; i < cats.length; i++) {
  if (i === cats.length - 1) {
    info += 'and ' + cats[i] + '.';
  } else {
    info += cats[i] + ', ';
  }
}
para.textContent = info;
// My cats are called Bill, Jeff, Pete, Biggles, Jasmin
```

#### Exiting loops with break

```html
<label for="search">Search by contact name: </label>
<input id="search" type="text">
<button>Search</button>
<p></p>
```

```js
const contacts = ['Chris:2232322', 'Sarah:3453456', 'Bill:7654322', 'Mary:9998769', 'Dianne:9384975'];
const para = document.querySelector('p');
const input = document.querySelector('input');
const btn = document.querySelector('button');

btn.addEventListener('click', function() {
  let searchName = input.value.toLowerCase();
  input.value = '';
  input.focus();

  for (let i = 0; i < contacts.length; i++) {
    let splitContact = contacts[i].split(':');

    if (splitContact[0].toLowerCase() === searchName) {
      para.textContent = splitContact[0] + '\'s number is ' + splitContact[1] + '.';
      break;
    } else {
      para.textContent = 'Contact not found.';
    }
  }
});

```

#### Skipping iterations with continue

`continue` statement: skips to the next iteration of the loop

```js
let num = input.value;

for (let i = 1; i <= num; i++) {
  let sqRoot = Math.sqrt(i);
  if (Math.floor(sqRoot) !== sqRoot) {
    continue;
  }

  para.textContent += i + ' ';
}
```

### while

```js

initializer
while (exit-condition) {
  // code to run
  final-expression
}


1.
let i = 0
while (i <= 10) {
  console.log(`For Loop Number: ${i}`)
  i++
}


2.
const cats = ['Bill', 'Jeff', 'Pete', 'Biggles', 'Jasmin'];
let info = 'My cats are called ';
const para = document.querySelector('p');

let i = 0;
while (i < cats.length) {
  if (i === cats.length - 1) {
    info += 'and ' + cats[i] + '.';
  } else {
    info += cats[i] + ', ';
  }
  i++;
}

```

### do...while

```js
const cats = ['Bill', 'Jeff', 'Pete', 'Biggles', 'Jasmin'];
let info = 'My cats are called ';
const para = document.querySelector('p');

let i = 0;
do {
  if (i === cats.length - 1) {
    info += 'and ' + cats[i] + '.';
  } else {
    info += cats[i] + ', ';
  }
  i++;
} while (i < cats.length);
```


---

## Conditionals
Conditionals
- code structures to test if an expression returns true or not,
- running alternative code revealed by its result.


```js
if (condition) {
  code to run if condition is true
} else {
  run some other code instead
}

if (condition) {
  code to run if condition is true
}
run some other code    //  it always runs, regardless of whether the condition returns true or false.

// Any value that is not false, undefined, null, 0, NaN, or an empty string ('') actually returns true


const x = 10;
const y = '10';

if(x===10){}   // true, match both value and data type
if(y===10){}   // false
if(y==10){}    // still true, === just value.

if(x===10){
  console.log('x is 10')
} else if (x>10) {
  console.log('x > 10')
} else {
  console.log('x < 10')
}

const x = 4;
const y = 10;

if(x > 5 || y > 10) {}  // just one true
if(x > 5 && y > 10) {}  // both have to be true


// else if
<label for="weather">Select the weather type today: </label>
<select id="weather">
  <option value="">--Make a choice--</option>
  <option value="sunny">Sunny</option>
  <option value="rainy">Rainy</option>
</select>
<p></p>


const select = document.querySelector('select');
const para = document.querySelector('p');

select.addEventListener('change', setWeather);

function setWeather() {
  const choice = select.value;
  if (choice === 'sunny') {
    para.textContent = 'It is nice and sunny outside today. Wear shorts! Go to the beach, or the park, and get an ice cream.';
  } else if (choice === 'rainy') {
    para.textContent = 'Rain is falling outside; take a rain coat and an umbrella, and don\'t stay out for too long.';
  } else {
    para.textContent = '';
  }
}


// switch
switch(color) {
  case 'red':
    console.log('color is red')
    break;
  case 'blue':
    console.log('color is blue')
    break;
  default:
    console.log('color is not red or blue')
}


// ternary operater
const x = (condition) ? true run code : false run code

const x = 10
const (color = x > 10) ? 'red' : 'blue'   // blue


<label for="theme">Select theme: </label>
<select id="theme">
  <option value="white">White</option>
  <option value="black">Black</option>
</select>
<h1>This is my website</h1>


const select = document.querySelector('select');
const html = document.querySelector('html');
document.body.style.padding = '10px';

function update(bgColor, textColor) {
  html.style.backgroundColor = bgColor;
  html.style.color = textColor;
}

select.onchange = function() {
  ( select.value === 'black' ) ? update('black','white') : update('white','black');
}

```

---

## Functions
Functions are a way of **packaging functionality wish to reuse**.
- call a function, with the function name, instead of rewriting the entire code.

- let myVariable = `document.querySelector`('h1');
- `alert`('hello!');


### Anonymous functions

```js
const myButton = document.querySelector('button');
myButton.onclick = function() {
  alert('hello');
}

const myGreeting = function() {
  alert('hello');
}
myGreeting();

let anotherGreeting = myGreeting;
myGreeting();
anotherGreeting();
```


```js
function multiply(num1,num2) {
  let result = num1 * num2;
  return result;
}

multiply(4, 7);
multiply(0.5, 3);


function addNums(num1,num2) {
  console.log(num1 + num2);
}
addNums()   // NAN not a number


function addNums(num1 = 1, num2 = 1) {
  return num1 + num2;                      // always return something
}
console.log(addNums(num1 + num2));
```

### Function scope and conflicts
`scope`
- create a function, the variables and other things defined inside the function are inside their own separate scope,
- meaning that they are locked away in their own separate compartments, unreachable from code outside the functions.
- The top level outside all your functions is called the `global scope`. Values defined in the global scope are accessible from everywhere in the code.

```html
<!-- Excerpt from my HTML -->
<!-- both javascript has function greeting(), only first.js will work-->
<script src="first.js"></script>
<script src="second.js"></script>
<script>
  greeting();
</script>
```


### arrow function

```js
const addNums = (num1 = 1, num2 = 1) => {
  return num1 + num2;
}

const addNums = (num1 = 1, num2 = 1) => num1 + num2;

console.log(addNums(num1 + num2));

const addNums = num1  => num1 + 5;

// lexical
```


---

## object-oriented programming

```js

// Built in constructors
const name = new String('Kevin');
console.log(typeof name); // Shows 'Object'
const num = new Number(5);
console.log(typeof num); // Shows 'Object'


// consturctor function with prototype
function Person(firstname, lastname, dob) {
  this.firstname = firstname;
  this.lastname = lastname;
  this.dob = new Date(dob);           // Fri Mar 06

  /* this will show the function in console.log(person1), better put in prototype
  this.getBirthYear = function() {
    return this.dob.getFullYear();
  }
  this.getFullName = function() {
    return `${this.firstname} ${this.lastname}`;
  }
  */
}


// prototype
Person.prototype.getBirthYear = function() {
    return this.dob.getFullYear();
}
Person.prototype.getFullName = function() {
    return `${this.firstname} ${this.lastname}`;
}
person1.getBirthYear()     // 1980
person1.getFullName()      // John Doe


// es6 classes
class Person {
  constructor(firstname, lastname, dob) {
    this.firstname = firstname;
    this.lastname = lastname;
    this.dob = new Date(dob);
  }
  getBirthYear = function() {
    return this.dob.getFullYear();
  }
  getFullName = function() {
    return `${this.firstname} ${this.lastname}`;
  }
}


// instantiate object (same output, just easier to use)
const person1 = new Person('John','Doe','4-3-1980')
person1.firstname          // John
person1.dob.getFullYear()  // 1980
person1.getBirthYear()     // 1980
person1.getFullName()      // John Doe

```


---

## Adding an image changer
add an additional image to our site using some more DOM API features, using some JavaScript to switch between the two when the image is clicked.

```js
let myImage = document.querySelector('img');

myImage.onclick = function() {
    let mySrc = myImage.getAttribute('src');
    if(mySrc === 'images/firefox-icon.png') {
      myImage.setAttribute ('src','images/firefox2.png');
    } else {
      myImage.setAttribute ('src','images/firefox-icon.png');
    }
}
```

---

## Adding a personalized welcome message

Next we will add another bit of code, changing the page's title to a personalized welcome message when the user first navigates to the site. This welcome message will persist, should the user leave the site and later return — we will save it using the Web Storage API. We will also include an option to change the user and, therefore, the welcome message anytime it is required.

1. In `index.html`:

 ```js
 <button>Change user</button>

 <script src="main.js"></script>
 ```

2. In `main.js`, place the following code at the bottom of the file
    - exactly as written — this takes references to the new button and the heading, storing them inside variables:

```js
let myButton = document.querySelector('button');
let myHeading = document.querySelector('h1');
```

3. add function for personalized greeting:

```js
function setUserName() {
  let myName = prompt('Please enter your name.');
  localStorage.setItem('name', myName);
  myHeading.textContent = 'Mozilla is cool, ' + myName;
}
```

- `prompt() function`: brings up dialog box, asks input, storing it in a variable after the user presses OK.
-  call on an API called `localStorage`, which allows us to store data in the browser and later retrieve it.
    - use localStorage `setItem()` function: to create and store a data item called 'name', setting its value to the myName variable
- Finally, we set the `textContent` of the heading to a string, plus the user's newly stored name.

4. Next, add `if ... else` block — we could call this the `initialization code`: structures the app when it first loads:

```js
if(!localStorage.getItem('name')) {
  setUserName();
} else {
  let storedName = localStorage.getItem('name');
  myHeading.textContent = 'Mozilla is cool, ' + storedName;
}
```

5. Finally, put the below `onclick event handler` on the button. When clicked, the setUserName() function is run. This allows the user to set a new name, when they wish, by pressing the button:

```js
myButton.onclick = function() {
  setUserName();
}
```

Now when you first visit the site, it will ask you for your username, then give you a personalized message. You can change the name any time you like by pressing the button. As an added bonus, because the name is stored inside localStorage, it persists after the site is closed down, keeping the personalized message there when you next open the site!


### A user name of null?
When you run the example and get the dialog box that prompts you to enter your user name, try pressing the Cancel button. You should end up with a title that reads "Mozilla is cool, null".
- when you cancel the prompt, the value is set as null, a special value in JavaScript that basically refers to the absence of a value.

If you wanted to avoid these problems,
- check that the user hasn't entered null or a blank name by updating your setUserName() function to this:

```js
function setUserName() {
  let myName = prompt('Please enter your name.');
  if(!myName || myName === null) {
    setUserName();
  } else {
    localStorage.setItem('name', myName);
    myHeading.innerHTML = 'Mozilla is cool, ' + myName;
  }
}
```

In human language — if myName has no value, or its value is null, run setUserName() again from the start. If it does have a value (if the above statements are not true), then store the value in localStorage and set it as the heading's text.

---

## DOM

DOM, user interface.
- document object model
- page is represented by the DOM
- tree of nodes/elements created by the browser
- javascript can be used to read/write/manipulate to the DOM
- Object Oriented Representation


**DOM object**
- **document** - the root of the tree:
  - document(.URL.hright.links.bgColor)
- **element** - node of the tree
  - returned by a member of the API
- **nodelist** - an array of the element
  - document.getElementByTagName('p')
  - document.getElementByClassName(calss)
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
- no mater wich javascript, the API always the same.


```js
// ELEMENT SELECTORS

console.log(window);  // very high, for alert, document ...

alert(1);
window.alert(1);

document();         // for DOM
window.document();

// Single Element Selectors
// chose by ID
const form = document.getElementById('my-form')
// select single elements, JQuery
console.log(document.querySelector('.container'));
// single elements, only select the frist one
document.querySelector('h1')


// Multiple Element Selectors, select more than one thing
// return a Node list, list array can use forEach()
console.log(document.querySelectorAll('.item'));
// by tag
console.log(document.getElementsByTagName('li'));
// return a HTMLCollection, not array
console.log(document.getElementsByClassName('item'));

// loop
const items = document.querySelectorAll('.item');
items.forEach((item) => console.log(item));


// MANIPULATING THE DOM
const ul = document.querySelector('.items');
ul.remove();                       // all ul will gone
ul.lastElementChild.remove();      // remove the last one
ul.firstElementChild.textContent = 'Hello';
ul.children[1].innerText = 'Brad';
ul.lastElementChild.innerHTML = '<h1>Hello</h1>';  // add html

const btn = document.querySelector('.btn');
btn.style.background = 'red';    // can add result caused by event, click, interaction


// add div in body
const myDiv = document.createElement('div');
document.body.appendChild(myDiv);
```

---

## Events

Real interactivity on a website needs **events handlers**.
- These are **code structures** which listen for things happening in the browser and run code in response.
- example:
- **click event**, fired by the browser when you click on something.

```js
// enter the following into your console:
document.querySelector('html').onclick = function() {
    alert('Ouch! Stop poking me!');
}

// equals to
function() {
    alert('Ouch! Stop poking me!');
}
let myHTML = document.querySelector('html');
myHTML.onclick = function() {};
```

### Event handler properties

```js
// EVENTS
const btn = document.querySelector('.btn');
```

```js
1.
btn.onclick = function() {code}
- user clicks on an HTML element

2.
function bgChange() {code}
btn.onclick =
bgChange;

3.
btn.onfocus and btn.onblur
// The color changes when the button is focused and unfocused; try pressing tab to focus on the button and press tab again to focus away from the button. These are often used to display information about how to fill in form fields when they are focused, or display an error message if a form field has just been filled in with an incorrect value.

4.
btn.ondblclick
// The color changes only when the button is double-clicked.

5.
window.onkeypress, window.onkeydown, window.onkeyup
// The color changes when a key is pressed on the keyboard. The keypress event refers to a general press (button down and then up), while keydown and keyup refer to just the key down and key up parts of the keystroke, respectively. Note that it doesn't work if you try to register this event handler on the button itself — we've had to register it on the window object, which represents the entire browser window.

btn.onmouseover and btn.onmouseout
// The color changes
// when the mouse pointer is moved so it begins hovering over the button, or when pointer stops hovering over the button and moves off of it, respectively.

.onsize
// browser window is resized
.onload
// borwser finishes loading the page


6.
addEventListener(event, code)
removeEventListener(event, code)
// removes a previously added listener.

myElement.onclick = functionA;
myElement.addEventListener('click', functionA);

```


### Mouse Event
- onclick.ondbclick.onmousedown.onmouseenter.onmouseleave.onmounsemove.onmouseout

```js
//                   action  event   action: mousehover, mouseout
btn.addEventListener('click', e => {
  console.log('click') // flash fast as it is a submit button.
  e.preventDefault();  // prevent default behavior, no submit

  console.log(e.target.className);  // actual info

  document.getElementById('my-form').style.background = '#ccc';
  document.querySelector('body').classList.add('bg-dark');  // add a class

  document.querySelector('.item').lastElementChild.innerHTML = '<h1>Changed</h1>';
  ul.lastElementChild.innerHTML = '<h1>Changed</h1>';
});
```

### Keyboard Event
- onkeydown.onkeypress.onkeyup

```js
const nameInput = document.querySelector('#name');
nameInput.addEventListener('input', e => {
  document.querySelector('.container').append(nameInput.value);
});
```

### Frame Events
- .onload.onresize.onscroll.onerror
```js
<body onload="message('LOAD')" onresize="message('RESIZE')">
  <h1>Events</h1>
  <p onclick="message('CLICK')">blablabla</p>
  <p id="output">blablabla</p>
</body>

function message(msg){
  const wrd = document.getElementById('output');
  wrd.innerHTML = msg + ' event';
}
```

### USER FORM SCRIPT
```js
// Put DOM elements into variables
const myForm = document.querySelector('#my-form');
const nameInput = document.querySelector('#name');
const emailInput = document.querySelector('#email');
const msg = document.querySelector('.msg');
const userList = document.querySelector('#users');  // add user, add a item to ul


// Listen for form submit
myForm.addEventListener('submit', onSubmit);

function onSubmit(e) {   //creat the function
  e.preventDefault();

  console.log(nameInput);        // it gives the actual element
  console.log(nameInput.value);  // it gives value

  if(nameInput.value === '' || emailInput.value === '') {
    // alert('Please enter all fields');  // alert will stop your code
    msg.classList.add('error');           // add the error class to the item
    msg.innerHTML = 'Please enter all fields';

    // Remove error after 3 seconds
    setTimeout(() => msg.remove(), 3000);
  } else {
    // Create new list item with user
    const li = document.createElement('li');

    // Add text node with input values
    li.appendChild(document.createTextNode(`${nameInput.value}: ${emailInput.value}`));

    // Add HTML
    // li.innerHTML = `<strong>${nameInput.value}</strong>e: ${emailInput.value}`;

    // Append to ul
    userList.appendChild(li);

    // Clear fields
    nameInput.value = '';
    emailInput.value = '';
  }
}
```


### Event objects

`function bgChange(e) {}`
- parameter specified with a name such as `event`, `evt`, or simply `e` is the `event object`

```js
function bgChange(e) {
  const rndCol = 'rgb(' + random(255) + ',' + random(255) + ',' + random(255) + ')';
  e.target.style.backgroundColor = rndCol;
  console.log(e);
}
btn.addEventListener('click', bgChange);

// e.target — is the button itself.
// The target property of the event object is always a reference to the element that the event has just occurred upon.
```


### Preventing default behavior

```js
// form to input the name and submit

form.onsubmit = function(e) {
  if (fname.value === '' || lname.value === '') {
    e.preventDefault();
    para.textContent = 'You need to fill in both names!';
}
```

### Event bubbling and capture

In the capturing phase:
- The browser checks to see if the element's outer-most ancestor (<html>) has an onclick event handler registered on it for the capturing phase, and runs it if so.
- Then it moves on to the next element inside <html> and does the same thing, then the next one, and so on until it reaches the element that was actually clicked on.

In the bubbling phase, the exact opposite occurs:
- The browser checks to see if the element that was actually clicked on has an onclick event handler registered on it for the bubbling phase, and runs it if so.
- Then it moves on to the next immediate ancestor element and does the same thing, then the next one, and so on until it reaches the <html> element.

The standard Event object function:`stopPropagation()`
- when invoked on a handler's event object, makes it so that first handler is run but the event doesn't bubble any further up the chain, so no more handlers will be run.
