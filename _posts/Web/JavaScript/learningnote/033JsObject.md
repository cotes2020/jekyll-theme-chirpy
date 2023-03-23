
# JavaScript Object

[toc]

---

## summary

```js
// 1. this.x = x type lines
function myConstructor(x,y,z) {
    this.x = x;
}
let myInstance = new myConstructor()).

// 2. Those defined directly on the constructor themselves, that are available only on the constructor. These are commonly only available on built-in browser objects, and are recognized by being chained directly onto a constructor, not an instance. For example, Object.keys(). These are also known as static properties/methods.

// 3.
myConstructor.prototype.x().

// 4.
var teacher1 = new Teacher( name = 'Chris' );
let teacher1 = { name = 'Chris' }
```


## Object basics

```js
// creating an object often begins with defining and initializing a variable.

const person = {
  name: ['Bob', 'Smith'],
  age: 32,
  gender: 'male',
  interests: ['music', 'skiing'],
  bio: function() {
    alert(this.name[0] + ' ' + this.name[1] + ' is ' + this.age + ' years old. He likes ' + this.interests[0] + ' and ' + this.interests[1] + '.');
  },
  greeting: function() {
    alert('Hi! I\'m ' + this.name[0] + '.');
  }
};

//objects:
person
person.name
person.name[0]
person.age
person.interests[1]
person.bio()
person.greeting()
```


`object` is made up of multiple members, each of which has a name (e.g. name and age above), and a value

The value of an object member can be pretty much anything
- object's `properties`. data items: a string, a number, two arrays, and two functions.
- object's `methods`. functions that allow the object to do something with that data,


An object like this is referred to as an object literal — literally written out the object contents as to create it.
in contrast to objects instantiated from classes.


### accesse the object's properties and methods


**Sub-namespaces**

```js
const person = {
  name : {
    first: 'Bob'
    last: 'Smith'
  }
}
```

#### Dot notation
- access the object's properties and methods using dot notation.
  - namespace.properties
  - namespace.methods
  - `person.age`
  - `person.name.first`


#### Bracket notation
- objects are sometimes called associative arrays — they map strings to values in the same way that arrays map numbers to values.
  - `person['age']`


### set (update) the value of object members

1. update the value of object members
```js
person.age = 45;
person['name']['last'] = 'Cratchit';
```

2. create new members
```js
person['eyes'] = 'hazel';
person.farewell = function() { alert("Bye everybody!"); }
```

3. bracket notation: can be used to set not only member values dynamically, but member names too.

```js
let NewPropertyname = NewPropertyname.value;
let NewPropertyValue = NewPropertyValue.value;
person[NewPropertyname] = NewPropertyValue;

let NewPropertyname = 'height';
let NewPropertyValue = '1.75m';
person[NewPropertyname] = NewPropertyValue;
person.height
```

---

### this

The `this` keyword refers to the `current object` the code is being written inside
- Object-oriented JavaScript for beginners article
- ensures that the correct values are used when a member's context changes
- for example, two different person object instances may have different names, but with same greeting

```js
const person1 = {
  name: 'Chris',
  greeting: function() {
    alert('Hi! I\'m ' + this.name + '.');
  }
}

const person2 = {
  name: 'Deepti',
  greeting: function() {
    alert('Hi! I\'m ' + this.name + '.');
  }
}

person1.greeting()
person2.greeting()
```

---

### some build in object
String class: `String.split(',');`
- myString.split(',');

Document class: `document.createElement('div')`
- const myDiv = document.createElement('div');
- const myVideo = document.querySelector('video');


---

## Object-oriented programming — the basics

JavaScript uses `constructor functions` to define and initialize objects and their features.

1. simply example:

    ```js
    function createNewPerson(name) {
      const obj = {};
      obj.name = name;
      obj.greeting = function() {
        alert('Hi! I\'m ' + obj.name + '.');
      };
      return obj;
    }

    const salva = createNewPerson('Salva');
    salva.name;
    salva.greeting();
    ```

2. Constructor functions: shortcut

    ```js
    function Person(name) {
      this.name = name;
      this.greeting = function() {
        alert('Hi! I\'m ' + this.name + '.');
      };
    }

    let person1 = new Person('Bob');
    let person2 = new Person('Sarah');
    person1.name
    person1.greeting()
    person2.name
    person2.greeting()
    ```

class:
- `abstraction`: creating a simple model

object instances:
- `instantiation`: the object instance is instantiated from the class.

child classes:
- inherit the data and code features of their parent class


**child classes:**
```js
class Rabbit extends Animal {
  hide() {
    alert(`${this.name} hides!`);
  }
}

let rabbit = new Rabbit("White Rabbit");



class PrimaryStudent extends Student {
    constructor(name, grade) {
        super(name); // 记得用super调用父类的构造方法!
        this.grade = grade;
    }
    myGrade() {      // 定义了新的myGrade方法。
        alert('I am at grade ' + this.grade);
    }
}


```

### create object instance

---

#### declaring an object literal


```js
const objectName() = {
  member1Name: member1Value,
};

function Person(first, last, age, gender, interests) {}

// 普通函数，用关键字new来调用这个函数，并返回一个对象：
// 如果不写new，这就是一个普通函数，它返回undefined

let person1 = new Person(...);
```


#### `Object()` constructor `let person1 = new Object(){}`

1. stores an empty object in the person1 variable
```js
let person1 = new Object();
// add properties and methods to this object
person1.name = 'Chris';
person1['age'] = 38;
person1.greeting = function() {
  alert('Hi!' + this.name + '.');
};
```

2. pass object literal to the Object() constructor as a parameter.
```js
let person1 = new Object({
  name: 'Chris',
  age: 38,
  greeting= function() {
    alert('Hi!' + this.name + '.');
  }
});
```


#### `Object.create()` method `let person2 = Object.create(person1)`

**Constructors**: create `constructors`, then create `instances`.

or create `object instances` without first creating constructors,
- create a new object based on any existing object.

```js
let person1 = Person('bob');
let person2 = Object.create(person1);
person2.name;
person2.greeting();
```

---

## prototype-based language
to provide inheritance, objects can have a `prototype object`

`prototype chain`
An object's prototype object may also have a prototype object, which it inherits methods and properties from
and explains why different objects have properties and methods defined on other objects available to them.

the properties and methods are defined on the `prototype` property on the Objects' constructor functions, not the object instances themselves.


### prototype objects

the members defined on person1 constructor
- Person() — `name, age, gender, interests, bio, and greeting`.
- also see some other members — `toString, valueOf` (defined on Person() constructor prototype object, `Object`).


```js
function Person(first, last, age, gender, interests) {
  // property and method definitions
  this.name = {
    'first': first,
    'last' : last
  };
  this.age = age;
  this.gender = gender;
}
let person1 = new Person('Bob', 'Smith', 32, 'male', ['music', 'skiing']);

person1.valueOf()
// Object.valueOf() is inherited by person1 because its constructor is Person()
// and Person()'s prototype is Object()
```

the Object reference page
- Some are inherited, and some aren't
  - the inherited ones are the ones defined on the `prototype` property, the ones that begin with `Object.prototype`.
  - not the ones that begin with just `Object`.
  - The prototype property's value is an object, which is basically a bucket for storing properties and methods that we want to be inherited by objects further down the prototype chain.
- So `Object.prototype.toString()`, `Object.prototype.valueOf()`, etc., are available to any object types that inherit from `Object.prototype`, including new object instances created from the Person() constructor.
- `Object.is()`, `Object.keys()`, etc. not defined inside the prototype bucket are not inherited by object instances or object types that inherit from Object.prototype. They are methods/properties available just on the `Object()` constructor itself.



### The `.constructor` property
Every constructor function has a `prototype` property whose value is an `object containing a constructor property`.
- This constructor property points to the `original constructor function`.

```js
let person1 = new Person('Bob', 'Smith', 32, 'male', ['music', 'skiing']);
let person2 = Object.create(person1);
person1.constructor  // the original constructor function.
person2.constructor
let person3 = new person1.constructor('Karen', 'Stephenson', 26, 'female', ['playing drums', 'mountain climbing']);

// to return the name of the constructor it is an instance of
person1.constructor.name
```


### Modifying prototypes `Person.prototype.newfunc = function() {}`


```js
// Constructor with property definitions
function Test(a, b, c, d) {
  console.log(1)
}
Test.prototype.x = function() { ... };  // First method definition
Test.prototype.y = function() { ... };  // Second method definition


Person.prototype.farewell = function() {
  alert(this.name.first + ' has left the building. Bye for now!');
};
person1.farewell();

// doesn't work, because this reference the global scope in this case, not the function scope.
Person.prototype.fullName = this.name.first + ' ' + this.name.last;

```

### Inheritance in JavaScript

#### Prototypal inheritance

```js
Person.prototype.greeting = function() {
  alert('Hi! I\'m ' + this.name.first + '.');
};
```

Defining a Teacher() constructor function

```js
1.
function Teacher(first, last, age, gender, interests, subject) {
  this.name = {
    first,
    last
  };
  this.age = age;
  this.gender = gender;
  this.interests = interests;
  this.subject = subject;
}


2.
function Teacher(first, last, age, gender, interests, subject) {
  Person.call(this, first, last, age, gender, interests);
  this.subject = subject;
}
```

the `call()` function
- to call a function defined somewhere else, but in the current context.
- The first parameter specifies the value of `this` that you want to use when running the function,
- other parameters are those that should be passed to the function when it is invoked.


#### Inheriting from a constructor with no parameters
don't need to specify them as additional arguments in call()

```js
function Brick() {
  this.width = 10;
  this.height = 20;
}

function BlueGlassBrick() {
  Brick.call(this);
  this.opacity = 0.5;
  this.color = 'blue';
}
```

#### to Inherite all methods

set child.prototype to reference an object that inherits its properties from parent.prototype

```js
Teacher.prototype = Object.create(Person.prototype);
Teacher.prototype.constructor
// ƒ Person(first, last, age, gender, interests) {
//     this.name = {first, last};
//     this.age = age;
//     this.gender = gender;
//     this.interests = interests;
// }

Object.defineProperty(Teacher.prototype, 'constructor', {
    value: Teacher,
    enumerable: false, // so that it does not appear in 'for in' loop
    writable: true });

Teacher.prototype.constructor
// ƒ Teacher(first, last, age, gender, interests, subject) {
//   Person.call(this, first, last, age, gender, interests);
//   this.subject = subject;
// }

```


#### overlap the old methods

```js
Teacher.prototype.greeting = function() {
  let prefix;

  if (this.gender === 'male' || this.gender === 'Male' || this.gender === 'm' || this.gender === 'M') {
    prefix = 'Mr.';
  } else if (this.gender === 'female' || this.gender === 'Female' || this.gender === 'f' || this.gender === 'F') {
    prefix = 'Ms.';
  } else {
    prefix = 'Mx.';
  }

  alert('Hello. My name is ' + prefix + ' ' + this.name.last + ', and I teach ' + this.subject + '.');
};

let teacher1 = new Teacher('Dave', 'Griffiths', 31, 'male', ['football', 'cookery'], 'mathematics');
```


### 创建原型继承

1. 编写一个函数来创建
```js
// 不要直接用obj.__proto__去改变一个对象的原型，
// Object.create()方法可以传入一个原型对象，并创建一个基于该原型的新对象，但是新对象什么属性都没有，


// 原型对象:
var Student = {
    name: 'Robot',
    height: 1.2,
    run: function () {
        console.log(this.name + ' is running...');
    }
};

function createStudent(name) {
    // 基于Student原型创建一个新对象:
    var s = Object.create(Student);
    // 初始化新对象:
    s.name = name;
    return s;
}

var xiaoming = createStudent('小明');
xiaoming.run(); // 小明 is running...
xiaoming.__proto__ === Student; // true
```

---

## class syntax


```js
class Person {
  constructor(first, last, age, gender, interests) {
    this.name = {first, last};
    this.age = age;
    this.gender = gender;
    this.interests = interests;
  }
  greeting() {
    console.log(`Hi! I'm ${this.name.first}`);
  };
  farewell() {
    console.log(`${this.name.first} has left the building. Bye for now!`);
  };
}
```

The `constructor()` method defines the constructor function that represents our Person class.

instantiate object instances: using the `new` operator

```js
let han = new Person('Han', 'Solo', 25, 'male', ['Smuggling']);
han.greeting();
// Hi! I'm Han

let leia = new Person('Leia', 'Organa', 19, 'female', ['Government']);
leia.farewell();
// Leia has left the building. Bye for now
```

### Inheritance with class syntax

```js
class Teacher extends Person {
  constructor(first, last, age, gender, interests, subject, grade) {
    super(first, last, age, gender, interests);

    // subject and grade are specific to Teacher
    this.subject = subject;
    this.grade = grade;
  }
  get subject() {
    return this._subject;
  }
  set subject(newSubject) {
    this._subject = newSubject;
  }
//   To show the current value of the _subject property, snape.subject, getter method.
//   To assign a new value to the _subject property, snape.subject="new value", setter method.
}

let snape = new Teacher('Severus', 'Snape', 58, 'male', ['Potions'], 'Dark arts', 5);
snape.greeting(); // Hi! I'm Severus.
snape.farewell(); // Severus has left the building. Bye for now.
snape.age // 58
snape.subject; // Dark arts

// Check the default value
console.log(snape.subject) // Returns "Dark arts"

// Change the value
snape.subject = "Balloon animals" // Sets _subject to "Balloon animals"

// Check it again and see if it matches the new value
console.log(snape.subject) // Returns "Balloon animals"
```




.
