---
title: "ES5 如何实现继承方法"
url: "2019-03-27-javascript-second-es5-extend"
date: 2019-03-27
---

> 题目：ES5 中常用继承方法。


**方法一：绑定构造函数**


缺点：不能继承父类原型方法/属性


```javascript
function Animal() {
    this.species = "动物";
}
function Cat() {
    // 执行父类的构造方法, 上下文为实例对象
    Animal.apply(this, arguments);
}
/**
 * 测试代码
 */
var cat = new Cat();
console.log(cat.species); // output: 动物
```


**方法二：原型链继承**


缺点：无法向父类构造函数中传递参数；子类原型链上定义的方法有先后顺序问题。


**注意**：js 中交换原型链，均需要修复`prototype.constructor`指向问题。


```javascript
function Animal(species) {
    this.species = species;
}
Animal.prototype.func = function() {
    console.log("Animal");
};
function Cat() {}
/**
 * func方法是无效的, 因为后面原型链被重新指向了Animal实例
 */
Cat.prototype.func = function() {
    console.log("Cat");
};
Cat.prototype = new Animal();
Cat.prototype.constructor = Cat; // 修复: 将Cat.prototype.constructor重新指向本身
/**
 * 测试代码
 */
var cat = new Cat();
cat.func(); // output: Animal
console.log(cat.species); // undefined
```


**方法 3:组合继承**


结合绑定构造函数和原型链继承 2 种方式，缺点是：调用了 2 次父类的构造函数。


```javascript
function Animal(species) {
    this.species = species;
}
Animal.prototype.func = function() {
    console.log("Animal");
};
function Cat() {
    Animal.apply(this, arguments);
}
Cat.prototype = new Animal();
Cat.prototype.constructor = Cat;
/**
 * 测试代码
 */
var cat = new Cat("cat");
cat.func(); // output: Animal
console.log(cat.species); // output: cat
```


**方法 4:寄生组合继承**


改进了组合继承的缺点，只需要调用 1 次父类的构造函数。**它是引用类型最理想的继承范式**。（引自：《JavaScript 高级程序设计》）


```javascript
/**
 * 寄生组合继承的核心代码
 * @param {Function} sub 子类
 * @param {Function} parent 父类
 */
function inheritPrototype(sub, parent) {
    // 拿到父类的原型
    var prototype = Object.create(parent.prototype);
    // 改变constructor指向
    prototype.constructor = sub;
    // 父类原型赋给子类
    sub.prototype = prototype;
}
function Animal(species) {
    this.species = species;
}
Animal.prototype.func = function() {
    console.log("Animal");
};
function Cat() {
    Animal.apply(this, arguments); // 只调用了1次构造函数
}
inheritPrototype(Cat, Animal);
/**
 * 测试代码
 */
var cat = new Cat("cat");
cat.func(); // output: Animal
console.log(cat.species); // output: cat
```


