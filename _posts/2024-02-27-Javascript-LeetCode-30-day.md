---
title: For the love of Engineering 
author: arpiku 
date: 2024-02-27 15:45:00 +0530
categories: [Journal, Blog, Engineering, Opinion]
tags: [Journal, Blog]
pin: false 
---

# 30 Days of JavaScript

So there is this study plan on leetcode titled "30 Days of JavaScript", it mostly comprises of simple problems that
focus on the functional nature of JavaScript. Following are the solutions to the listed problems along with a brief 
discussion.

## Create Hello World Function 

{% raw %}
```js 
var createHelloWorld = function() {
    
    return function(...args) {
        
    }
};
```
{% endraw %}

- In the above code a function is being declared as a 'var', this is because in **functional programming paradigm, functions are first class objects**.
- That means they can be thrown around to other funtions, can be mapped on other funtions, can be passed as a parameter to other funtions.
- In this the solution is rather simple.

{% raw %}
```js 
var createHelloWorld = function() {
    
    return function(...args) { return "Hello World";        
    }
};
```
{% endraw %}


## Counter
Now this problem is interesting, the solution looks like this 

{% raw %}
```js
var createCounter = function(n) {
    return function() {
        return n++;
    };
};
```
{% endraw %}

- The interesting part here is how n lives even though no memory got allocated explicitly to preserve it, this is called a **closure**, a function 
returning a function, the ability for the inner function to maintain it's variable is a feature of JS closures.
- In Cpp similar behaviour would require us to somehow assign where the variable lives, as follows.

{% raw %}
```Cpp
#include<iostream>
#include<functional>

class Counter {
private:
  int _n;
public:
  Counter(int initValue) : _n(initValue) {} //Member init list
  
  std::functiona<int()> createCounter() {
  return [this]() -> {return n++};
  }
};
```
{% endraw %}

- See how an object exists that is getting passed to the inner function via "this", that's what closure behave like in JavaScript.


## To Be Or Not To Be 
- There is an attempt to be philosophical here, I am gonna choose to be and solve this question.
{% raw %}
```js
/**
 * @param {string} val
 * @return {Object}
 */
var expect = function(val) {
    // In JS we can treat functions as first class obejcts

    return {
        toBe: (cmp) => {
            if(val !== cmp) throw new Error("Not Equal");
            else return true;
        },

        notToBe: (val2) => {
            if(val === val2) throw new Error("Equal");
            else return true;
        }
    }
    
};

```
{% end %}

- The **function returns an object of functions!**, think about that.
- What is going on is that imagine the wrapping funtion to be object, one its on we have the "val", other values will be passed later on, what it is 
returning is a object which is "aware" of the "val", and needs the other values to funtion (which we are passing in the calls).

## Counter II 

- It's a combination of problems before, this time one particular value needs to stay constant and stored somewhere in order to be used later, so 
we just create that variable and do exactly that.
{% raw %}
```js
/**
 * @param {integer} init
 * @return { increment: Function, decrement: Function, reset: Function }
 */
var createCounter = function(init) {
    const initValue = init;

    return {
        increment: () => {return ++init;}, //The "++" is like C++
        decrement: () => {return --init;},
        reset: () => {
            init = initValue;
            return init}
    }
    
};

/**
 * const counter = createCounter(5)
 * counter.increment(); // 6
 * counter.reset(); // 5
 * counter.decrement(); // 4
 */
```
{% end %}


## Appy Transform Over Each Element in Array
- Following is just a simple implementation of arra.map 
{% raw %}
```js
/**
 * @param {number[]} arr
 * @param {Function} fn
 * @return {number[]}
 */
var map = function(arr, fn) {
  const result = [];
  for (let i = 0; i < arr.length; i++) {
    result.push(fn(arr[i], i));
  }
  return result;
};
```
{% end %}

## Filter Elements From Array
- In this problem we just simply evaluate and check if the element should be pushed into the return array, if the answer is yes, we push if not 
we continue.
- It's a simple question to show you how array.filter works.

{% raw %}
```js
/**
 * @param {number[]} arr
 * @param {Function} fn
 * @return {number[]}
 */
var filter = function(arr, fn) {
    const newArr = [];
    for(let i = 0; i<arr.length; i++) {
        if(fn(arr[i],i)) {
            newArr.push(arr[i]);
        }
    }
    return newArr;
 
};
```
{% endraw %}

- What perhaps is more interesting is how this is a functional approach (not the code), but the idea of using filters. In Haskell for example you can 
do 

{% raw %} 
```Haskell 
xs = [1,2,3,4,5,6,7,8,9,...] --This ... is actual syntax 
x_prime = [x <- xs | x > 5] -- This defination is simlar to how sets are defined in maths.
```
{% endraw %} 

- Here the condition is acting as "filter", generally you can have more elegant looking implementations when such filtering is required when using
funtional paradigm rather than loops and such in OOP.

## Array Reduce Transformation 

- This problem just builds a little but on the concepts we have already seen, here is a simple solution.

{% raw %}
```js
/**
 * @param {number[]} nums
 * @param {Function} fn
 * @param {number} init
 * @return {number}
 */
var reduce = function(nums, fn, init) {
    if(nums.length == 0)
        return init;
    let val = init;
    for(let i =0; i<nums.length; i++) {
        val = fn(val,nums[i]);
    }
    return val;
    
};
```
{% endraw %}

## Function Composition 

- Function compositin is nothing but the simple math fact that if you apply f(x) on g(x) on h(x) then the result is equal to F(x) = f(g(h(x)))
- This can be written in code as follows
{% raw %}
```js
/**
 * @param {Function[]} functions
 * @return {Function}
 */
var compose = function(funcs) {
    
    return function(x) {
        return funcs.reduceRight((acc, func) => func(acc), x);
    };
};

/**
 * const fn = compose([x => x + 1, x => 2 * x])
 * fn(4) // 9
 */
```
- Following is the implementation of similar concept in C++
{% raw %}
```cpp
#include <iostream>
#include <vector>
#include <functional>

// Function to compose a list of functions
std::function<int(int)> composeFunctions(const std::vector<std::function<int(int)>>& funcs) {
    return [funcs](int x) {
        int result = x;
        // Iterate through the functions in reverse order
        for (auto it = funcs.rbegin(); it != funcs.rend(); ++it) {
            result = (*it)(result);
        }
        return result;
    };
}

int main() {
    // Example functions
    auto f1 = [](int x) { return x + 2; };
    auto f2 = [](int x) { return x * 3; };
    auto f3 = [](int x) { return x - 5; };

    // Compose the functions
    auto composedFn = composeFunctions({f1, f2, f3});

    // Use the composed function
    std::cout << "Result: " << composedFn(5) << std::endl; // This would compute f1(f2(f3(5)))

    return 0;
}
```
{% endraw %}


## Return Length Of Argument 
- I mean, there isn't much to this one tbh.

{% raw %}
```js 
/**
 * @param {...(null|boolean|number|string|Array|Object)} args
 * @return {number}
 */
var argumentsLength = function(...args) {
    return args.length;  
};

/**
 * argumentsLength(1, 2, 3); // 3
 */
```
{% endraw %}

## Allow One Function Call
- Here is the solution 

{% raw %}
```js 
/**
 * @param {Function} fn
 * @return {Function}
 */
var once = function(fn) {
    let called = false;
    let result;

    return function(...args) {
        if (!called) {
            result = fn.apply(this, args);
            called = true;
            return result;
        }
        return undefined;
    }
};

/**
 * let fn = (a,b,c) => (a + b + c)
 * let onceFn = once(fn)
 *
 * onceFn(1,2,3); // 6
 * onceFn(2,3,6); // returns undefined without calling fn
 */
```
{% endraw %}
- You can omit the last line, as when nothing is specified JS by default returns undefined.


## Memoize
- It's just the heading, just create an object to store the result of function calls, if the answer has been calculated already return that,
or if not calculate and store it for later calls.

{% raw %}
```js
function memoize(fn) {
    const cache = {};

    return function(...args) {
        const key = JSON.stringify(args);
        if (key in cache) {
            return cache[key];
        } else {
            const result = fn.apply(this, args);
            cache[key] = result;
            return result;
        }
    };
}
```
{% endraw %}

## Add Two Promises
- In this we just wait till both promises are complete and then just return 

{% raw %}
```js
/**
 * @param {Promise} promise1
 * @param {Promise} promise2
 * @return {Promise}
 */
var addTwoPromises = async function(promise1, promise2) {
    const  [val1,val2] = await Promise.all([promise1,promise2]);
    return val1+val2;
};

/**
 * addTwoPromises(Promise.resolve(2), Promise.resolve(2))
 *   .then(console.log); // 4
 */
```
{% endraw %}


## Sleep
- I would like to tbh.
- But let's solve this problem first
- We just have to create a promise and wait for it, we pass the waiting time to setTimeout for that to happen after creating a new promise.

{% raw %}
```js
/**
 * @param {number} millis
 * @return {Promise}
 */
async function sleep(millis) {
    await new Promise(resolve => setTimeout(resolve,millis));
}

/** 
 * let t = Date.now()
 * sleep(100).then(() => console.log(Date.now() - t)) // 100
 */```
{% endraw %}

## TimeOut Cancellation

- I need to think a little about this one. update coming soon.
- The solution is below
{% raw %}
```js
const cancellable = function(fn, args, t) {
    const cancelFn = function (){
      clearTimeout(timer);
  };
  const timer = setTimeout(()=>{
      fn(...args)
  }, t);
  return cancelFn ;
};
```
{% endraw %}

## Interval Cancellation
- Same as above, but with different goal.

{% raw %}
```js

var cancellable = function(fn, args, t) {
    fn(...args);
    const timer = setInterval(() => fn(...args), t);

    const cancelFn = () => clearInterval(timer);
    return cancelFn;
};
```
{% endraw %}


## Promise Time Limit
- We just implement what we have seen already with a different syntax and goal.

{% raw %}
```js
var timeLimit = function(fn, t) {
  return async function(...args) {
    return new Promise((delayresolve, reject) => {
      const timeoutId = setTimeout(() => {
        clearTimeout(timeoutId);
        reject("Time Limit Exceeded");
      }, t);

      fn(...args)
        .then((result) => {
          clearTimeout(timeoutId);
          delayresolve(result);
        })
        .catch((error) => {
          clearTimeout(timeoutId);
          reject(error);
        });
    });
  };
};
```
{% endraw %}

## Cache with Time Limit


{% raw %}
- We just use the concepts learned and create and object to do our bidding.
```js 
var TimeLimitedCache = function() {
    this.memory= new Map();
};

/** 
 * @param {number} key
 * @param {number} value
 * @param {number} duration time until expiration in ms
 * @return {boolean} if un-expired key already existed
 */
TimeLimitedCache.prototype.set = function(key, value, duration) {
    let isthere=false;
    if(this.memory.has(key)){
        isthere=true;
        clearTimeout(this.memory.get(key)[1]);
    }
    const timeoutID=setTimeout(()=>{
        this.memory.delete(key);
    },duration)
    this.memory.set(key,[value,timeoutID]);
    return isthere;
};

/** 
 * @param {number} key
 * @return {number} value associated with key
 */
TimeLimitedCache.prototype.get = function(key) {
    if(this.memory.has(key)) return this.memory.get(key)[0];
    return -1;
};

/** 
 * @return {number} count of non-expired keys
 */
TimeLimitedCache.prototype.count = function() {
    return this.memory.size;
};

/**
 * const timeLimitedCache = new TimeLimitedCache()
 * timeLimitedCache.set(1, 42, 1000); // false
 * timeLimitedCache.get(1) // 42
 * timeLimitedCache.count() // 1
 */
```


## Debounce 
- Wait, didn't we already solve this too?

{% raw %}
```js
var debounce = function(fn, t = 1000) {
    let timer;
    return function(...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), t);
    }
};
```
{% endraw %}
- This list could have been better you know.

## Execute Asynchronous Functionsl Parallely
{% raw %}
```js
/**
 * @param {Array<Function>} functions
 * @return {Promise<any>}
 */
var promiseAll = function(functions) {
      return new Promise((resolve, reject) => {
    const results = [];
    let completed = 0;
    const totalFunctions = functions.length;

    if (totalFunctions === 0) {
      resolve(results); // Immediately resolve if the array is empty
      return;
    }

    functions.forEach((func, index) => {
      // Execute each function which returns a promise
      func()
        .then(result => {
          results[index] = result; // Store the result at the corresponding index
          completed += 1; // Increment the count of completed promises
          if (completed === totalFunctions) {
            resolve(results); // Resolve the main promise when all functions have completed
          }
        })
        .catch(error => {
          reject(error); // Reject the main promise if any promise fails
        });
    });
  });
};

/**
 * const promise = promiseAll([() => new Promise(res => res(42))])
 * promise.then(console.log); // [42]
 */```
{% endraw %}


## Is Object Empty
{% raw %}
```js
/**
 * @param {Object|Array} obj
 * @return {boolean}
 */
var isEmpty = function(obj) {
    return Object.keys(obj).length === 0;
};
```
{% endraw %}
- Yup, that was it!


## Chunk Array

{% raw %}
```js

/**
 * @param {Array} arr
 * @param {number} size
 * @return {Array}
 */
var chunk = function(arr, size) {
    const chunkedArr = [];
  for (let i = 0; i < arr.length; i += size) {
    chunkedArr.push(arr.slice(i, i + size));
  }
  return chunkedArr;
};
```
{% endraw %}
- Yeah, not much to this one either.


## Array prototype Last
- In this we just update the base class and add a new functionality to it.

{% raw %} 
```js
/**
 * @return {null|boolean|number|string|Array|Object}
 */
Array.prototype.last = function() {
    if (this.length === 0) {
      return -1;
    }
    return this[this.length - 1];
  };


/**
 * const arr = [1, 2, 3];
 * arr.last(); // 3
 */
```
{% endraw %}

## Group By
- Same concept as above but now for groupBy functionality.

{% raw %} 
```js

/**
 * @param {Function} fn
 * @return {Object}
 */
Array.prototype.groupBy = function(fn) {
        return this.reduce((acc, item) => {
      const key = fn(item);
      if (!acc[key]) {
        acc[key] = [];
      }
      acc[key].push(item);
      return acc;
    }, {}); 
  };


/**
 * [1,2,3].groupBy(String) // {"1":[1],"2":[2],"3":[3]}
 */
```
{% endraw %}


## Sort By 
- I remeber when I saw a backend engineer write code like this, I didn't know you could do things like this.


{% raw %}
```js
/**
 * @param {Array} arr
 * @param {Function} fn
 * @return {Array}
 */
var sortBy = function(arr, fn) {
    const clonedArr = [...arr];
    clonedArr.sort((a,b) => fn(a) - fn(b));
    return clonedArr;
};
```
{% endraw %}


## Join Two Arrays By ID
- Just join them  and sort them, and return!

{% raw %}
```js
/**
 * @param {Array} arr1
 * @param {Array} arr2
 * @return {Array}
 */
var join = function(arr1, arr2) {
  const merged = [...arr1, ...arr2];

  const mergedMap = new Map();

  merged.forEach(obj => {
    if (mergedMap.has(obj.id)) {
      mergedMap.set(obj.id, { ...mergedMap.get(obj.id), ...obj });
    } else {
      mergedMap.set(obj.id, obj);
    }
  });

  const joinedArray = Array.from(mergedMap.values());

  joinedArray.sort((a, b) => a.id - b.id);

  return joinedArray;
};
```
{% endraw %}


## Flatten Depply Nested Array
- Following is a recursive solution, it exceeds time limit for longer inputs.

{% raw %}
```js
/**
 * @param {Array} arr
 * @param {number} depth
 * @return {Array}
 */
var flat = function (arr, n,cd=0) {
    let result = [];

  arr.forEach(item => {
    if (Array.isArray(item) && cd < n) {
      result = result.concat(flat(item, n, cd + 1));
    } else {
      result.push(item);
    }

  });    return result;

};
```
{% endraw %}

- However with memoization it works fine

{% raw %}
```js
/**
 * @param {any[]} arr
 * @param {number} depth
 * @return {any[]}
 */
var flat = function (arr, n) {
    if(n==0) return arr;
    let result = [];
    const traverse = (a, n) => {
        for(let i in a) {
            if(n>0 && Array.isArray(a[i]))
                traverse(a[i], n-1)
            else
                result.push(a[i])
        }
    }
    traverse(arr, n);
    return result;
};
```
{% endraw %}


## Compact Object

{% raw %}
```js
var compactObject = function(obj) {
    if (typeof obj === 'object') {
        if (Array.isArray(obj)) {
            const tempArray = [];
            for (let index = 0; index < obj.length; index++) {
                if (Boolean(obj[index])) {
                    tempArray.push(compactObject(obj[index]));
                }
            }
            return tempArray;
        } else {
            const tempObject = {};
            for (const key in obj) {
                if (Boolean(obj[key])) {
                    tempObject[key] = compactObject(obj[key]);
                }
            }
            return tempObject;
        }
    }
    return obj;
};
```
{% endraw %}

## Event Emitter
- I sometimes forget that JS has classes too, cause I never think of it as an OOP langugage.

{% raw %}
```js
class EventEmitter {
    constructor () {
        this.subscriptions = new Map();
    }

    /**
     * @param {string} eventName
     * @param {Function} callback
     * @return {Object}
     */
    subscribe(eventName, callback) {
        const id = Symbol(callback);
        this.subscriptions.set(eventName, this.subscriptions.has(eventName)
            ? [...this.subscriptions.get(eventName), { id, callback }]
            : [{ id, callback }]
        );

        return {
            unsubscribe: () =>  this.subscriptions.set(eventName, this.subscriptions.get(eventName).filter(({ id: subId}) => subId !== id))
        };
    }
    
    /**
     * @param {string} eventName
     * @param {Array} args
     * @return {Array}
     */
    emit(eventName, args = []) {
        return (this.subscriptions.get(eventName) || []).map(({ callback }) => callback(...args));
    }
}

/**
 * const emitter = new EventEmitter();
 *
 * // Subscribe to the onClick event with onClickCallback
 * function onClickCallback() { return 99 }
 * const sub = emitter.subscribe('onClick', onClickCallback);
 *
 * emitter.emit('onClick'); // [99]
 * sub.unsubscribe(); // undefined
 * emitter.emit('onClick'); // []
 */
```
{% endraw %}

## Array Wrapper 
{% raw %}
```js
/**
 * @param {number[]} nums
 * @return {void}
 */
var ArrayWrapper = function(nums) {
    this.array = nums;
    
};

/**
 * @return {number}
 */
ArrayWrapper.prototype.valueOf = function() {
    return this.array.reduce((p,c) => p+c,0)
    
}

/**
 * @return {string}
 */
ArrayWrapper.prototype.toString = function() {
    return JSON.stringify(this.array);
}

/**
 * const obj1 = new ArrayWrapper([1,2]);
 * const obj2 = new ArrayWrapper([3,4]);
 * obj1 + obj2; // 10
 * String(obj1); // "[1,2]"
 * String(obj2); // "[3,4]"
 */
```
{% endraw %}

## Calculator With Method Chaining

{% raw %}
```js
class Calculator {

    /** 
     * @param {number} value
     */
    constructor(value) {
        this.result = value
    }

    /** 
     * @param {number} value
     * @return {Calculator}
     */
    add(value){
        this.result += value;
        return this
    }

    /** 
     * @param {number} value
     * @return {Calculator}
     */
    subtract(value){
        this.result -= value
        return this
    }

    /** 
     * @param {number} value
     * @return {Calculator}
     */  
    multiply(value) {
        this.result *= value
        return this
    }

    /** 
     * @param {number} value
     * @return {Calculator}
     */
    divide(value) {
        if(value === 0 ) throw new Error("Division by zero is not allowed") 
        this.result  = this.result / value
        return this
  }

    /** 
     * @param {number} value
     * @return {Calculator}
     */
    power(value) {
        this.result = Math.pow(this.result,value)
        return this
    }

    /** 
     * @return {number}
     */
    getResult() {
        return this.result
    }
}
```
{% endraw %}
