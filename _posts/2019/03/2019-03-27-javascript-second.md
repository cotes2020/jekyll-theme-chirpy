---
title: "Javascript进阶-双向数据绑定"
date: 2019-03-27
permalink: /2019-03-27-javascript-second/
---
要想实现，就要先看看什么是“双向数据绑定”，它和“单向数据绑定”有什么区别？这样才能知道要实现什么效果嘛。


**双向绑定**：视图（View）的变化能实时让数据模型（Model）发生变化，而数据的变化也能实时更新到视图层。


**单向数据绑定**：只有从数据到视图这一方向的关系。


### ES5 的 Object.defineProperty


```html
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta http-equiv="X-UA-Compatible" content="ie=edge" />
        <title>Document</title>
        <script>
            const obj = {
                value: ""
            };
            function onKeyUp(event) {
                obj.value = event.target.value;
            }
            // 对 obj.value 进行拦截
            Object.defineProperty(obj, "value", {
                get: function() {
                    return value;
                },
                // input是不受控组件，自带value。这种自带value的html组件更新需要特殊操作
                // 1、将其上的数据同步到数据模型
                // 2、数据模型的更新是有钩子的，例如这里的 Object.defineProperty
                // 3、钩子里可以对html组件自身的value做改变
                // 4、钩子成功后，数据模型也更新了，那么再更新视图

                // 上面是vuejs双向绑定原理，下面是简易实现
                set: function(newValue) {
                    value = newValue; // 更新数据模型
                    document.querySelector("#value").innerHTML = newValue; // 更新视图层
                    document.querySelector("input").value = newValue; // 更新html组件的value
                }
            });
        </script>
    </head>
    <body>
        <p>值是：<span id="value"></span></p>

        <input type="text" onkeyup="onKeyUp(event)" />
    </body>
</html>
```


### ES6 的 Proxy


随着，vue3.0 放弃支持了 IE 浏览器。而且`Proxy`兼容性越来越好，能支持 13 种劫持操作。


因此，vue3.0 选择使用`Proxy`来实现双向数据绑定，而不再使用`Object.defineProperty`。


```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <title>Document</title>
  <script>
    const obj = {}
    const newObj = new Proxy(obj, {
      get: function(target, key, receiver) {
        return Reflect.get(target, key, receiver)
      },
      set: function(target, key, value, receiver) {
        if(key === 'value') {
          document.querySelector('#value').innerHTML = value
          document.querySelector('input').value = value
        }
        return Reflect.set(target, key, value, receiver)
      }
    })
    function onKeyUp(event) {
      newObj.value = event.target.value
    }
  </script>
</head>
<body>
  <p>
    值是：<span id="value"></span>
  </p>
  <input type="text" onkeyup="onKeyUp(event)">
</body>
</html>
```


