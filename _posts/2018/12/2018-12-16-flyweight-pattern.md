---
title: "JavaScript「结构型」设计模式"
date: 2018-12-16
permalink: /2018-12-16-flyweight-pattern/
---
> 结构型模式：解决怎样组装现有对象，设计交互方式，从而达到实现一定的功能目的。例如，以封装为目的的适配器和桥接，以扩展性为目的的代理、装饰器


## 享元模式


享元模式：运用共享技术来减少创建对象的数量，从而减少内存占用、提高性能。


### 什么是“享元模式”？


享元模式：运用共享技术来减少创建对象的数量，从而减少内存占用、提高性能。

1. 享元模式提醒我们将一个**对象的属性划分为内部和外部状态**。
	- 内部状态：可以被对象集合共享，通常不会改变
	- 外部状态：根据应用场景经常改变
2. 享元模式是**利用时间换取空间**的优化模式。

### 应用场景


享元模式虽然名字听起来比较高深，但是实际使用非常容易：**只要是需要大量创建重复的类的代码块，均可以使用享元模式抽离内部/外部状态**，减少重复类的创建。


为了显示它的强大，下面的代码是简单地实现了大家耳熟能详的“对象池”，以彰显这种设计模式的魅力。


### 代码实现


这里利用`python`和`javascript`实现了一个“**通用对象池**”类--`ObjectPool`。这个类管理一个装载空闲对象的数组，**如果外部需要一个对象，直接从对象池中获取，而不是通过** **`new`** **操作**。


对象池可以大量减少重复创建相同的对象，从而节省了系统内存，提高运行效率。


为了形象说明“享元模式”在“对象池”实现和应用，特别准备了模拟了`File`类，并且模拟了“文件下载”操作。


通过阅读下方代码可以发现：**对于****`File`****类，内部状态是****`pool`****性和****`download`****方法；外部状态是****`name`****和****`src`****(文件名和文件链接)**。借助对象池，实现了`File`类的复用。


_注：为了方便演示，__`Javascript`__**实现的是并发操作，**__`Python`__实现的是串行操作。输出结果略有不同。_


### Python3 实现


```python
from time import sleep
class ObjectPool:  # 通用对象池
    def __init__(self):
        self.__pool = []
    # 创建对象
    def create(self, Obj):
        # 对象池中没有空闲对象，则创建一个新的对象
        # 对象池中有空闲对象，直接取出，无需再次创建
        return self.__pool.pop() if len(self.__pool) > 0 else Obj(self)
    # 对象回收
    def recover(self, obj):
        return self.__pool.append(obj)
    # 对象池大小
    def size(self):
        return len(self.__pool)
class File:  # 模拟文件对象
    def __init__(self, pool):
        self.__pool = pool
    def download(self):  # 模拟下载操作
        print('+ 从', self.src, '开始下载', self.name)
        sleep(0.1)
        print('-', self.name, '下载完成')
        # 下载完毕后，将对象重新放入对象池
        self.__pool.recover(self)
if __name__ == '__main__':
    obj_pool = ObjectPool()
    file1 = obj_pool.create(File)
    file1.name = '文件1'
    file1.src = '<https://download1.com>'
    file1.download()
    file2 = obj_pool.create(File)
    file2.name = '文件2'
    file2.src = '<https://download2.com>'
    file2.download()
    file3 = obj_pool.create(File)
    file3.name = '文件3'
    file3.src = '<https://download3.com>'
    file3.download()
    print('*' * 20)
    print('下载了3个文件, 但其实只创建了', obj_pool.size(), '个对象')

```


输出结果(这里为了方便演示直接使用了`sleep`方法，没有再用多线程模拟）：


```shell
+ 从 <https://download1.com> 开始下载 文件1
- 文件1 下载完成
+ 从 <https://download2.com> 开始下载 文件2
- 文件2 下载完成
+ 从 <https://download3.com> 开始下载 文件3
- 文件3 下载完成
********************
下载了3个文件, 但其实只创建了 1 个对象
```


### ES6 实现


```javascript
// 对象池
class ObjectPool {
    constructor() {
        this._pool = []; //
    }
    // 创建对象
    create(Obj) {
        return this._pool.length === 0
            ? new Obj(this) // 对象池中没有空闲对象，则创建一个新的对象
            : this._pool.shift(); // 对象池中有空闲对象，直接取出，无需再次创建
    }
    // 对象回收
    recover(obj) {
        return this._pool.push(obj);
    }
    // 对象池大小
    size() {
        return this._pool.length;
    }
}
// 模拟文件对象
class File {
    constructor(pool) {
        this.pool = pool;
    }
    // 模拟下载操作
    download() {
        console.log(`+ 从 ${this.src} 开始下载 ${this.name}`);
        setTimeout(() => {
            console.log(`- ${this.name} 下载完毕`); // 下载完毕后, 将对象重新放入对象池
            this.pool.recover(this);
        }, 100);
    }
}
/****************** 以下是测试函数 **********************/
let objPool = new ObjectPool();
let file1 = objPool.create(File);
file1.name = "文件1";
file1.src = "<https://download1.com>";
file1.download();
let file2 = objPool.create(File);
file2.name = "文件2";
file2.src = "<https://download2.com>";
file2.download();
setTimeout(() => {
    let file3 = objPool.create(File);
    file3.name = "文件3";
    file3.src = "<https://download3.com>";
    file3.download();
}, 200);
setTimeout(
    () =>
        console.log(
            `${"*".repeat(
                50
            )}\\n下载了3个文件，但其实只创建了${objPool.size()}个对象`
        ),
    1000
);
```


输出结果如下：


```shell
+ 从 <https://download1.com> 开始下载 文件1
+ 从 <https://download2.com> 开始下载 文件2
- 文件1 下载完毕
- 文件2 下载完毕
+ 从 <https://download3.com> 开始下载 文件3
- 文件3 下载完毕
**************************************************
下载了3个文件，但其实只创建了2个对象
```


## 代理模式


代理模式的定义：为一个对象提供一种代理以方便对它的访问。


### 什么是代理模式？


代理模式的定义：为一个对象提供一种代理以方便对它的访问。


**代理模式可以解决避免对一些对象的直接访问**，以此为基础，常见的有保护代理和虚拟代理。保护代理可以在代理中直接拒绝对对象的访问；虚拟代理可以延迟访问到真正需要的时候，以节省程序开销。


### 代理模式优缺点


代理模式有高度解耦、对象保护、易修改等优点。


同样地，因为是通过“代理”访问对象，因此开销会更大，时间也会更慢。


### 代码实现


### python3 实现


```python
class Image:
  def __init__(self, filename):
    self.filename = filename
  def load_img(self):
    print("finish load " + self.filename)
  def display(self):
    print("display " + self.filename)
# 借助继承来实现代理模式
class ImageProxy(Image):
  def __init__(self, filename):
    super().__init__(filename)
    self.loaded = False
  def load_img(self):
    if self.loaded == False:
      super().load_img()
    self.loaded = True
  def display(self):
    return super().display()
if __name__ == "__main__":
  proxyImg = ImageProxy("./js/image.png")
  # 只加载一次，其它均被代理拦截
  # 达到节省资源的目的
  for i in range(0,10):
    proxyImg.load_img()
  proxyImg.display()
```


### javascript 实现


**`main.js`**


```javascript
// main.js
const myImg = {
    setSrc(imgNode, src) {
        imgNode.src = src;
    }
};
// 利用代理模式实现图片懒加载
const proxyImg = {
    setSrc(imgNode, src) {
        myImg.setSrc(imgNode, "./image.png"); // NO1. 加载占位图片并且将图片放入<img>元素
        let img = new Image();
        img.onload = () => {
            myImg.setSrc(imgNode, src); // NO3. 完成加载后, 更新 <img> 元素中的图片
        };
        img.src = src; // NO2. 加载真正需要的图片
    }
};
let imgNode = document.createElement("img"),
    imgSrc =
        "<https://upload-images.jianshu.io/upload_images/5486602-5cab95ba00b272bd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp>";
document.body.appendChild(imgNode);
proxyImg.setSrc(imgNode, imgSrc);

```


**`main.html`**


```html
<!-- main.html -->
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta http-equiv="X-UA-Compatible" content="ie=edge" />
        <title>每天一个设计模式 · 代理模式</title>
    </head>
    <body>
        <script src="./main.js"></script>
    </body>
</html>
```


## 桥接模式


桥接模式：将抽象部分和具体实现部分分离，两者可独立变化，也可以一起工作。


### 什么是桥接模式


桥接模式：将抽象部分和具体实现部分分离，两者可独立变化，也可以一起工作。


在这种模式的实现上，需要一个对象担任“桥”的角色，起到连接的作用。


### 应用场景


在封装开源库的组件时候，经常会用到这种设计模式。


例如，对外提供暴露一个`afterFinish`函数,


如果用户有传入此函数, 那么就会在某一段代码逻辑中调用。


这个过程中，组件起到了“桥”的作用，而具体实现是用户自定义。


### 多语言实现


### ES6 实现


JavaScript 中桥接模式的典型应用是：`Array`对象上的`forEach`函数。


此函数负责循环遍历数组每个元素，是抽象部分；


而回调函数`callback`就是具体实现部分。


下方是模拟`forEach`方法：


```javascript
const forEach = (arr, callback) => {
    if (!Array.isArray(arr)) return;
    const length = arr.length;
    for (let i = 0; i < length; ++i) {
        callback(arr[i], i);
    }
};
// 以下是测试代码
let arr = ["a", "b"];
forEach(arr, (el, index) => console.log("元素是", el, "位于", index));
```


### python3 实现


和 Js 一样，这里也是模拟一个`for_each`函数：


它会循环遍历所有的元素，并且对每个元素执行指定的函数。


```python
from inspect import isfunction
# for_each 起到了“桥”的作用
def for_each(arr, callback):
  if isinstance(arr, list) == False or isfunction(callback) == False:
    return
  for (index, item) in enumerate(arr):
    callback(item, index)
# 具体实现部分
def callback(item, index):
  print('元素是', item, '; 它的位置是', index)
# 以下是测试代码
if __name__ == '__main__':
  arr = ['a', 'b']
  for_each(arr, callback)
```


## 装饰者模式


装饰者模式：在**不改变**对象自身的基础上，**动态**地添加功能代码。


### 什么是“装饰者模式”？


装饰者模式：在**不改变**对象自身的基础上，**动态**地添加功能代码。


根据描述，装饰者显然比继承等方式更灵活，而且**不污染**原来的代码，代码逻辑松耦合。


### 应用场景


装饰者模式由于松耦合，多用于一开始不确定对象的功能、或者对象功能经常变动的时候。


尤其是在**参数检查**、**参数拦截**等场景。


### 代码实现


### ES6 实现


ES6 的装饰器语法规范只是在“提案阶段”，而且**不能**装饰普通函数或者箭头函数。


下面的代码，`addDecorator`可以为指定函数增加装饰器。


其中，装饰器的触发可以在函数运行之前，也可以在函数运行之后。


**注意**：装饰器需要保存函数的运行结果，并且返回。


```typescript
const isFn = fn => typeof fn === "function";
const addDecorator = (fn, before, after) => {
    if (!isFn(fn)) {
        return () => {};
    }
    return (...args) => {
        let result;
        // 按照顺序执行“装饰函数”
        isFn(before) && before(...args);
        // 保存返回函数结果
        isFn(fn) && (result = fn(...args));
        isFn(after) && after(...args);
        // 最后返回结果
        return result;
    };
};
/******************以下是测试代码******************/
const beforeHello = (...args) => {
    console.log(`Before Hello, args are ${args}`);
};
const hello = (name = "user") => {
    console.log(`Hello, ${name}`);
    return name;
};
const afterHello = (...args) => {
    console.log(`After Hello, args are ${args}`);
};
const wrappedHello = addDecorator(hello, beforeHello, afterHello);
let result = wrappedHello("godbmw.com");
console.log(result);
```


### Python3 实现


python 直接提供装饰器的语法支持。用法如下：


```python
# 不带参数
def log_without_args(func):
    def inner(*args, **kw):
        print("args are %s, %s" % (args, kw))
        return func(*args, **kw)
    return inner
# 带参数
def log_with_args(text):
    def decorator(func):
        def wrapper(*args, **kw):
            print("decorator's arg is %s" % text)
            print("args are %s, %s" % (args, kw))
            return func(*args, **kw)
        return wrapper
    return decorator
@log_without_args
def now1():
    print('call function now without args')
@log_with_args('execute')
def now2():
    print('call function now2 with args')
if __name__ == '__main__':
    now1()
    now2()
```


其实 python 中的装饰器的实现，也是通过“闭包”实现的。


以上述代码中的`now1`函数为例，装饰器与下列语法等价：


```python
# ....
def now1():
    print('call function now without args')
# ...
now_without_args = log_without_args(now1) # 返回被装饰后的 now1 函数
now_without_args() # 输出与前面代码相同
```


## 组合模式


组合模式，将对象组合成树形结构以表示“部分-整体”的层次结构。


### 什么是“组合模式”？


组合模式，将对象组合成树形结构以表示“部分-整体”的层次结构。

1. 用小的子对象构造更大的父对象，而这些子对象也由更小的子对象构成
2. **单个对象和组合对象对于用户暴露的接口具有一致性**，而同种接口不同表现形式亦体现了多态性

### 应用场景


组合模式可以在需要针对“树形结构”进行操作的应用中使用，例如扫描文件夹、渲染网站导航结构等等。


### 代码实现


这里用代码**模拟文件扫描功能**，封装了`File`和`Folder`两个类。在组合模式下，用户可以向`Folder`类嵌套`File`或者`Folder`来模拟真实的“文件目录”的树结构。


同时，两个类都对外提供了`scan`接口，`File`下的`scan`是扫描文件，`Folder`下的`scan`是调用子文件夹和子文件的`scan`方法。整个过程采用的是**深度优先**。


### python3 实现


```python
class File:  # 文件类
    def __init__(self, name):
        self.name = name
    def add(self):
        raise NotImplementedError()
    def scan(self):
        print('扫描文件：' + self.name)
class Folder:  # 文件夹类
    def __init__(self, name):
        self.name = name
        self.files = []
    def add(self, file):
        self.files.append(file)
    def scan(self):
        print('扫描文件夹: ' + self.name)
        for item in self.files:
            item.scan()
if __name__ == '__main__':
    home = Folder("用户根目录")
    folder1 = Folder("第一个文件夹")
    folder2 = Folder("第二个文件夹")
    file1 = File("1号文件")
    file2 = File("2号文件")
    file3 = File("3号文件")
    # 将文件添加到对应文件夹中
    folder1.add(file1)
    folder2.add(file2)
    folder2.add(file3)
    # 将文件夹添加到更高级的目录文件夹中
    home.add(folder1)
    home.add(folder2)
    # 扫描目录文件夹
    home.scan()
```


执行`$ python main.py`, 最终输出结果是：


```text
扫描文件夹: 用户根目录
扫描文件夹: 第一个文件夹
扫描文件：1号文件
扫描文件夹: 第二个文件夹
扫描文件：2号文件
扫描文件：3号文件
```


### ES6 实现


```javascript
// 文件类
class File {
    constructor(name) {
        this.name = name || "File";
    }
    add() {
        throw new Error("文件夹下面不能添加文件");
    }
    scan() {
        console.log("扫描文件: " + this.name);
    }
}
// 文件夹类
class Folder {
    constructor(name) {
        this.name = name || "Folder";
        this.files = [];
    }
    add(file) {
        this.files.push(file);
    }
    scan() {
        console.log("扫描文件夹: " + this.name);
        for (let file of this.files) {
            file.scan();
        }
    }
}
let home = new Folder("用户根目录");
let folder1 = new Folder("第一个文件夹"),
    folder2 = new Folder("第二个文件夹");
let file1 = new File("1号文件"),
    file2 = new File("2号文件"),
    file3 = new File("3号文件");
// 将文件添加到对应文件夹中
folder1.add(file1);
folder2.add(file2);
folder2.add(file3);
// 将文件夹添加到更高级的目录文件夹中
home.add(folder1);
home.add(folder2);
// 扫描目录文件夹
home.scan();

```


执行`$ node main.js`，最终输出结果是：


```text
扫描文件夹: 用户根目录
扫描文件夹: 第一个文件夹
扫描文件: 1号文件
扫描文件夹: 第二个文件夹
扫描文件: 2号文件
扫描文件: 3号文件
```


## 适配器模式


适配器模式：为多个不兼容接口之间提供“转化器”。


### 什么是适配器模式？


适配器模式：为多个不兼容接口之间提供“转化器”。


它的实现非常简单，检查接口的数据，进行过滤、重组等操作，使另一接口可以使用数据即可。


### 应用场景


当数据不符合使用规则，就可以借助此种模式进行格式转化。


### 多语言实现


假设编写了不同平台的音乐爬虫，破解音乐数据。而对外向用户暴露的数据应该是具有一致性。


下面，`adapter`函数的作用就是转化数据格式。


事实上，在我开发的**音乐爬虫库**--[music-api-next](https://github.com/dongyuanxin/music-api-next)就采用了下面的处理方法。


因为，网易、QQ、虾米等平台的音乐数据不同，需要处理成一致的数据返回给用户，方便用户调用。


### ES6 实现


```javascript
const API = {
    qq: () => ({
        n: "菊花台",
        a: "周杰伦",
        f: 1
    }),
    netease: () => ({
        name: "菊花台",
        author: "周杰伦",
        f: false
    })
};
const adapter = (info = {}) => ({
    name: info.name || info.n,
    author: info.author || info.a,
    free: !!info.f
});
/*************测试函数***************/
console.log(adapter(API.qq()));
console.log(adapter(API.netease()));
```


### python 实现


```python
def qq_music_info():
    return {
        'n': "菊花台",
        'a': "周杰伦",
        'f': 1
    }
def netease_music_info():
    return {
        'name': "菊花台",
        'author': "周杰伦",
        'f': False
    }
def adapter(info):
    result = {}
    result['name'] = info["name"] if 'name' in info else info['n']
    result['author'] = info['author'] if 'author' in info else info['a']
    result['free'] = not not info["f"]
    return result
if __name__ == '__main__':
    print(adapter(qq_music_info()))
    print(adapter(netease_music_info()))
```


## 参考

- 《JavaScript 设计模式和开发实践》
- [代理模式](https://www.runoob.com/design-pattern/proxy-pattern.html)
- [JavaScript Decorators: What They Are and When to Use Them](https://www.sitepoint.com/javascript-decorators-what-they-are/)
- [《阮一峰 ES6-Decorator》](http://es6.ruanyifeng.com/#docs/decorator)
- [《廖雪峰 python-Decorator》](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014318435599930270c0381a3b44db991cd6d858064ac0000)

