---
title: "JavaScript「行为型」设计模式"
date: 2018-10-25
permalink: /2018-10-25-stragegy-pattern/
categories: ["设计模式手册"]
---
> 行为型模式：描述多个类或对象怎样交互以及怎样分配职责


## 策略模式


策略模式定义：就是能够把一系列“可互换的”算法封装起来，并根据用户需求来选择其中一种。


### 什么是策略模式？


策略模式定义：就是能够把一系列“可互换的”算法封装起来，并根据用户需求来选择其中一种。


策略模式的**实现核心**就是：将算法的使用和算法的实现分离。算法的实现交给策略类。算法的使用交给环境类，环境类会根据不同的情况选择合适的算法。


### 策略模式优缺点


在使用策略模式的时候，需要了解所有的“策略”（strategy）之间的异同点，才能选择合适的“策略”进行调用。


### 代码实现


### python3 实现


```python
class Stragegy():
  # 子类必须实现 interface 方法
  def interface(self):
    raise NotImplementedError()
# 策略A
class StragegyA():
  def interface(self):
    print("This is stragegy A")
# 策略B
class StragegyB():
  def interface(self):
    print("This is stragegy B")
# 环境类：根据用户传来的不同的策略进行实例化，并调用相关算法
class Context():
  def __init__(self, stragegy):
    self.__stragegy = stragegy()
  # 更新策略
  def update_stragegy(self, stragegy):
    self.__stragegy = stragegy()
  # 调用算法
  def interface(self):
    return self.__stragegy.interface()
if __name__ == "__main__":
  # 使用策略A的算法
  cxt = Context( StragegyA )
  cxt.interface()
  # 使用策略B的算法
  cxt.update_stragegy( StragegyB )
  cxt.interface()

```


### javascript 实现


```text
// 策略类
const strategies = {
    A() {
        console.log("This is stragegy A");
    },
    B() {
        console.log("This is stragegy B");
    }
};
// 环境类
const context = name => {
    return strategies[name]();
};
// 调用策略A
context("A");
// 调用策略B
context("B");

```


## 迭代器模式


迭代器模式是指提供一种方法顺序访问一个集合对象的各个元素，使用者不需要了解集合对象的底层实现。


### 什么是迭代器模式？


迭代器模式是指提供一种方法顺序访问一个集合对象的各个元素，使用者不需要了解集合对象的底层实现。


### 内部迭代器和外部迭代器


内部迭代器：封装的方法完全接手迭代过程，外部只需要一次调用。


外部迭代器：用户必须显式地请求迭代下一元素。熟悉 C++的朋友，可以类比 C++内置对象的迭代器的 `end()`、`next()`等方法。


### python3 实现


python3 的迭代器可以用作`for()`循环和`next()`方法的对象。同时，在实现迭代器的时候，可以在借助生成器`yield`。python 会生成传给`yeild`的值。


```python
def my_iter():
  yield 0, "first"
  yield 1, "second"
  yield 2, "third"
if __name__ == "__main__":
  # 方法1: Iterator可以用for循环
  for (index, item) in my_iter():
    print("At", index , "is", item)
  # 方法2: Iterator可以用next()来计算
  # 需要借助 StopIteration 来终止循环
  _iter = iter(my_iter())
  while True:
    try:
      index,item = next(_iter)
      print("At", index , "is", item)
    except StopIteration:
      break

```


### ES6 实现


这里实现的是一个外部迭代器。需要实现边界判断函数、元素获取函数和更新索引函数。


```typescript
const Iterator = obj => {
    let current = 0;
    let next = () => (current += 1);
    let end = () => current >= obj.length;
    let get = () => obj[current];
    return {
        next,
        end,
        get
    };
};
let myIter = Iterator([1, 2, 3]);
while (!myIter.end()) {
    console.log(myIter.get());
    myIter.next();
}

```


## 订阅发布模式


订阅-发布模式：定义了对象之间的一种一对多的依赖关系，当一个对象的状态发生改变时，所有依赖它的对象都可以得到通知。


### 什么是“订阅-发布模式”？


订阅-发布模式：定义了对象之间的一种一对多的依赖关系，当一个对象的状态发生改变时，所有依赖它的对象都可以得到通知。


了解过事件机制或者函数式编程的朋友，应该会体会到“订阅-发布模式”所带来的“**时间解耦**”和“**空间解耦**”的优点。借助函数式编程中闭包和回调的概念，可以很优雅地实现这种设计模式。


### “订阅-发布模式” vs 观察者模式


订阅-发布模式和观察者模式概念相似，但在订阅-发布模式中，订阅者和发布者之间多了一层中间件：一个被抽象出来的信息调度中心。


但其实没有必要太深究 2 者区别，因为《Head First 设计模式》这本经典书都写了：**发布+订阅=观察者模式**。**其核心思想是状态改变和发布通知。**在此基础上，根据语言特性，进行实现即可。


### python3 实现


python 中我们定义一个事件类`Event`, 并且为它提供 事件监听函数、（事件完成后）触发函数，以及事件移除函数。任何类都可以通过继承这个通用事件类，来实现“订阅-发布”功能。


```python
class Event:
  def __init__(self):
    self.client_list = {}
  def listen(self, key, fn):
    if key not in self.client_list:
      self.client_list[key] = []
    self.client_list[key].append(fn)
  def trigger(self, *args, **kwargs):
    fns = self.client_list[args[0]]
    length = len(fns)
    if not fns or length == 0:
      return False
    for fn in fns:
      fn(*args[1:], **kwargs)
    return False
  def remove(self, key, fn):
    if key not in self.client_list or not fn:
      return False
    fns = self.client_list[key]
    length = len(fns)
    for _fn in fns:
      if _fn == fn:
        fns.remove(_fn)
    return True
# 借助继承为对象安装 发布-订阅 功能
class SalesOffice(Event):
  def __init__(self):
    super().__init__()
# 根据自己需求定义一个函数：供事件处理完后调用
def handle_event(event_name):
  def _handle_event(*args, **kwargs):
    print("Price is", *args, "at", event_name)
  return _handle_event
if __name__ == "__main__":
  # 创建2个回调函数
  fn1 = handle_event("event01")
  fn2 = handle_event("event02")
  sales_office = SalesOffice()
  # 订阅event01 和 event02 这2个事件，并且绑定相关的 完成后的函数
  sales_office.listen("event01", fn1)
  sales_office.listen("event02", fn2)
  # 当两个事件完成时候，触发前几行绑定的相关函数
  sales_office.trigger("event01", 1000)
  sales_office.trigger("event02", 2000)
  sales_office.remove("event01", fn1)
  # 打印：False
  print(sales_office.trigger("event01", 1000))
```


### ES6 实现


JS 中一般用事件模型来代替传统的发布-订阅模式。任何一个对象的原型链被指向`Event`的时候，这个对象便可以绑定自定义事件和对应的回调函数。


```typescript
const Event = {
    clientList: {},
    // 绑定事件监听
    listen(key, fn) {
        if (!this.clientList[key]) {
            this.clientList[key] = [];
        }
        this.clientList[key].push(fn);
        return true;
    },
    // 触发对应事件
    trigger() {
        const key = Array.prototype.shift.apply(arguments),
            fns = this.clientList[key];
        if (!fns || fns.length === 0) {
            return false;
        }
        for (let fn of fns) {
            fn.apply(null, arguments);
        }
        return true;
    },
    // 移除相关事件
    remove(key, fn) {
        let fns = this.clientList[key];
        // 如果之前没有绑定事件
        // 或者没有指明要移除的事件
        // 直接返回
        if (!fns || !fn) {
            return false;
        }
        // 反向遍历移除置指定事件函数
        for (let l = fns.length - 1; l >= 0; l--) {
            let _fn = fns[l];
            if (_fn === fn) {
                fns.splice(l, 1);
            }
        }
        return true;
    }
};
// 为对象动态安装 发布-订阅 功能
const installEvent = obj => {
    for (let key in Event) {
        obj[key] = Event[key];
    }
};
let salesOffices = {};
installEvent(salesOffices);
// 绑定自定义事件和回调函数
salesOffices.listen(
    "event01",
    (fn1 = price => {
        console.log("Price is", price, "at event01");
    })
);
salesOffices.listen(
    "event02",
    (fn2 = price => {
        console.log("Price is", price, "at event02");
    })
);
salesOffices.trigger("event01", 1000);
salesOffices.trigger("event02", 2000);
salesOffices.remove("event01", fn1);
// 输出: false
// 说明删除成功
console.log(salesOffices.trigger("event01", 1000));
```


## 命令模式


命令模式定义：将一个请求封装为一个对象，从而使我们可用不同的请求对客户进行参数化；对请求排队或者记录请求日志，以及支持可撤销的操作。


### 什么是“命令模式”？


命令模式（别名：动作模式、事务模式）定义：将一个请求封装为一个对象，从而使我们可用不同的请求对客户进行参数化；对请求排队或者记录请求日志，以及支持可撤销的操作。


简单来说，它的**核心思想**是：不直接调用类的内部方法，而是通过给“指令函数”传递参数，由“指令函数”来调用类的内部方法。


在这过程中，分别有 3 个不同的主体：调用者、传递者和执行者。


### 应用场景


当想降低调用者与执行者（类的内部方法）之间的耦合度时，可以使用此种设计模式。比如：设计一个命令队列，将命令调用记入日志。


### ES6 实现


为了方便演示，这里模拟了购物的场景。封装一个商场类，可以查看已有商品的名称和单价。


```typescript
// 为了方便演示，mock的假数据
const mockData = {
    10001: {
        name: "电视",
        price: 3888
    },
    10002: {
        name: "MacPro",
        price: 17000
    }
};
/**
 * 商品类（执行者）
 */
class Mall {
    static request(id) {
        if (!mockData[id]) {
            return `商品不存在`;
        }
        const { name, price } = mockData[id];
        return `商品名: ${name} 单价: ${price}`;
    }
    static buy(id, number) {
        if (!mockData[id]) {
            return `商品不存在`;
        }
        if (number < 1) {
            return `至少购买1个商品`;
        }
        return mockData[id].price * number;
    }
}
```


毫无疑问，我们可以直接调用商场类上的方法。但是这样会增加调用者和执行者的耦合度。如果之后的函数名称改变了，那么修改成本自然高。


根据命令模式的思想，封装一个“传递者”函数，专门用来传递指令和参数。如果之后商场类的函数名改变了，只需要在“传递者”函数中做个简单映射即可。


```typescript
/**
 * 传递者
 */
function execCmd(cmd, ...args) {
    if (typeof Mall[cmd] !== "function") {
        return;
    }
    console.log(`<LOG> At ${Date.now()}, call ${cmd}`); // 真实场景中，可以向数据库写入日志，或者微服务上报日志
    return Mall[cmd](...args);
}
```


最后，下面代码展示了外界的“调用者”如何调用命令：


```javascript
// 调用者
console.log(execCmd("request", 10001));
console.log("10个mbp的总价是", execCmd("buy", 10002, 10));
```


### 更多思考


在写这篇文章的时候，发现“命令模式”的思路，可以很好的组织不同版本的 api 调用。只需要在“传递者”函数中进行版本识别，然后传递到对应版本的类中即可。


这对于外界调用者来说，是无感的。即便想调用老版本的函数 api，也可以通过给“传递者”函数指定代表版本的参数来实现。


## 责任链模式


责任链模式定义：多个对象均有机会处理请求，从而解除发送者和接受者之间的耦合关系。这些对象连接成为“链式结构”，每个节点转发请求，直到有对象处理请求为止。


其**核心思想**就是：请求者不必知道是谁哪个节点对象处理的请求。如果当前不符合终止条件，那么把请求转发给下一个节点处理。


### 什么是“责任链模式”？


责任链模式定义：多个对象均有机会处理请求，从而解除发送者和接受者之间的耦合关系。这些对象连接成为“链式结构”，每个节点转发请求，直到有对象处理请求为止。


其**核心思想**就是：请求者不必知道是谁哪个节点对象处理的请求。如果当前不符合终止条件，那么把请求转发给下一个节点处理。


而当需求具有“传递”的性质时（代码中其中一种体现就是：多个`if、else if、else if、else`嵌套），就可以考虑将每个分支拆分成一个节点对象，拼接成为责任链。


### 优点与代价


优点：

- 可以根据需求变动，任意向责任链中添加 / 删除节点对象
- 没有固定的“开始节点”，可以从任意节点开始

代价：**责任链最大的代价就是每个节点带来的多余消耗**。当责任链过长，很多节点只有传递的作用，而不是真正地处理逻辑。


### 代码实现


为了方便演示，模拟常见的“日志打印”场景。模拟了 3 种级别的日志输出：

- `LogHandler`: 普通日志
- `WarnHandler`：警告日志
- `ErrorHandler`：错误日志

首先我们会构造“责任链”：`LogHandler` -> `WarnHandler` -> `ErrorHandler`。`LogHandler`作为链的开始节点。


如果是普通日志，那么就由 `LogHandler` 处理，停止传播；如果是 Warn 级别的日志，那么 `LogHandler` 就会自动向下传递，`WarnHandler` 接收到并且处理，停止传播；Error 级别日志同理。


### ES6 实现


```typescript
class Handler {
    constructor() {
        this.next = null;
    }
    setNext(handler) {
        this.next = handler;
    }
}
class LogHandler extends Handler {
    constructor(...props) {
        super(...props);
        this.name = "log";
    }
    handle(level, msg) {
        if (level === this.name) {
            console.log(`LOG: ${msg}`);
            return;
        }
        this.next && this.next.handle(...arguments);
    }
}
class WarnHandler extends Handler {
    constructor(...props) {
        super(...props);
        this.name = "warn";
    }
    handle(level, msg) {
        if (level === this.name) {
            console.log(`WARN: ${msg}`);
            return;
        }
        this.next && this.next.handle(...arguments);
    }
}
class ErrorHandler extends Handler {
    constructor(...props) {
        super(...props);
        this.name = "error";
    }
    handle(level, msg) {
        if (level === this.name) {
            console.log(`ERROR: ${msg}`);
            return;
        }
        this.next && this.next.handle(...arguments);
    }
}
/******************以下是测试代码******************/
let logHandler = new LogHandler();
let warnHandler = new WarnHandler();
let errorHandler = new ErrorHandler();
// 设置下一个处理的节点
logHandler.setNext(warnHandler);
warnHandler.setNext(errorHandler);
logHandler.handle("error", "Some error occur");
```


### Python3 实现


```python
class Handler():
    def __init__(self):
        self.next = None
    def set_next(self, handler):
        self.next = handler
class LogHandler(Handler):
    def __init__(self):
        super().__init__()
        self.__name = "log"
    def handle(self, level, msg):
        if level == self.__name:
            print('LOG: ', msg)
            return
        if self.next != None:
            self.next.handle(level, msg)
class WarnHandler(Handler):
    def __init__(self):
        super().__init__()
        self.__name = "warn"
    def handle(self, level, msg):
        if level == self.__name:
            print('WARN: ', msg)
            return
        if self.next != None:
            self.next.handle(level, msg)
class ErrorHandler(Handler):
    def __init__(self):
        super().__init__()
        self.__name = "error"
    def handle(self, level, msg):
        if level == self.__name:
            print('ERROR: ', msg)
            return
        if self.next != None:
            self.next.handle(level, msg)
# 以下是测试代码
log_handler = LogHandler()
warn_handler = WarnHandler()
error_handler = ErrorHandler()
# 设置下一个处理的节点
log_handler.set_next(warn_handler)
warn_handler.set_next(error_handler)
log_handler.handle("error", "Some error occur")
```


## 状态模式


状态模式：对象行为是根据状态改变，而改变的。


### 什么是“状态模式”？


状态模式：对象行为是根据状态改变，而改变的。


正是由于内部状态的变化，导致对外的行为发生了变化。例如：相同的方法在不同时刻被调用，行为可能会有差异。


### 优缺点


优点：

- 封装了转化规则，对于大量分支语句，可以考虑使用状态类进一步封装。
- 每个状态都是确定的，对象行为是可控的。

缺点：状态模式的**实现关键**是将事物的状态都封装成单独的类，这个类的各种方法就是“此种状态对应的表现行为”。因此，程序开销会增大。


### ES6 实现


在 JavaScript 中，可以直接用 JSON 对象来代替状态类。


下面代码展示的就是 FSM（有限状态机）里面有 3 种状态：`download`、`pause`、`deleted`。控制状态转化的代码也在其中。


`DownLoad`类就是，常说的`Context`对象，它的行为会随着状态的改变而改变。


```typescript
const FSM = (() => {
    let currenState = "download";
    return {
        download: {
            click: () => {
                console.log("暂停下载");
                currenState = "pause";
            },
            del: () => {
                console.log("先暂停, 再删除");
            }
        },
        pause: {
            click: () => {
                console.log("继续下载");
                currenState = "download";
            },
            del: () => {
                console.log("删除任务");
                currenState = "deleted";
            }
        },
        deleted: {
            click: () => {
                console.log("任务已删除, 请重新开始");
            },
            del: () => {
                console.log("任务已删除");
            }
        },
        getState: () => currenState
    };
})();
class Download {
    constructor(fsm) {
        this.fsm = fsm;
    }
    handleClick() {
        const { fsm } = this;
        fsm[fsm.getState()].click();
    }
    hanldeDel() {
        const { fsm } = this;
        fsm[fsm.getState()].del();
    }
}
// 开始下载
let download = new Download(FSM);
download.handleClick(); // 暂停下载
download.handleClick(); // 继续下载
download.hanldeDel(); // 下载中，无法执行删除操作
download.handleClick(); // 暂停下载
download.hanldeDel(); // 删除任务
```


### Python3 实现


python 的代码采用的是“面向对象”的编程，没有过度使用函数式的闭包写法（python 写起来也不难）。


因此，负责状态转化的类，专门拿出来单独封装。


其他 3 个状态类的状态，均由这个状态类来管理。


```python
# 负责状态转化
class StateTransform:
  def __init__(self):
    self.__state = 'download'
    self.__states = ['download', 'pause', 'deleted']
  def change(self, to_state):
    if (not to_state) or (to_state not in self.__states) :
      raise Exception('state is unvalid')
    self.__state = to_state
  def get_state(self):
    return self.__state
# 以下是三个状态类
class DownloadState:
  def __init__(self, transfomer):
    self.__state = 'download'
    self.__transfomer = transfomer
  def click(self):
    print('暂停下载')
    self.__transfomer.change('pause')
  def delete(self):
    print('先暂停, 再删除')
class PauseState:
  def __init__(self, transfomer):
    self.__state = 'pause'
    self.__transfomer = transfomer
  def click(self):
    print('继续下载')
    self.__transfomer.change('download')
  def delete(self):
    print('删除任务')
    self.__transfomer.change('deleted')
class DeletedState:
  def __init__(self, transfomer):
    self.__state = 'deleted'
    self.__transfomer = transfomer
  def click(self):
    print('任务已删除, 请重新开始')
  def delete(self):
    print('任务已删除')
# 业务代码
class Download:
  def __init__(self):
    self.state_transformer = StateTransform()
    self.state_map = {
      'download': DownloadState(self.state_transformer),
      'pause': PauseState(self.state_transformer),
      'deleted': DeletedState(self.state_transformer)
    }
  def handle_click(self):
    state = self.state_transformer.get_state()
    self.state_map[state].click()
  def handle_del(self):
    state = self.state_transformer.get_state()
    self.state_map[state].delete()
if __name__ == '__main__':
  download = Download()
  download.handle_click(); # 暂停下载
  download.handle_click(); # 继续下载
  download.handle_del(); # 下载中，无法执行删除操作
  download.handle_click(); # 暂停下载
  download.handle_del(); # 删除任务
```


## 解释器模式


解释器模式: 提供了评估语言的**语法**或**表达式**的方式。


### 什么是“解释器模式？


解释器模式定义: 提供了评估语言的**语法**或**表达式**的方式。


这是基本不怎么使用的一种设计模式。确实想不到什么场景一定要用此种设计模式。


实现这种模式的**核心**是：

1. 抽象表达式：主要有一个`interpret()`操作
- 终结符表达式：`R = R1 + R2`中，`R1` `R2`就是终结符
- 非终结符表达式：`R = R1 - R2`中，就是终结符
1. 环境(Context): **存放**文法中各个**终结符**所对应的具体值。比如前面`R1`和`R2`的值。

### 优缺点


优点显而易见，每个**文法规则**可以表述为一个类或者方法。这些文法互相不干扰，符合“开闭原则”。


由于每条文法都需要构建一个类或者方法，文法数量上去后，很难维护。并且，语句的执行效率低（一直在不停地互相调用）。


### ES6 实现


为了方便说明，下面省略了“抽象表达式”的实现。


```typescript
class Context {
    constructor() {
        this._list = []; // 存放 终结符表达式
        this._sum = 0; // 存放 非终结符表达式(运算结果)
    }
    get sum() {
        return this._sum;
    }
    set sum(newValue) {
        this._sum = newValue;
    }
    add(expression) {
        this._list.push(expression);
    }
    get list() {
        return [...this._list];
    }
}
class PlusExpression {
    interpret(context) {
        if (!(context instanceof Context)) {
            throw new Error("TypeError");
        }
        context.sum = ++context.sum;
    }
}
class MinusExpression {
    interpret(context) {
        if (!(context instanceof Context)) {
            throw new Error("TypeError");
        }
        context.sum = --context.sum;
    }
}
/** 以下是测试代码 **/
const context = new Context();
// 依次添加: 加法 | 加法 | 减法 表达式
context.add(new PlusExpression());
context.add(new PlusExpression());
context.add(new MinusExpression());
// 依次执行: 加法 | 加法 | 减法 表达式
context.list.forEach(expression => expression.interpret(context));
console.log(context.sum);
```


## 备忘录模式


备忘录模式：属于行为模式，保存某个状态，并且在**需要**的时候直接获取，而不是**重复计算**。


### 什么是备忘录模式


备忘录模式：属于行为模式，保存某个状态，并且在**需要**的时候直接获取，而不是**重复计算**。


**注意**：备忘录模式实现，不能破坏原始封装。也就是说，能拿到内部状态，将其保存在外部。


### 应用场景


最典型的例子是“斐波那契数列”递归实现。


不借助备忘录模式，数据一大，就容易爆栈；借助备忘录，算法的时间复杂度可以降低到$ O(N) $


除此之外，数据的缓存等也是常见应用场景。


### ES6 实现


首先模拟了一下简单的拉取分页数据。


如果当前数据没有被缓存，那么就模拟异步请求，并将结果放入缓存中；


如果已经缓存过，那么立即取出即可，无需多次请求。


**main.js**：


```typescript
const fetchData = (() => {
    // 备忘录 / 缓存
    const cache = {};
    return page =>
        new Promise(resolve => {
            // 如果页面数据已经被缓存, 直接取出
            if (page in cache) {
                return resolve(cache[page]);
            }
            // 否则, 异步请求页面数据
            // 此处, 仅仅是模拟异步请求
            setTimeout(() => {
                cache[page] = `内容是${page}`;
                resolve(cache[page]);
            }, 1000);
        });
})();
// 以下是测试代码
const run = async () => {
    let start = new Date().getTime(),
        now;
    // 第一次: 没有缓存
    await fetchData(1);
    now = new Date().getTime();
    console.log(`没有缓存, 耗时${now - start}ms`);
    // 第二次: 有缓存 / 备忘录有记录
    start = now;
    await fetchData(1);
    now = new Date().getTime();
    console.log(`有缓存, 耗时${now - start}ms`);
};
run();
```


最近在项目中还遇到一个场景，在`React`中加载微信登陆二维码。


这需要编写一个插入`script`标签的函数。


要考虑的情况是：

1. 同一个`script`标签不能被多次加载
2. 对于加载错误，要正确处理
3. 对于几乎同时触发加载函数的情况, 应该考虑锁住

基于此，**main2.js**文件编码如下：


```typescript
// 备忘录模式: 防止重复加载
const loadScript = src => {
    let exists = false;
    return () =>
        new Promise((resolve, reject) => {
            if (exists) return resolve();
            // 防止没有触发下方的onload时候, 又调用此函数重复加载
            exists = true;
            // 开始加载
            let script = document.createElement("script");
            script.src = src;
            script.type = "text/javascript";
            script.onerror = ev => {
                // 加载失败: 允许外部再次加载
                script.remove();
                exists = false;
                reject(new Error("Load Error"));
            };
            script.onload = () => {
                // 加载成功: exists一直为true, 不会多次加载
                resolve();
            };
            document.body.appendChild(script);
        });
};
/************** 测试代码 **************/
// 专门用于加载微信SDK的代码
const wxLoader = loadScript(
    "<https://res.wx.qq.com/connect/zh_CN/htmledition/js/wxLogin.jser>"
);
// html中只有1个微信脚本
setInterval(() => {
    wxLoader()
        .then()
        .catch(error => console.log(error.message));
}, 5000);
```


在`index2.html`中引入上述代码，即可查看效果：


```html
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta http-equiv="X-UA-Compatible" content="ie=edge" />
        <title>Document</title>
    </head>
    <body>
        <script src="./main2.js"></script>
    </body>
</html>
```


### python3 实现


这里实现一下借助“备忘录模式”优化过的、递归写法的“斐波那契数列”。


```python
def fibonacci(n):
  # 结果缓存
  mem = {1: 1, 2: 1}
  def _fibonacci(_n):
    # 是否缓存
    if _n in mem:
      return mem[_n]
    mem[_n] = _fibonacci(_n - 1) + _fibonacci(_n - 2)
    return mem[_n]
  return _fibonacci(n)
if __name__ == '__main__':
  print(fibonacci(999))
```


## 模版模式


模板模式是：抽象父类定义了子类需要重写的相关方法。并且这些方法，仍然是通过父类方法调用的。


### 什么是模板模式？


模板模式是：抽象父类定义了子类需要重写的相关方法。并且这些方法，仍然是通过父类方法调用的。


根据描述，父类提供了“模板”并决定是否调用，子类进行具体实现。


### 应用场景


一些系统的架构或者算法骨架，由“BOSS”编写抽象方法，具体的实现，交给“小弟们”实现。


而用不用“小弟们”的方法，还是看“BOSS”的心情。


### ES6 实现


`Animal`是抽象类，`Dog`和`Cat`分别具体实现了`eat()`和`sleep()`方法。


`Dog`或`Cat`实例可以通过`live()`方法调用`eat()`和`sleep()`。


**注意**：`Cat`和`Dog`实例会被**自动添加**`live()`方法。不暴露`live()`是为了防止`live()`被子类重写，保证父类的**控制权**。


```typescript
class Animal {
    constructor() {
        // this 指向实例
        this.live = () => {
            this.eat();
            this.sleep();
        };
    }
    eat() {
        throw new Error("模板类方法必须被重写");
    }
    sleep() {
        throw new Error("模板类方法必须被重写");
    }
}
class Dog extends Animal {
    constructor(...args) {
        super(...args);
    }
    eat() {
        console.log("狗吃粮");
    }
    sleep() {
        console.log("狗睡觉");
    }
}
class Cat extends Animal {
    constructor(...args) {
        super(...args);
    }
    eat() {
        console.log("猫吃粮");
    }
    sleep() {
        console.log("猫睡觉");
    }
}
/********* 以下为测试代码 ********/
// 此时, Animal中的this指向dog
let dog = new Dog();
dog.live();
// 此时, Animal中的this指向cat
let cat = new Cat();
cat.live();
```


## 参考

- 《JavaScript 设计模式与开发实践》
- [策略模式-Python 四种实现方式](https://zhuanlan.zhihu.com/p/30576518)
- [Python 设计模式 - 策略模式](http://www.isware.cn/python-design-pattern/03-strategy/)
- [python 迭代器](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143178254193589df9c612d2449618ea460e7a672a366000)
- [维基百科·订阅-发布模式](https://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern)
- [观察者模式和订阅-发布模式的不同](https://www.zhihu.com/question/23486749)
- [javascript 之 责任链模式](https://www.cnblogs.com/editor/p/5679552.html)
- [职责链模式](https://www.yiibai.com/python_design_patterns/python_design_patterns_chain_of_responsibility.html)
- [23 种设计模式全解析](https://www.cnblogs.com/geek6/p/3951677.html)
- [菜鸟教程状态模式](http://www.runoob.com/design-pattern/state-pattern.html)
- [菜鸟教程--解释器模式](http://www.runoob.com/design-pattern/interpreter-pattern.html)
- [@工匠若水](https://blog.csdn.net/yanbober/article/details/45537601)
- [《JavaScript 设计模式 10》模板方法模式](http://www.alloyteam.com/2012/10/commonly-javascript-design-patterns-template-method-pattern/)

