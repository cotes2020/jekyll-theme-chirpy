 
- [# 面向对象设计](#-面向对象设计)
  - [解题策略](#解题策略)
    - [Abstractions, Object and Decoupling](#abstractions-object-and-decoupling)
    - [继承/组合/参数化类型](#继承组合参数化类型)
  - [设计模式](#设计模式)
    - [创建型](#创建型)
      - [单例模式 Singleton Pattern](#单例模式-singleton-pattern)
      - [工厂模式 Factory Pattern](#工厂模式-factory-pattern)
    - [结构型](#结构型)
      - [适配器](#适配器)
    - [行为型](#行为型)
      - [观察者](#观察者)
      - [状态](#状态)
  - [工具箱](#工具箱)
    - [有限状态机](#有限状态机)
    - [多态](#多态)
    - [创建型设计模式补充](#创建型设计模式补充)
      - [Builder](#builder)
    - [结构型设计模式补充](#结构型设计模式补充)
      - [Bridge](#bridge)
      - [Composite](#composite)
      - [Decorator](#decorator)
      - [Façade](#façade)
      - [Proxy](#proxy)
    - [行为型设计模式补充](#行为型设计模式补充)
      - [Chain of Responsibility](#chain-of-responsibility)
      - [Command](#command)
      - [Iterator](#iterator)
      - [Template Method](#template-method)
- [Design Patterns](#design-patterns)
  - [Singleton Class](#singleton-class)
  - [Factory Method](#factory-method)
- [设计模式](#设计模式-1)
  - [面向对象](#面向对象)
- [Handling the Question](#handling-the-question)
  - [Design](#design)
  - [Algorithms that Scale](#algorithms-that-scale)
- [Key Concepts](#key-concepts)
  - [Horizontal vs. Vertical Scaling](#horizontal-vs-vertical-scaling)
  - [Load Balancer](#load-balancer)
  - [Database Denormalization and NoSQL](#database-denormalization-and-nosql)
  - [Database Partitioning (Sharding)](#database-partitioning-sharding)
  - [Caching](#caching)
  - [Asynchronous Processing & Queues](#asynchronous-processing--queues)
  - [Networking Metrics](#networking-metrics)
  - [MapReduce](#mapreduce)
- [Considerations](#considerations)


# 面向对象设计
======

对于初级程序员的面试，最难的部分可能就是所谓的设计题。

设计题可以分成两个类别：
- 系统架构设计: 涉及的技术往往包括数据库，并发处理和分布式系统等等
- 利用面向对象编程原理进行程序设计。


1.  题目描述
    * 往往非常简单，如：设计一个XX系统。或者：你有没有用过XXX，给你看一下它的界面和功能，你来设计一个。
2.  阐述题意
    * 面试者需向面试官询问系统的具体要求。如，需要什么功能，需要承受的流量大小，是否需要考虑可靠性，容错性等等。
3.  面试者提供一个初步的系统设计
4.  面试官这对初步的系统中提出一些后续的问题：如果要加某个功能怎么办，如果流量大了怎么办，如何考虑一致性，如果机器挂了怎么办。
5.  面试者根据面试官的后续问题逐步完善系统设计
6.  完成面试

总体特点是以交流为主，画图和代码为辅。

根据我们面试别人和参与面试的经验，先从面试官的角度给出一些考量标准：

* 适应变化的需求(Adapt to the changing requirements )
* 设计干净，优美，考虑周到的系统(Produce a system that is clean, elegant, well thought )
* 解释为何这么实现(Explain why you choose this implementation )
* 对自己的能力水平很熟练(Be familiar with your experience level to make decisions )
* 在一些高层结构和复杂性方面有设计(Answer in high level of scale and complexity )


---

## 解题策略 


### Abstractions, Object and Decoupling

通常，关于OOP，面试官会让面试者设计一个程序框架，该程序能够实现一些特定的功能。

比如，如何实现一个音乐播放器，如何设计一个车库管理程序等等。

对于此类问题，设计的关键过程一般包括
- 抽象(abstraction)，
- 设计对象(object)
- 和设计合理的层次／接口(decoupling)。

这里，我们举一个例子简单说明这些过程分别需要做些什么，在“模式识别”给出更为具体和完整的实例。

### 继承/组合/参数化类型

在面向对象中最常用的两种代码复用技术就是**继承**和**组合**。

在设计对象的时候，“Is-A”表示一种继承关系。
- 比如，班长“Is-A”学生，
- 那么，学生就是基类，班长就是派生类。


在确定了派生关系之后，我们需要分析什么是`基类变量(base class variables)`什么是`子类变量(sub class variables)`，并由此确定基类和派生类之间的联系。

而“Has-A”表示一种从属关系，这就是组合。
- 比如，班长“Has-A”眼镜，那就可以解释为班长实例中拥有一个眼镜实例变量(instance variable)。
- 在具体实现的时候，
- 班长类中定义一个眼镜的基类指针。“在生成班长实例的时候，同时生成一个眼镜实例，利用眼镜的基类指针指向这个实例。任何关于眼镜的操作函数都可以利用这个基类指针实现多态(polymorphism)。注意，多态是OOP相关的一个重要概念，也是面试常考的概念之一。关于多态的解释请见“工具箱”。

在通常情况下，我们更偏向于“Has-A”的设计模式。
- 因为该模式减少了两个实例之间的相关性。
- 对于继承的使用，通常情况下我们会定义一个虚基类，由此派生出多个不同的实例类。
- 在业界的程序开发中，多重继承并不常见，Java甚至不允许从多个父类同时继承，产生一个子类。

此外，我们还要提及参数化类型。参数化类型，或者说模版类也是一种有效的代码复用技术。在C++的标准模版库中大量应用了这种方式。例如，在定义一个List的变量时，List被另一个类型String所参数化。

设计模式着重于代码的复用，所以在选择复用技术上，有必要看看上述三种复用技术优劣。

**继承**

* 通过继承方式，子类能够非常方便地改写父类方法，同时
* 保留部分父类方法，可以说是能够最快速地达到`代码复用`。
* 继承是在静态编译时候就定义了，所以无法再运行时刻改写父类方法。
* 因为子类没有改写父类方法的话，就相当于依赖了父类这个方法的实现细节,被认为破坏封装性。
* 并且如果父类接口定义需要更改时，子类也需要提更改响应接口。

**组合**

* 对象组合通过获得其他对象引用而在运行时刻动态定义的。
* 组合要求对象遵守彼此约定，进而要求更仔细地定义接口，而这些接口并不妨碍你将一个对象和另外一个对象一起使用。
* 对象只能够通过接口来访问，所以我们并没有破坏封装性。
* 而且只要抽象类型一致，对象是可以被替换的。
* 使用组合方式，我们可以将类层次限制在比较小的范围内，不容易产生类的爆炸。
* 相对于继承来说,组合可能需要编写“更多的代码。

**参数化类型**

* 参数化类型方式是基于接口的编程，在一定程度上消除了类型给程序设计语言带来的限制。
* 相对于组合方式来说，缺少的是动态修改能力。
* 因为参数化类型本身就不是面向对象语言的一个特征，所以在面向对象的设计模式里面，没有一种模式是于参数化类型相关的。
* 实践上我们方面是可以使用参数化类型来编写某种模式的。

**总结**

* **对象组合**允许你在运行时刻改变被组合的行为，但是它存在间接性，相对来说比较低效。
* **继承**允许你提供操作的缺省实现，通过子类来重定义这些操作，但是不能够在运行时改变。
* **参数化**允许你改变所使用的类型，同样不能够在运行时改变。


----


## 设计模式 

所谓的设计模式是指人们在开发软件的过程中，对于一些普适需求而总结的设计模版。根据模式目的可以分为三类：

* 创建型(Creational).创建型模式 与 `对象的创建`相关。
* 结构型(Structural).结构型模式 处理 `类或者是对象的组合`。
* 行为型(Behavioral).行为型模式 对 `类或者是对象怎样交互和怎样分配职责进行描述`。

下面我们对每种类型进行介绍。具体的模式请见“工具箱”。值得提醒的是，在面试或工作中不可盲目相信设计模式。设计模式更多地只是提供一些思路，能够直接套用设计模式的情况并不多，更多的时候是对现成设计模式的改进和组合。所以对于设计模式的学习更多应该着眼于模式的意图，而不是模式的具体实现方法。


---


### 创建型

一个类的创建型模式使用继承改变被实例化的类，而一个对象的创建型模式将实例化委托给另外一个对象。 在这些模式中有两种不断出现的主旋律：

* 将该系统使用哪些具体的类封装起来
* 隐藏了实例是如何被创建和存储的

总而言之，效果就是用户创建对象的结果是得到一个`基类指针`，
- 用户通过基类指针调用继承类的方法。
- 用户不需要知道在使用哪些继承类。



#### 单例模式 Singleton Pattern

意图：单例模式(Singleton Pattern)是一种常见的设计模式。其目的在于保证一个类仅仅有一个实例并且提供一个访问它的全局访问点。

这个模式主要的对比对象就是全局变量。相对于全局变量，单例有下面这些好处：

* 全局变量不能够保证只有一个实例。
* 某些情况下面，我们需要稍微计算才能够初始化这个单例。全局变量也行但是不自然。
* C++下面没有保证全局变量的初始化顺序.

比如
- 音乐播放器设计中，我们引入了 歌曲管理器 实现数据的存储。
- 歌曲管理器在整个程序中应当`实例化`一次，其他所有关于数据的操作都应该在这个实例上进行。
- 所以，歌曲管理器应该应用单例模式。
- 实现单例模式的关键在于利用`静态变量(static variable)`，
- 通过判断静态变量是否已经初始化判断该类是否已经实例化。
- 此外，还需要把构造函数设为私有函数，通过公共接口getSharedInstance进行调用。我们举例如下：

```java
// Example for singleton pattern
// class definition
class MySingleton {
    private:
    // Private Constructor
        MySingleton();
    // Stop the compiler generating methods of copy the object
        MySingleton(const MySingleton &copy);    // Not Implemented
        MySingleton &operator=(const MySingleton &copy);    // Not Implemented
        static MySingleton *m_pInstance;

    public:
        static MySingleton *getSharedInstance() {
            if (!m_pInstance) {
                m_pInstance = new MySingleton;
            }
            return m_pInstance;
        }
};
    // in the source file
    MySingleton *MySingleton::m_pInstance = NULL;
    

注意，本例中的实现方式针对非多线程的情况。如果有过个线程想要同时调用getSharedInstance函数，则需要用mutex保护下列代码：

    pthread_mutex_lock(&mutex);
    if (!m_pInstance) {
        m_pInstance = new MySingleton;
    }
    pthread_mutex_unlock(&mutex);
```





#### 工厂模式 Factory Pattern

意图：抽象类需要创建一个对象时，让子类决定实例化哪一个类

所谓的工厂模式(Factory Pattern)，就是指定义一个创建对象的接口，但让实现这个接口的类来决定实例化哪个类。
- 通常，接口提供传入参数，用以决定实例化什么类。
- 工厂模式常见于工具包和框架中，当需要生成一系列类似的子类时，可以考虑使用工厂模式。举例如下：


```java
    // class for factory pattern
    enum ImageType{
        GIF,
        JPEG
    };
    
    class ImageReader {
        // implementation for image reader base class
    };
    
    class GIFReader : public ImageReader {
        // implementation for GIF reader derived class
    };
    
    class JPEGReader : public ImageReader {
        // implementation for JPEG reader derived class
    };
    
    class ImageReaderFactory {
    public:
        static ImageReader *imageReaderFactoryMethod(ImageType imageType) {
            ImageReader *product = NULL;
            switch (imageType) {
                case GIF: product = new GIFReader();
                case JPEG: product = new JPEGReader();
                //...
            }
            return product;
        }
    };
```



---

### 结构型

类的结构型模式采用继承机制来组合接口。对象的结构型模式不是对接口进行组合， 而是描述如何对一些对象进行组合，从而实现新功能。

#### 适配器

意图：适配器(Adapter)将一个类的接口转化成为客户希望的另外一个接口。

假设A实现了Foo()接口，但是B希望A同样实现一个Bar()接口，事实上Foo()基本实现了Bar()接口功能。 Adapter模式就是设计一个新类C，C提供Bar()接口，但实现的方式是内部调用 A的Foo()。

在实现层面上可以通过继承和组合两种方式达到目的：C可以继承A，或者C把A作为自己的成员变量。两者孰优孰劣需要视情况而定。


---

### 行为型

行为型涉及到算法和对象之间职责的分配。行为模式不仅描述对象或者类的功能行为，还描述它们之间的通信模式。 这些模式刻画了在运行时难以追踪的控制流，它们将你的注意从控制流转移到对象之间的联系上来。

#### 观察者

意图：观察者模式(observer)定义对象之间的依赖关系，当一个对象“状态发生改变的话，所有依赖这个对象的对象都会被通知并且进行更新。

被观察的对象需要能够动态地增删观察者对象，这就要求观察者提供一个公共接口比如Update()。然后每个观察者实例注册到被观察对象里面去，在被观察对象状态更新时候能够遍历所有注册观察者并且调用Update()。

至于观察者和被观察之间是采用push还是pull模式完全取决于应用。对于观察这件事情来说的话， 我们还可以引入方面(Aspect)这样一个概念，在注册观察者的时候不仅仅只是一个观察者对象， 还包括一个Aspect参数，可以以此告诉被观察者仅在发生某些变化时通过调用Update()通知我。

#### 状态

意图：状态模式(state)允许一个对象在其内部状态改变时改变它的行为。

这里状态模式意图是，对于实例A，当A的状态改变时，将A可能改变的行为封装成为一个类S(有多少种可能的状态就有多少个S的子类,比如S1,S2,S3等)。当A的状态转换时，在A内部切换S的实例。从A的用户角度来看，A的接口不变，但A的行为因A的状态改变而改变，这是因为行为的具体实现由S完成。

工具箱
---

### 有限状态机

参见[这里](http://en.wikipedia.org/wiki/Finite-state_machine)

### 多态

在C++中，最常见的多态指的是用基类指针指向一个派生类的实例，当用该指针调用一个基类中的虚函数时，实际调用的是派生类的函数实现，而不是基类函数。如果该指针指向另一个派生类实例，则调用另一个派生类的函数实现。因此，比如工厂模式返回一个实例，上层函数不需要知道实例来自哪个派生类，只需要用一个基类指针指向它，就可以直接获得需要的行为。从编译的角度来看，函数的调用地址并不是在编译阶段静态决定，而是在运行阶段，动态地决定函数的调用地址。

多态是通过虚函数表实现的。当基类中用virtual关键字定义函数时，系统自动分配一个指针，指向该类的虚函数表。虚函数表中存储的是函数指针。在生成派生类的时候，会将派生类中对应的函数的地址写到虚函数表。之后，当利用基类指针调用函数时，先通过虚函数表指针找到对应的虚函数表，再通过表内存储的函数指针调用对应函数。由于函数指针指向派生类的实现，因此函数行为自然也就是派生类中定义的行为了。

### 创建型设计模式补充

#### Builder

**意图：将一个复杂对象构建过程和元素表示分离。**

假设我们需要创建一个复杂对象，而这个复杂对象是由很多元素构成的。这些元素的组合逻辑可能非常复杂， 但是逻辑组合和创建这些元素是无关的，独立于这些元素本身的。

那么我们可以将元素的组合逻辑以及元素构建分离，元素构建我们单独放在Builder这样一个类里面，而元素的组合逻辑通过Director来指导，Director内部包含Builder对象。创建对象是通过Director来负责组合逻辑部分的， Director内部调用Builder来创建元素并且组装起来。最终通过Builder的GetResult来获得最终复杂对象。

### 结构型设计模式补充

#### Bridge

**意图：将抽象部分和具体实现相分离，使得它们之间可以独立变化。**

一个很简单的例子就是类Shape,有个方法Draw\[抽象\]和DrawLine\[具体\]和DrawText\[具体\],而Square和SquareText 继承于Shape实现Draw()这个方法，Square调用DrawLine()，而SquareText调用DrawLine()+DrawText()。而且假设DrawLine和DrawText分别有LinuxDrawLine,LinuxDrawText和Win32DrawLine和Win32DrawText。如“果我们简单地 使用子类来实现的话，比如构造LinuxSquare,LinuxSquareText,Win32Square和Win32SquareText，那么很快就会类爆炸。

事实上我们没有必要在Shape这个类层面跟进变化，即通过继承Shape类实现跨平台，而只需要在实现底层跟进变化。为此我们就定义一套接口，如例子中的DrawLine和DrawText，然后在Linux和Win32下实现一个这样接口实例(比如称为跨平台GDI)，最终 Shape内部持有这个GDI对象，Shape的DrawLine和DrawText只是调用GDI的接口而已。这样，我们把Shape及其子类的DrawLine和DrawText功能Bridge到GDI，GDI可以通过工厂模式在不同平台下实现不同的实例。

例子中Shape成为了完全抽象的部分，具体实现完全交给GDI类，若以后需要增加更多的平台支持，开发者也不需要添加更多的Shape子类，只需要扩展GDI即可。总之，抽象部分是和具体实现部分需要独立开来的时候，就可以使用Bridge模式。

#### Composite

**意图：将对象组合成为树形以表示层级结构，对于叶子和非叶子节点对象使用需要有一致性。**

Composite模式强调在这种层级结构下，叶子和非叶子节点需要一致对待，所以关键是需要定义一个抽象类，作为叶节点的子节点。 然后对于叶子节点操作没有特殊之处，而对于非叶子节点操作不仅仅需要操作自身，还要操作所管理的子节点。 至于遍历子节点和处理顺序是由应用决定的，在Composite模式里面并不做具体规定。

#### Decorator

**意图：动态地给对象添加一些额外职责，通过组合而非继承方式完成。**

给对象添加一些额外职责，例如增加新的方法，很容易会考虑使用子类方式来实现。使用子类方式实现很快但是却不通用，考虑一个抽象类X，子类有SubX1,SubX2等。现在需要为X提供一个附加方法echo，如果用继承的方式添加，那么需要为每个子类都实现echo方法，并且代码往往是重复的。我们可以考虑Decorator模式，定义一个新类，使其持有持有指向X基类的指针，并且新类只需要单独实现echo方法，而其他方法直接利用X基类指针通过多态调用即可。

值得注意的是，装饰出来的对象必须包含被装饰对象的所有接口。所以很明显这里存在一个问题， 那就是X一定不能够有过多的方法，不然Echo类里面需要把X方法全部转发一次(理论上说Echo类可以仅转发X的部分方法，但Decorator默认需要转发被装饰类的全部方法)。

#### Façade

**意图：为子系统的一组接口提供一个一致的界面。**

编译器是一个非常好的的例子。对于编译器来说，有非常多的子系统包括词法语法解析，语义检查,中间代码生成，代码优化，以及代码生成这些逻辑部件。但是对于大多数用户来说，不关心这些子系统，而只是关心编译这一个过程。

所以我们可以提供Compiler的类，里面只有很简单的方法比如Compile()，让用户直接使用Compile()这个接口。 一方面用户使用起来简单，另外一方面子系统和用户界面耦合性也降低了。

Facade模式对于大部分用户都是满足需求的。对于少部分不能够满足需求的用户，可以让他们绕过Facade模式提供的界面， 直接控制子系统即可。就好比GCC提供了很多特殊优化选项来让高级用户来指定，而不是仅仅指定-O2这样的选项。

#### Proxy

**意图：为其他对象提供一种代理以控制对这个对象的访问。**

通常使用Proxy模式是想针对原本要访问的对象做一些手脚，以达到一定的目的，包括访问权限设置，访问速度优化，或者是加入一些自己特有的逻辑。至于实现方式上，不管是继承还是组合都行，可能代价稍微有些不同，视情况而定。但是偏向组合方式，因为对于Proxy而言，完全可以定义一套新的访问接口。

Adapter,Decorator以及Proxy之间比较相近，虽然说意图上差别很大，但是对于实践中， 三者都是通过引用对象来增加一个新类来完成的，但是这个新类在生成接口方面有点差别：

* Adapter模式的接口一定要和对接的接口相同。
* Decorator模式的接口一定要包含原有接口，通常来说还要添加新接口。
* Proxy模式完全可以重新定义一套新的接口

### 行为型设计模式补充

#### Chain of Responsibility

**意图：将对象连成一条链并沿着链传递某个请求，直到有某个对象处理它为止。**

大部分情况下连接起来的对象本身就存在一定的层次结构关系，少数情况下面这些连接起来的对象是内部构造的。 职责链通常与Composite模式一起使用，一个构件的父构件可以作为它的后继结点。许多类库使用职责链模式来处理事件， 比如在UI部分的话View本来就是相互嵌套的，一个View对象可能存在Parent View对象。如果某个UI不能够处理事件的话， 那么完全可以交给Parent View来完成事件处理以此类推。

#### Command

**意图：将一个请求封装成为一个对象。**

Command模式可以说是回调机制(Callback)的一个面向对象的替代品。对于回调函数来说需要传递一个上下文参数(context)， 同时内部附带一些逻辑。将上下文参数以及逻辑包装起来的话那么就是一个Command对象。 Command对象接口可以非常简单只有Execute/UnExecute，但是使用Command对象来管理请求之后， 就可以非常方便地实现命令的复用，排队，重做，撤销，事务等。

#### Iterator

**意图：提供一种方法顺序访问一个聚合对象中各个元素，但是又不需要暴露该对象内部表示。**

将遍历机制与聚合对象表示分离，使得我们可以定义不同的迭代器来实现不同的迭代策略，而无需在聚合对象接口上面列举他们。 一个健壮的迭代器,应该保证在聚合对象上面插入和删除操作不会干扰遍历，“同时不需要copy这个聚合对象。 一种实现方式就是在聚合对象上面注册某个迭代器，一旦聚合对象发生改变的话，需要调整迭代器内部的状态。

#### Template Method

**意图：定义一个操作里面算法的骨架，而将一些步骤延迟到子类。**

假设父类A里面有抽象方法Step1(),Step2(),默认方法Step3()。并且A提供一个操作X()，分别依次使用Step1(),Step2(),Step3()。对于A的子类，通过实现自己的Step1(),Step2() (选择性地实现Step3())，提供属于子类的X具体操作。 这里操作X()就是算法的骨架，子类需要复写其中部分step，但不改变X的执行流程。

很重要的一点是模板方法必须指明哪些操作是钩子操作(可以被重定义的，比如Step3),以及哪些操作是抽象操作“(必须被重定义，比如Step1和Step2)。要有效地重用一个抽象类，子类编写者必须明确了解哪些操作是设计为有待重定义的。

* * *

1.  Handle Ambiguity
    * make assumptions & ask clarifying questions
    * **who** is going to use it and **how** they are going to use it
    * who, what, where, when, how, why
2.  Define the core objects Suppose we are designing for a restaurant. Our core objects might be things like `Table`, `Guest`, `Party`, `Order`, `Meal`, `Employee`, `Server`, and `Host`.
3.  Analyze Relationships
4.  Investigate Actions

Design Patterns
===============

Singleton and Factory Method design patterns are widely used in intervies.

Singleton Class
---------------

Ensures that a class has only on instance and ensures access to the instance through the application. It can be useful in cases where you have a global object with exactly one instance.

    public class Restaurant{
        private static Restaurant _instance = null;
        protected Restaurant() {...}
        public static Restaurant getInstance(){
            if (_instance == null){
                _instance = new Restaurant();
            }
            return _instance;
        }
    }
    

Factory Method
--------------

Offers an interface for creating an instance of a class, with its subclasses deciding which class to instantiate.

    public class CardGame {
        public static CardGame createCardGame(GameType type){
            if (type == GameType.Poker) {
                return new PokerGame();
            }
            else if (type == GameType.BlackJack) {
                return new BlackJackGame();
            }
            return null;
        }
    }
    

* * *

设计模式
====

面向对象
----

**面向对象的三个基本特征是：封装、继承、多态**

* 封装
    * 封装最好理解了。封装是面向对象的特征之一，是对象和类概念的主要特性。封装，也就是把客观事物封装成抽象的类，并且类可以把自己的数据和方法只让可信的类或者对象操作，对不可信的进行信息隐藏。
* 继承
    * 继承是指这样一种能力：它可以使用现有类的所有功能，并在无需重新编写原来的类的情况下对这些功能进行扩展。通过继承创建的新类称为“子类”或“派生类”，被继承的类称为“基类”、“父类”或“超类”。
    * 要实现继承，可以通过“继承”（Inheritance）和“组合”（Composition）来实现。
* 多态性
    * 多态性（polymorphisn）是允许你将父对象设置成为和一个或更多的他的子对象相等的技术，赋值之后，父对象就可以根据当前赋值给它的子对象的特性以不同的方式运作。简单的说，就是一句话：允许将子类类型的指针赋值给父类类型的指针。
    * 实现多态，有两种方式，覆盖和重载。覆盖和重载的区别在于，覆盖在运行时决定，重载是在编译时决定。并且覆盖和重载的机制不同，例如在 Java 中，重载方法的签名必须不同于原先方法的，但对于覆盖签名必须相同。

* * *

Handling the Question
=====================

* **Communicate**: A key goal of system design questions is to evaluate your ability to communicate. Stay engaged with the interviewer. Ask them questions. Be open about the issues of your system.
* **Go broad first**: Don't dive straight into the algorithm part or get excessively focused on one part.
* **Use the whiteboard**: Using a whiteboard helps your interviewer follow your proposed design. Get up to the whiteboard in the very beginning and use it to draw a picture of what you're proposing.
* **Acknowledge interview concerns**: Your interviewer will likely jump in with concers. Don't brush them off; validate them. Acknowledge the issues your interviewer points out and make changes accordingly.
* **Be careful about assumptions**: An incorrect assumption can dramatically change the problem.
* **State your assumptions explicitly**: When you do make assumptions, state them. This allows your interviewer to correct you if you're mistaken, and shows that you at least know what assumptions you're making.
* **Estimate when necessary**: In many cases, you might not have the data you need. You can estimate this with other data you know.
* **Drive**: As the candidate, you should stay in the driver's seat. This doesn't mean you don't talk to your interviewer; in fact, you _must_ talk to your interviewer. However, you should be driving through the question. Ask questions. Be open about tradeoffs. Continue to go deeper. Continue to make improvements.

Design
------

1.  Scope the Problem
2.  Make Reasonable Assumption
3.  Draw the Major Components
4.  Identify the Key Issues
5.  Redesign for the Key Issues

Algorithms that Scale
---------------------

In some cases, you're being asked to design a single feature or algorithm, but you have to do it in a scalable way.

1.  Ask Questiosn
2.  Make Believe
3.  Get Real
4.  Solve Problems

Key Concepts
============

Horizontal vs. Vertical Scaling
-------------------------------

* Vertical scaling means increasing the resoures of a specific node. For example, you might add additional memory to a server to improve its ability to handle load changes.
* Horizontal scaling means increasing the number of nodes. For example, you might add additional servers, thus decreasing the load on any one server.

Vertiacal scaling is generally easer than horizontal scaling, but it's limited.

Load Balancer
-------------

Typically, some frontend parts of a scalable website will be thrown behind a load balancer. This allows a system to distribute the load evenly so that one server doesn't crash and take down the whole system. To do so, of course, you have to build out a network of cloned servers that all have essentially the same code and access to the same data.

Database Denormalization and NoSQL
----------------------------------

Joins in a relational database such as SQL can get very slow as the system grows bigger. For this reason, you would generally avoid them.

Denormalization is one part of this. Denormalization means adding redundant information into a database to speed up reads. For example, imagine a database describing projects and tasks (in addition to the project table).

Or, you can go with a NoSQL database. A NoSQL database does not support joins and might structure data in a different way. It is designed to scale better..

Database Partitioning (Sharding)
--------------------------------

Sharding means splitting the data across multiple machines while ensuring you have a way of figuring out which data is on which machine.

A few common ways of partitioning include:

* **Vertical Partitioning**: This is basically partitioning by feature.
* **Key-Based (or Hash-Based) Partitioning**: This uses some part of the data to partition it. A very simple way to do this is to allocate N servers and put he data on mode(key, n). One issue with this is that the number of servers you have is effectively fixed. Adding additional servers means reallocating all the data -- a very expensive task.
* **Directory-Based Partitioning**: In this scheme, you maintain a lookup table for where the data can be found. This makes it relatively easy to add additional servers, but it comes with two major drawbacks. First the lookup table can be a single point of failure. Second, constantly access this table impacts performance.

Caching
-------

An in-memory cache can deliver very rapid results. It is a simple key-value pairing and typically sits between your application layer and your data store.

Asynchronous Processing & Queues
--------------------------------

Slow operations should ideally be done asynchronously. Otherwise, a user might get stuck waiting and waiting for a process to complete.

Networking Metrics
------------------

* **Bandwidth**: This is the maximum amount of data that can be transferred in a unit of time. It is typically expressed in bits per seconds.
* **Throughput**: Whereas bandwidth is the maximum data that can be transferred in a unit of time, throughput is the actual amoutn of data that is transferred.
* **Latency**: This is how long it takes data to go from one end to the other. That is, it is the delay between the sender sending information (even a very small chunk of data) and the receiver receiving it.

MapReduce
---------

A MapReduce program is typically used to process large amounts of data.

* Map takes in some data and emits a pair
* Reduce takes a key and a set of associated values and reduces them in some way, emitting a new key and value.

MapReduce allows us to do a lot of processing in parallel, which makes processing huge amounts of data more scalable.

Considerations
==============

* **Failures**: Essentially any part of a system can fail. You'll need to plan for many or all of these failures.
* **Availability and Reliability**: Availability is a function of the percentage of time the system is operatoinal. Redliability is a function of the probability that the system is operational for a certain unit of time.
* **Read-heavy vs. Write-heavy**: Whether an application will do a lot of reads or a lot of writes implacts the design. If it's write-heavy, you could consider queuing up the writes (but think about potential failure here!). If it's read-heavy, you might want to cache.
* **Security**: Security threats can, of course, be devastating for a system. Think about the tyupes of issues a system might face and design around thos.

  
  

* * *

[« 位操作](14520595848890.html "Previous Post: 位操作")

[Swap Bits »](14520596469127.html "Next Post: Swap Bits")

$(function(){ var currentURL = '14520596997643.html'; $('#side-nav a').each(function(){ if($(this).attr('href') == currentURL){ $(this).parent().addClass('active'); } }); });

* * *

Copyright © 2015 Powered by [MWeb](http://www.mweb.im),  Theme used [GitHub CSS](http://github.com).

[TOP](#header)

$(document).foundation();