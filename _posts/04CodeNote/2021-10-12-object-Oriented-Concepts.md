---
title: Coding - Object Oriented Programming
date: 2021-10-12 11:11:11 -0400
description:
categories: [04CodeNote]
tags: []
---

- [Object-oriented programming 面向对象编程](#object-oriented-programming-面向对象编程)
  - [`面向过程`和`OOP`在程序流程上的不同之处。](#面向过程和oop在程序流程上的不同之处)
    - [code different](#code-different)
  - [Object-oriented programming language](#object-oriented-programming-language)
  - [Object-Oriented Design Principles](#object-oriented-design-principles)
    - [派生关系](#派生关系)
    - [继承/组合/参数化 类型 (复用技术)](#继承组合参数化-类型-复用技术)
      - [组合](#组合)
      - [继承](#继承)
      - [参数化/模版类类型](#参数化模版类类型)
      - [封装](#封装)
      - [Abstraction](#abstraction)
      - [Encapsulation](#encapsulation)
      - [Modularity](#modularity)
      - [Inheritance](#inheritance)
      - [有限状态机](#有限状态机)
      - [polymorphisn 多态](#polymorphisn-多态)
  - [设计模式](#设计模式)
    - [创建型](#创建型)
      - [单例模式 Singleton Pattern](#单例模式-singleton-pattern)
      - [工厂模式 Factory Pattern](#工厂模式-factory-pattern)
      - [Builder](#builder)
    - [行为型](#行为型)
      - [观察者模式 observer](#观察者模式-observer)
      - [状态模式 state](#状态模式-state)
      - [职责链模式 Chain of Responsibility](#职责链模式-chain-of-responsibility)
      - [Command](#command)
      - [Iterator](#iterator)
      - [Template Method](#template-method)
    - [结构型](#结构型)
      - [适配器 Adapter](#适配器-adapter)
      - [Bridge](#bridge)
      - [Composite](#composite)
      - [Decorator](#decorator)
      - [Façade](#façade)
      - [Proxy](#proxy)
- [Packages and Imports](#packages-and-imports)
  - [Import](#import)
- [Class & Object](#class--object)
  - [Class](#class)
    - [Access control to Class](#access-control-to-class)
    - [Python `class&Instance`](#python-classinstance)
    - [Java](#java)
  - [Object](#object)
    - [Class hierarchy versus instance hierarchy](#class-hierarchy-versus-instance-hierarchy)
- [difference between class and object:](#difference-between-class-and-object)
  - [Types of Class](#types-of-class)
  - [Uses of Class](#uses-of-class)
  - [Use of Object](#use-of-object)
- [example](#example)
  - [OOPs in Python](#oops-in-python)
  - [OOPs in Java: Classes and Objects](#oops-in-java-classes-and-objects)
- [2.2. Inheritance](#22-inheritance)
  - [22.1. Introduction: Class Inheritance](#221-introduction-class-inheritance)
    - [Class Inheritance example](#class-inheritance-example)
    - [example1](#example1)
    - [example2](#example2)
  - [Polymorphism and Dynamic Dispatch](#polymorphism-and-dynamic-dispatch)
  - [Inheriting Variables and Methods](#inheriting-variables-and-methods)
    - [Mechanics of Defining a Subclass](#mechanics-of-defining-a-subclass)
    - [How the interpreter looks up attributes](#how-the-interpreter-looks-up-attributes)
  - [code implement](#code-implement)
    - [The `__init__()` Method for a Child Class](#the-__init__-method-for-a-child-class)
    - [Instances as Attributes](#instances-as-attributes)
    - [`Overrid` 覆写 Methods](#overrid-覆写-methods)
  - [22.4. Invoke 调用 the Parent Class’s Method](#224-invoke-调用-the-parent-classs-method)
  - [继承和多态](#继承和多态)
    - [继承](#继承-1)
    - [继承好处](#继承好处)
    - [多态](#多态)
    - [多态的好处](#多态的好处)
  - [静态语言 vs 动态语言](#静态语言-vs-动态语言)
  - [Importing Classes](#importing-classes)
    - [Importing a Single Class](#importing-a-single-class)
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

---

# Object-oriented programming 面向对象编程



1. Machine language
2. Assembly language
3. **structured programming language**
   1. The basic principle of the structured programming approach is to divide a program into functions and modules.
   2. The use of modules and functions makes the program more understandable and readable.
   3. It helps to write cleaner code and to maintain control over the functions and modules.
   4. This approach gives importance to functions rather than data.
   5. It focuses on the development of large software applications, for example, C was used for modern operating system development.
   6. The programming languages: PASCAL (introduced by Niklaus Wirth) and C (introduced by Dennis Ritchie) follow this approach.
4. **procedural programming language**
   1. This approach is also known as the top-down approach.
   2. In this approach, a program is divided into functions that perform specific tasks.
   3. This approach is mainly used for medium-sized applications.
   4. Data is global, and all the functions can access global data.
   5. The basic drawback of the procedural programming approach is that data is not secured because data is global and can be accessed by any function.
   6. Program control flow is achieved through function calls and goto statements.
   7. The programming languages: FORTRAN (developed by IBM) and COBOL (developed by Dr. Grace Murray Hopper) follow this approach.
   8. These programming constructs were developed in the late 1970s and 1980s. There were still some issues with these languages, though they fulfilled the criteria of well-structured programs, software, etc. They were not as structured as the requirements were at that time. They seem to be over-generalized and don’t correlate with real-time applications.
   9. To solve such kinds of problems, OOP, an object-oriented approach was developed as a solution.
5. **object oriented programming**


Ask
1. Handle Ambiguity
* make assumptions & ask clarifying questions
* **who** is going to use it and **how** they are going to use it
* who, what, where, when, how, why
2. Define the core objects Suppose we are designing for a restaurant. Our core objects might be things like `Table`, `Guest`, `Party`, `Order`, `Meal`, `Employee`, `Server`, and `Host`.
3. Analyze Relationships
4. Investigate Actions



## `面向过程`和`OOP`在程序流程上的不同之处。

`procedural programming` 面向过程
- focus is on writing function/procedure which operate on data.
- 把计算机程序 视为一系列的命令集合，即一组函数的顺序执行。
- 为了简化程序设计，面向过程把函数继续切分为子函数，即把大块函数通过切割成小块函数来降低系统的复杂度。


`Object Oriented Programming - OOP 面向对象`的程序设计
- focus is on the creation of objects which contain both data and functionality together.
- 一种程序设计思想。
- 把`Object`作为程序的基本单元
- 一个`Object`包含了数据和操作数据的函数。
- 把 计算机程序 视为一组对象的集合，而每个对象都可以接收其他对象发过来的消息，并处理这些消息，计算机程序的执行就是一系列消息在各个对象之间传递。
- basically designed to overcome the drawback of the above programming methodologies, which were not so close to real-world applications. 
- The demand was increased, but still, conventional methods were used. 
- This new approach brought a revolution in the programming methodology field.
- OOP allows the writing of programs with the help of certain classes and real-time objects.
- very close to the real-world and its applications because the state and behaviour of these classes and objects are almost the same as real-world objects. 




### code different

1. `面向过程`的程序

    ```py
    # 为了表示一个学生的成绩，用一个dict表示：
    std1 = { 'name': 'Michael', 'score': 98 }
    std2 = { 'name': 'Bob', 'score': 81 }
    # 处理学生成绩可以通过函数实现，比如打印学生的成绩：
    def print_score(std):
        print('%s: %s' % (std['name'], std['score']))
    ```  

2. `面向对象`的程序设计思想

   - 首选思考的不是程序的执行流程，而是Student这种数据类型应该被视为一个对象，这个对象拥有name和score这两个属性（`Property`）。
   - 如果要打印一个学生的成绩
     - 首先创建出学生对应的对象，

      ```py
      class Student(object):
          def __init__(self, name, score):
              self.name = name
              self.score = score
          def print_score(self):
              print('%s: %s' % (self.name, self.score))

      bart = Student('Bart Simpson', 59)
      lisa = Student('Lisa Simpson', 87)
      ```        

     - 然后，给对象发一个print_score消息，让对象自己把自己的数据打印出来。
       - 给对象发消息实际上就是调用`对象对应的关联函数`
       - 称之为`对象的方法（Method）`。

      ```py
      bart.print_score()
      lisa.print_score()
      ```

`面向对象`的设计思想是从自然界中来的，因为在自然界中，类（Class）和实例（Instance）的概念是很自然的。
- `Class`是一种抽象概念，比如我们定义的Class——Student，是指学生这个概念
- 而实例（`Instance`）则是一个个具体的Student，比如，Bart Simpson和Lisa Simpson是两个具体的Student。

所以，面向对象的设计思想是抽象出`Class`，根据`Class`创建`Instance`。
- 面向对象的抽象程度又比函数要高，因为一个Class既包含数据，又包含操作数据的方法
 





---


## Object-oriented programming language


对于初级程序员的面试，最难的部分可能就是所谓的设计题。

设计题可以分成两个类别：
- 系统架构设计: 涉及的技术往往包括数据库，并发处理和分布式系统等等
- 利用面向对象编程原理进行程序设计。


1. 题目描述
* 往往非常简单，如：设计一个XX系统。或者：你有没有用过XXX，给你看一下它的界面和功能，你来设计一个。
2. 阐述题意
* 面试者需向面试官询问系统的具体要求。如，需要什么功能，需要承受的流量大小，是否需要考虑可靠性，容错性等等。
3. 面试者提供一个初步的系统设计
4. 面试官这对初步的系统中提出一些后续的问题：如果要加某个功能怎么办，如果流量大了怎么办，如何考虑一致性，如果机器挂了怎么办。
5. 面试者根据面试官的后续问题逐步完善系统设计
6. 完成面试

总体特点是以交流为主，画图和代码为辅。

根据我们面试别人和参与面试的经验，先从面试官的角度给出一些考量标准：

* 适应变化的需求(Adapt to the changing requirements )
* 设计干净，优美，考虑周到的系统(Produce a system that is clean, elegant, well thought )
* 解释为何这么实现(Explain why you choose this implementation )
* 对自己的能力水平很熟练(Be familiar with your experience level to make decisions )
* 在一些高层结构和复杂性方面有设计(Answer in high level of scale and complexity )
 
---




## Object-Oriented Design Principles

Chief among the principles of the object-oriented approach, which are intended to facilitate the goals outlined above
- Abstraction
- Encapsulation 封装
- Modularity


通常，关于OOP，面试官会让面试者设计一个程序框架，该程序能够实现一些特定的功能。
- 比如，如何实现一个音乐播放器，如何设计一个车库管理程序等等。

对于此类问题，设计的关键过程一般包括
- 抽象(abstraction)，
- 设计对象(object)
- 和设计合理的层次／接口(decoupling)。

这里，我们举一个例子简单说明这些过程分别需要做些什么，在“模式识别”给出更为具体和完整的实例。



### 派生关系

在设计对象的时候，“Is-A”表示一种继承关系。
- 比如，班长“Is-A”学生，
- 那么，学生就是基类，班长就是派生类。
- 派生类 -> 基类1, 基类2, 基类3

在确定了派生关系之后，我们需要分析什么是`基类变量(base class variables)`什么是`子类变量(sub class variables)`，并由此确定基类和派生类之间的联系。



### 继承/组合/参数化 类型 (复用技术)

在面向对象中最常用的两种代码复用技术就是**继承**和**组合**。

设计模式着重于代码的复用，所以在选择复用技术上，有必要看三种复用技术优劣。

**总结**

* **继承**: 允许你提供操作的缺省实现，通过子类来重定义这些操作，但是不能够在运行时改变。
* **对象组合**: 允许你在运行时刻改变被组合的行为，但是它存在间接性，相对来说比较低效。
* **参数化**: 允许你改变所使用的类型，同样不能够在运行时改变。

#### 组合

组合: “Has-A”表示一种从属关系，这就是组合。
- 比如，班长“Has-A”眼镜，那就可以解释为班长实例中拥有一个眼镜实例变量`(instance variable)`。
- 在具体实现的时候，
- 班长类中定义一个眼镜的基类指针。
- 在生成班长实例的时候，同时生成一个眼镜实例，利用眼镜的基类指针指向这个实例。任何关于眼镜的操作函数都可以利用这个基类指针实现`多态(polymorphism)`。
- 在通常情况下，我们更偏向于“Has-A”的设计模式。
  - 因为该模式减少了两个实例之间的相关性。

* 对象组合通过获得其他对象引用而在运行时刻动态定义的。
* 组合要求对象遵守彼此约定，进而要求更仔细地定义接口，而这些接口并不妨碍你将一个对象和另外一个对象一起使用。
* 对象只能够通过接口来访问，所以我们并没有破坏封装性。
* 而且只要抽象类型一致，对象是可以被替换的。
* 使用组合方式，我们可以将类层次限制在比较小的范围内，不容易产生类的爆炸。
* 相对于继承来说,组合可能需要编写“更多的代码。


#### 继承
- 对于继承的使用，通常情况下我们会定义一个虚基类，由此派生出多个不同的实例类。
  * 通过继承创建的新类称为“子类”或“派生类”，
  * 被继承的类称为“基类”、“父类”或“超类”。
- 在业界的程序开发中，多重继承并不常见，Java甚至不允许从多个父类同时继承，产生一个子类。

* 通过继承方式，子类能够非常方便地改写父类方法，
* 同时保留部分父类方法，最快速地达到`代码复用`。
* 使用现有类的所有功能，并在无需重新编写原来的类的情况下对这些功能进行扩展。

* 继承是在静态编译时候就定义了，所以无法再运行时刻改写父类方法。
* 因为子类没有改写父类方法的话，就相当于依赖了父类这个方法的实现细节,被认为破坏封装性。
* 并且如果父类接口定义需要更改时，子类也需要提更改响应接口。


#### 参数化/模版类类型
- 此外，我们还要提及参数化类型。参数化类型，或者说模版类也是一种有效的代码复用技术。
- 在C++的标准模版库中大量应用了这种方式。
- 例如，在定义一个List的变量时，List被另一个类型String所参数化。

* 参数化类型方式是基于接口的编程，在一定程度上消除了类型给程序设计语言带来的限制。
* 相对于组合方式来说，缺少的是动态修改能力。
* 因为参数化类型本身就不是面向对象语言的一个特征，所以在面向对象的设计模式里面，没有一种模式是于参数化类型相关的。
* 实践上我们方面是可以使用参数化类型来编写某种模式的。


#### 封装

- 封装是面向对象的特征之一，是对象和类概念的主要特性。
- 把客观事物封装成抽象的类
- 并且类可以把自己的数据和方法只让可信的类或者对象操作，对不可信的进行信息隐藏。




#### Abstraction

Abstraction refers to the act of representing important and special features without including the background details or explanation about that feature. Data abstraction simplifies database design.

1. Physical Level: 
   1. It describes how the records are stored, which are often hidden from the user. 
   2. It can be described with the phrase, “block of storage.” 

2. Logical Level: 
   1. It describes data stored in the database and the relationships between the data. 
   2. The programmers generally work at this level as they are aware of the functions needed to maintain the relationships between the data. 

3. View Level: 
   1. Application programs hide details of data types and information for security purposes. 
   2. This level is generally implemented with the help of GUI, and details that are meant for the user are shown. 



- The notion of abstraction is to distill a complicated system down to its most fundamental parts.
- Typically, describing the parts of a system involves naming them and explaining their functionality.
- Applying the abstraction paradigm to the design of data structures gives rise to **abstract data types (ADTs)**.

**abstract data types (ADTs)**
- An ADT is a mathematical model of a data structure that specifies `the type of data stored, the operations supported on them, and the types of parameters of the operations`.
- An ADT specifies what each operation does, but not how it does it.










**interface**
- In Java, an ADT can be expressed by an **interface**
- simply a list of method declarations, where each method has an empty body.

**class**
- An ADT is realized by a concrete data structure, which is modeled in Java by a **class**.
- A class defines the data being stored and the operations supported by the objects that are instances of the class.
- Also, unlike interfaces, classes specify how the operations are performed in the body of each method.
- A Java class is said to implement an interface if its methods include all the methods declared in the interface, thus providing a body for them.
- However, a class can have more methods than those of the interface.







#### Encapsulation

- `different components of a software system should not reveal 揭示 the internal details of their respective implementations`.

- It describes the idea of wrapping data and the methods that work on data within one unit, e.g., a class in Java. 
- This concept is often used to hide the internal state representation of an object from the outside.

- One of the **main advantages**: gives programmer freedom to implement the details of a component, without concern that other programmers will be writing code that intricately depends on those internal decisions.

- The only constraint on the programmer of a component is to **maintain the public interface for the component**, as other programmers will be writing code that depends on that interface.
- Encapsulation yields robustness and adaptability, for it allows the implementation details of parts of a program to change without adversely affecting other parts, thereby making it easier to fix bugs or add new functionality with relatively local changes to a component.



#### Modularity
- Modern software systems typically consist of several different components that must interact correctly in order for the entire system to work properly.
- Keeping these interactions straight requires that these different components be well organized.
- Modularity refers to an organizing principle in which `different components of a software system are divided into separate functional units`.
- Robustness is greatly increased because it is easier to test and debug separate components before they are integrated into a larger software system.


#### Inheritance
- A natural way to organize various structural components of a software package is in a hierarchical fashion, 
- with similar abstract definitions grouped together in a level-by-level manner that goes from specific to more general as one traverses up the hierarchy. 

- An example of such a hierarchy is shown in Figure 2.3. Using mathematical notations, the set of houses is a subset of the set of buildings, but a superset of the set of ranches. The correspondence between levels is often referred to as an “is a” relationship, as a house is a building, and a ranch is a house.




#### 有限状态机

参见[这里](http://en.wikipedia.org/wiki/Finite-state_machine)


#### polymorphisn 多态

- 在C++中，最常见的`多态`指的是用基类指针指向一个派生类的实例
- 当用该指针调用一个基类中的虚函数时，实际调用的是派生类的函数实现，而不是基类函数。
- 如果该指针指向另一个派生类实例，则调用另一个派生类的函数实现。因此，比如工厂模式返回一个实例，上层函数不需要知道实例来自哪个派生类，只需要用一个基类指针指向它，就可以直接获得需要的行为。从编译的角度来看，函数的调用地址并不是在编译阶段静态决定，而是在运行阶段，动态地决定函数的调用地址。

多态是通过虚函数表实现的。当基类中用virtual关键字定义函数时，系统自动分配一个指针，指向该类的虚函数表。虚函数表中存储的是函数指针。在生成派生类的时候，会将派生类中对应的函数的地址写到虚函数表。之后，当利用基类指针调用函数时，先通过虚函数表指针找到对应的虚函数表，再通过表内存储的函数指针调用对应函数。由于函数指针指向派生类的实现，因此函数行为自然也就是派生类中定义的行为了。

- 允许你将父对象设置成为和一个或更多的他的子对象相等的技术，赋值之后，父对象就可以根据当前赋值给它的子对象的特性以不同的方式运作。
  * 简单的说，就是一句话：允许将子类类型的指针赋值给父类类型的指针。
- 实现多态，有两种方式，覆盖和重载。
  * 覆盖和重载的区别在于，覆盖在运行时决定，重载是在编译时决定。
  * 并且覆盖和重载的机制不同，例如在 Java 中，重载方法的签名必须不同于原先方法的，但对于覆盖签名必须相同。

---


## 设计模式

所谓的设计模式是指人们在开发软件的过程中，对于一些普适需求而总结的设计模版。根据模式目的可以分为三类：

* 创建型(Creational).创建型模式 与 `对象的创建`相关。
* 结构型(Structural).结构型模式 处理 `类或者是对象的组合`。
* 行为型(Behavioral).行为型模式 对 `类或者是对象怎样交互和怎样分配职责进行描述`。

下面我们对每种类型进行介绍。
- 不可盲目相信设计模式。
- 设计模式更多地只是提供一些思路，能够直接套用设计模式的情况并不多，更多的时候是对现成设计模式的改进和组合。
- 所以对于设计模式的学习更多应该着眼于模式的意图，而不是模式的具体实现方法



### 创建型

一个类的创建型模式 使用继承 改变被实例化的类，而一个对象的创建型模式将实例化委托给另外一个对象。

在这些模式中有两种不断出现的主旋律：
* 将该系统使用哪些具体的类封装起来
* 隐藏了实例是如何被创建和存储的

总而言之，效果就是用户创建对象的结果是得到一个`基类指针`，
- 用户通过基类指针调用继承类的方法。
- 用户不需要知道在使用哪些继承类。



#### 单例模式 Singleton Pattern


Ensures that a class has only on instance and ensures access to the instance through the application.
- It can be useful in cases where you have a **global object** with exactly one instance.


意图：其目的在于保证一个类仅仅有一个实例并且提供一个访问它的全局访问点。

这个模式主要的对比对象就是全局变量。相对于全局变量，单例有下面这些好处：

* 全局变量不能够保证只有一个实例。
* 某些情况下面，我们需要稍微计算才能够初始化这个单例。全局变量也行但是不自然。
* C++下面没有保证全局变量的初始化顺序.

比如
- 音乐播放器设计中，我们引入了歌曲管理器实现数据的存储。
- 歌曲管理器在整个程序中应当`实例化`一次，其他所有关于数据的操作都应该在这个实例上进行。
- 所以，歌曲管理器应该应用单例模式。
- 实现单例模式的关键在于利用`静态变量(static variable)`，
- 通过判断静态变量是否已经初始化判断该类是否已经实例化。
- 此外，还需要把构造函数设为私有函数，通过公共接口getSharedInstance进行调用。我们举例如下：



```java
public class Restaurant{
    private static Restaurant _instance = null;

    protected Restaurant() {...}

    public static Restaurant getInstance(){
        if (_instance == null)_instance = new Restaurant();
        return _instance;
    }
}
```    


```c
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

Offers an interface for creating an instance of a class
- with its subclasses deciding which class to instantiate.

意图：抽象类需要创建一个对象时，让子类决定实例化哪一个类
- 所谓的工厂模式(Factory Pattern)，就是指定义一个创建对象的接口，但让实现这个接口的类来决定实例化哪个类。
- 通常，接口提供传入参数，用以决定实例化什么类。
- 工厂模式常见于工具包和框架中，当需要生成一系列类似的子类时，可以考虑使用工厂模式。

举例如下：


```java
public class CardGame {
    public static CardGame createCardGame(GameType type){
        if (type == GameType.Poker) return new PokerGame();
        else if (type == GameType.BlackJack) return new BlackJackGame();
        return null;
    }
}
```


```c
// class for factory pattern
enum ImageType{
    GIF, JPEG
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
            case GIF:
                product = new GIFReader();
            case JPEG:
                product = new JPEGReader();
                //...
        }
        return product;
    }
};
```



#### Builder

意图：将一个复杂对象构建过程和元素表示分离。
- 假设我们需要创建一个复杂对象，而这个复杂对象是由很多元素构成的。
- 这些元素的组合逻辑可能非常复杂，但是逻辑组合和创建这些元素是无关的，独立于这些元素本身的。

那么我们可以将元素的组合逻辑以及元素构建分离，
- 元素构建我们单独放在Builder这样一个类里面，
- 元素的组合逻辑通过Director来指导，Director内部包含Builder对象。
- 创建对象是通过Director来负责组合逻辑部分的，
- Director内部调用Builder来创建元素并且组装起来。
- 最终通过Builder的GetResult来获得最终复杂对象。



---


### 行为型

行为型涉及到算法和对象之间职责的分配。
- 行为模式不仅描述对象或者类的功能行为，还描述它们之间的通信模式。
- 这些模式刻画了在运行时难以追踪的控制流，它们将你的注意从控制流转移到对象之间的联系上来。

#### 观察者模式 observer

意图：观察者模式(observer)定义对象之间的依赖关系，当一个对象“状态发生改变的话，所有依赖这个对象的对象都会被通知并且进行更新。
- 被观察的对象需要能够动态地增删观察者对象，这就要求观察者提供一个公共接口比如Update()。
- 然后每个观察者实例注册到被观察对象里面去，在被观察对象状态更新时候能够遍历所有注册观察者并且调用Update()。
- 至于观察者和被观察之间是采用push还是pull模式完全取决于应用。
- 对于观察这件事情来说的话， 我们还可以引入方面(Aspect)这样一个概念，在注册观察者的时候不仅仅只是一个观察者对象， 还包括一个Aspect参数，可以以此告诉被观察者仅在发生某些变化时通过调用Update()通知我。

#### 状态模式 state

意图：状态模式(state)允许一个对象在其内部状态改变时改变它的行为。
- 这里状态模式意图是，对于实例A，当A的状态改变时，将A可能改变的行为封装成为一个类S(有多少种可能的状态就有多少个S的子类,比如S1,S2,S3等)。当A的状态转换时，在A内部切换S的实例。
- 从A的用户角度来看，A的接口不变，但A的行为因A的状态改变而改变，这是因为行为的具体实现由S完成。



#### 职责链模式 Chain of Responsibility

意图：将对象连成一条链并沿着链传递某个请求，直到有某个对象处理它为止。

- 大部分情况下连接起来的对象本身就存在一定的层次结构关系，
- 少数情况下面这些连接起来的对象是内部构造的。
- 职责链通常与Composite模式一起使用，一个构件的父构件可以作为它的后继结点。
- 许多类库使用`职责链模式`来处理事件， 比如在UI部分的话View本来就是相互嵌套的，一个View对象可能存在Parent View对象。
- 如果某个UI不能够处理事件的话， 那么完全可以交给Parent View来完成事件处理以此类推。



#### Command

意图：将一个请求封装成为一个对象。

- Command模式可以说是回调机制(Callback)的一个面向对象的替代品。
- 对于回调函数来说需要传递一个上下文参数(context)， 同时内部附带一些逻辑。
- 将上下文参数以及逻辑包装起来的话那么就是一个Command对象。
- Command对象接口可以非常简单只有Execute/UnExecute，但是使用Command对象来管理请求之后， 就可以非常方便地实现命令的复用，排队，重做，撤销，事务等。



#### Iterator

意图：提供一种方法顺序访问一个聚合对象中各个元素，但是又不需要暴露该对象内部表示。

- 将遍历机制与聚合对象表示分离，使得我们可以定义不同的迭代器来实现不同的迭代策略，而无需在聚合对象接口上面列举他们。
- 一个健壮的迭代器,应该保证在聚合对象上面插入和删除操作不会干扰遍历，“同时不需要copy这个聚合对象。
- 一种实现方式就是在聚合对象上面注册某个迭代器，一旦聚合对象发生改变的话，需要调整迭代器内部的状态。



#### Template Method

意图：定义一个操作里面算法的骨架，而将一些步骤延迟到子类。

- 假设父类A里面有抽象方法Step1(),Step2(),默认方法Step3()。
- 并且A提供一个操作X()，分别依次使用Step1(),Step2(),Step3()。
- 对于A的子类，通过实现自己的Step1(),Step2() (选择性地实现Step3())，提供属于子类的X具体操作。 这里操作X()就是算法的骨架，子类需要复写其中部分step，但不改变X的执行流程。

很重要的一点是模板方法必须指明哪些操作是**钩子操作**(可以被重定义的，比如Step3),以及哪些操作是**抽象操作**“(必须被重定义，比如Step1和Step2)。
- 要有效地重用一个抽象类，子类编写者必须明确了解哪些操作是设计为有待重定义的。


---


### 结构型

类的**结构型模式**采用继承机制来组合接口。
- 对象的`结构型模式`不是对接口进行组合，而是描述如何对一些对象进行组合，从而实现新功能。


#### 适配器 Adapter

意图：适配器(Adapter)将一个类的接口转化成为客户希望的另外一个接口。
- 假设A实现了Foo()接口，
- 但是B希望A同样实现一个Bar()接口，事实上Foo()基本实现了Bar()接口功能。
- Adapter模式就是设计一个新类C，C提供Bar()接口，但实现的方式是内部调用 A的Foo()。

在实现层面上可以通过继承和组合两种方式达到目的：C可以继承A，或者C把A作为自己的成员变量。两者孰优孰劣需要视情况而定。


#### Bridge

意图：将抽象部分和具体实现相分离，使得它们之间可以独立变化。

一个很简单的例子就是类Shape,
- 有个方法Draw[抽象]和DrawLine[具体]和DrawText[具体],
- 而Square和SquareText 继承于Shape 实现Draw()这个方法，
- Square调用DrawLine()，
- SquareText调用DrawLine()+DrawText()。
- 而且假设DrawLine和DrawText分别有LinuxDrawLine,LinuxDrawText和Win32DrawLine和Win32DrawText。
- 如果我们简单地 使用子类来实现的话，比如构造LinuxSquare,LinuxSquareText,Win32Square和Win32SquareText，那么很快就会类爆炸。

事实上我们没有必要在Shape这个类层面跟进变化，即通过继承Shape类实现跨平台，而只需要在实现底层跟进变化。
- 为此我们就定义一套接口，如例子中的DrawLine和DrawText，然后在Linux和Win32下实现一个这样接口实例(比如称为跨平台GDI)，
- 最终 Shape内部持有这个GDI对象，Shape的DrawLine和DrawText只是调用GDI的接口而已。
- 这样，我们把Shape及其子类的DrawLine和DrawText功能Bridge到GDI，GDI可以通过工厂模式在不同平台下实现不同的实例。

例子中Shape成为了完全抽象的部分，具体实现完全交给GDI类，若以后需要增加更多的平台支持，开发者也不需要添加更多的Shape子类，只需要扩展GDI即可。总之，抽象部分是和具体实现部分需要独立开来的时候，就可以使用Bridge模式。


#### Composite

意图：将对象组合成为树形以表示层级结构，对于叶子和非叶子节点对象使用需要有一致性。

Composite模式强调在这种层级结构下，
- 叶子和非叶子节点需要一致对待，所以关键是需要定义一个抽象类，作为叶节点的子节点。
- 然后对于叶子节点操作没有特殊之处，
- 而对于非叶子节点操作不仅仅需要操作自身，还要操作所管理的子节点。
- 至于遍历子节点和处理顺序是由应用决定的，在Composite模式里面并不做具体规定。


#### Decorator

意图：动态地给对象添加一些额外职责，通过组合而非继承方式完成。

给对象添加一些额外职责，例如增加新的方法，很容易会考虑使用子类方式来实现。使用子类方式实现很快但是却不通用，
- 考虑一个抽象类X，子类有SubX1,SubX2等。现在需要为X提供一个附加方法echo，如果用继承的方式添加，那么需要为每个子类都实现echo方法，并且代码往往是重复的。
- 我们可以考虑Decorator模式，定义一个新类，使其持有持有指向X基类的指针，并且新类只需要单独实现echo方法，而其他方法直接利用X基类指针通过多态调用即可。

值得注意的是，装饰出来的对象必须包含被装饰对象的所有接口。所以很明显这里存在一个问题， 那就是X一定不能够有过多的方法，不然Echo类里面需要把X方法全部转发一次(理论上说Echo类可以仅转发X的部分方法，但Decorator默认需要转发被装饰类的全部方法)。

#### Façade

意图：为子系统的一组接口提供一个一致的界面。

编译器是一个非常好的的例子。对于编译器来说，有非常多的子系统包括词法语法解析，语义检查,中间代码生成，代码优化，以及代码生成这些逻辑部件。但是对于大多数用户来说，不关心这些子系统，而只是关心编译这一个过程。

所以我们可以提供Compiler的类，里面只有很简单的方法比如Compile()，让用户直接使用Compile()这个接口。 一方面用户使用起来简单，另外一方面子系统和用户界面耦合性也降低了。

Facade模式对于大部分用户都是满足需求的。对于少部分不能够满足需求的用户，可以让他们绕过Facade模式提供的界面， 直接控制子系统即可。就好比GCC提供了很多特殊优化选项来让高级用户来指定，而不是仅仅指定-O2这样的选项。


#### Proxy

意图：为其他对象提供一种代理以控制对这个对象的访问。

通常使用Proxy模式是想针对原本要访问的对象做一些手脚，以达到一定的目的，包括访问权限设置，访问速度优化，或者是加入一些自己特有的逻辑。
- 至于实现方式上，不管是继承还是组合都行，可能代价稍微有些不同，视情况而定。
- 但是偏向组合方式，因为对于Proxy而言，完全可以定义一套新的访问接口。

Adapter,Decorator以及Proxy之间比较相近，虽然说意图上差别很大，但是对于实践中， 三者都是通过引用对象来增加一个新类来完成的，但是这个新类在生成接口方面有点差别：

* Adapter模式的接口一定要和对接的接口相同。
* Decorator模式的接口一定要包含原有接口，通常来说还要添加新接口。
* Proxy模式完全可以重新定义一套新的接口



---


---

# Packages and Imports

- Every stand-alone public class defined in Java must be given in a separate file.
- The file name is the name of the class with a .java extension.

**package**
- To aid in the organization of large code repository, Java allows a group of related type definitions to be grouped into what is known as a `package`.
- For types to belong to a package named `packageName`, their source code must all be located in a directory named `packageName` and each file must begin with the line: `package packageName;`


## Import


```java
// Import Statements in java
java.util.Scanner input = new java.util.Scanner(System.in);

import java.util.Scanner;
Scanner input = new Scanner(System.in);


// Importing a Whole Package
import packageName.∗;
```
 

---


# Class & Object

It is the basic concept of OOP; an extended concept of the structure used in C. It is an abstract and user-defined data type. It consists of several variables and functions. 

- The primary purpose of the class is to store data and information. 
- The members of a class define the behaviour of the class. 
- A class is the blueprint of the object, the implementation of the class is the object. 
- The class is not visible to the world, but the object is.





---

## Class

- an entity
- determines how an object will behave and what the object will contain.
- a blueprint or a set of instruction to build a specific type of object.
- It provides initial values for member variables and member functions or methods.


User-defined Classes

Python provides a way to define `new functions` in programs, it also provides a way to `define new classes of objects`.



### Access control to Class

**Access level modifiers** determine `whether other classes can use a particular field or invoke a particular method`.

There are two levels of access control:
- At the top level—`public`, or `package-private` (no explicit modifier).
- At the member level—`public, private, protected`, or `package-private` (no explicit modifier)


class declared with the modifier `public`
- that class is visible to all classes everywhere.
- If a class has no modifier (the default, package-private), it is visible only within its own package (packages are named groups of related classes)


The `private` modifier
- specifies that the member can only be accessed in its own class.


The `protected` modifier
- specifies that the member can only be accessed within its own package (as with package-private)
- and, in addition, by a subclass of its class in another package.

---

### Python `class&Instance`

Python is an `object-oriented programming` language.
- provides features that support `object-oriented programming (OOP)`.


在Python中，所有数据类型都可以视为Object，当然也可以自定义对象。
- 自定义的对象数据类型就是面向对象中的类（`Class`）的概念。

---


### Java

the main “actors” in the object-oriented paradigm are called `objects`.

- Each `object` is an `instance` of a `class`.

- Each `class` presents to the outside world a concise and consistent view of the objects that are instances of this class, without going into too much unnecessary detail or giving others access to the inner workings of the objects.

- The class definition typically specifies the data fields, also known as instance variables, that an object contains, as well as the methods (operations) that an object can execute.  

Software implementations should achieve robustness, adaptability, and reusability.

---

## Object

- a self-contained component
- consists of methods and properties to make a data useful.
- helps to determines the behavior of the class.
- For example
  - send a message to an object
  - asking the object to invoke or execute one of its methods.

- From a programming point of view, an object can be a data structure, a variable, or a function that has a memory location allocated.
- The object is designed as class hierarchies.


User creatable objects
- In a user program, only objects belonging to certain classes can be created directly.
- In the class hierarchy chart, the yellow boxes denote these user-instantiable classes.
- The other classes fall into one of four categories:
  - Non-instantiable superclasses such as Base and View
  - Classes designed to function only as composite class members, such as PlotManager and subclasses of the Transformation class
  - The classes that can have only one instance, such as Error and Workspace; they are automatically instantiated when the HLU library is initialized
  - Classes that are instantiated by certain objects for a specialized purpose on behalf of the user; these currently include the XyDataSpec and AnnoManager classes

Dynamically associated objects
- In addition to the class hierarchy and composite class relationships, the HLU library has a mechanism that allows you to associate independently-created View objects dynamically. You can "overlay" Transform class objects onto a plot object's data space. You can also make any View object into an "annotation" of a plot object. The combination of the base plot object, its overlays, and its annotations acts in many ways like a single object. Plot objects, overlays, and annotations are discussed in the PlotManager class module, and also in the AnnoManager class module.

---


### Class hierarchy versus instance hierarchy
Besides the class hierarchy of subclasses derived from the Base superclass, you should be aware that the HLU library defines an "instance hierarchy" of the objects that are created in the course of executing an HLU program. These two hierarchies are completely distinct, and you should be careful not confuse them.
Whenever you create an object, you must specify the object's "parent" as one of the parameters to the create call. Each object you create is therefore the "child" of some parent object. The initial parent, the "ancestor" of all the objects created, must be an "application" (App) object. Depending on the call used to initialize the HLU library, you may need to create this object yourself, or the library may automatically create it for you.

The instance hierarchy is significant in the following ways:

When you destroy a parent object all its children are destroyed along with it.
A View object must have a Workstation class ancestor that supplies the viewspace on which it is drawn.
The resource database uses the instance hierarchy to determine how resource specifications in resource files apply to particular objects in an HLU program.


---



# difference between class and object:

| **Class**                                                   | **Object**                                                   |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| `template` for creating objects in program.                 | an instance of a class.                                      |
| `logical entity`                                            | Object is a physical entity                                  |
| `does not allocate memory space` when it is created.        | Object allocates memory space when been created.             |
| You can declare class only once.                            | can create more than one object using a class.               |
| Example: Car.                                               | Example: Jaguar, BMW, Tesla, etc.                            |
| Class generates objects                                     | Objects provide life to the class.                           |
| `can't be manipulated` as they are not available in memory. | can be manipulated.                                          |
| `doesn't have any values` associated with the fields.       | Each and every object has values associated with the fields. |
| create class by "class" keyword. `Class XX {}`              | create object by "new" keyword. `XX aa= new XX()`            |

---

## Types of Class

- <font color=red> Derived Classes and Inheritance </font>
  - A derived class is a class which is created or derived from other remining class.
  - It is used for increasing the functionality of base class.
  - This type of class derives and inherits properties from existing class.
  - It can also add or share/extends its own properties.

1. <font color=red> Superclasses </font>
   - A superclass is a class from which you can derive many sub classes.

2. <font color=red> Subclasses </font>
   - A subclass is a class that derives from superclass.

3. <font color=red> Mixed classes </font>
   - combine the functionality from other classes into a new class.
     - inherit the properties of one class to another.
   - It uses a subset of the functionality of class, whereas a derive class uses the complete set of superclass functionality.
   - different
     - A mixed class
       - manages the properties of other classes
       - and may only use a subset of the functionality of a class
     - a derived class
       - uses the complete set of functionality of its superclasses
       - and usually extends this functionality.
   - ![concepts.figure.id.9](https://i.imgur.com/VRvJI7o.gif)

---

## Uses of Class  

- Class is used to <font color=red> hold both data variables and member functions </font>
- for <font color=red> create user define objects </font>
  - provides a way to organize information about data.
- can use class to <font color=red> inherit the property of other class </font>
- take advantage of <font color=red> constructor or destructor </font>
- can be used for a large amount of data and complex applications.

---

## Use of Object  

- <font color=red> give the type of message accepted and the type of returned responses </font>
- use an object to <font color=red> access a piece of memory using an object reference variable </font>
- <font color=red> It is used to manipulate data </font>
- Objects represent <font color=red> a real-world problem </font> for which you are finding a solution.
- It enables data members and member functions to perform the desired task.



---

# example

## OOPs in Python


> Class > Instance > Instance variables/Attributes > Methods

1. import <font color=red> class </font> like `Turtle` or `Screen`

2. create a new <font color=red> instance </font>

    ```py
    import Turtles

    # make a new window for turtles to paint in
    wn = turtle.Screen()  
    # make a new turtle
    alex = turtle.Turtle()  
    ```

   - `alex = turtle.Turtle()`
   - The Python interpreter find that `Turtle` is a <font color=blue> class, not function </font>
   - so it <font color=blue> creates a new instance of the class </font> and returns it.
     - Since the Turtle class was defined in a separate module, (confusingly, also named turtle)
     - had to refer to the class as `turtle.Turtle`.




3. Each instance can have <font color=red> attributes / instance variables </font>

    ```py
    # For example
    # the following code would print out 1100.

    alex.price = 500
    tess.price = 600
    print(alex.price + tess.price)
    ```
    - use `=` to assign values to an attribute


4. <font color=red> Classes have associated methods </font>

    ```py
    alex.forward(50)
    ```

    - The interpreter looks up `alex`
    - finds `alex` is an <font color=blue> instance of the class </font> `Turtle`.
    - Then it looks up the <font color=blue> attribute </font> `forward`
    - finds that it is a <font color=blue> method </font>
      - `Methods` return `values`, like `functions`
      - However, none of the methods of the `Turtle class` return values the way the `len` function does.
    - the interpreter invokes the method, passing 50 as a **parameter**.


The only difference between <font color=red> invocation </font> and <font color=red> function calls </font>
- the `object instance` itself is also passed as a parameter.
- Thus `alex.forward(50)` moves `alex`, while `tess.forward(50)` moves `tess`.


---


## OOPs in Java: Classes and Objects

> design any program using this OOPs approach.



To developing a pet management system, specially meant for dogs.

declared a class called Dog
1. need to <font color=red> model dogs into software entities </font>
   - <img src="https://i.imgur.com/Ditinne.jpg" width="400">


2. need various information about the dogs
   - List down the differences between them.
   - <img src="https://i.imgur.com/WHebHfY.jpg" width="400">
   - differences are also some common characteristics shared by these dogs.
   - These characteristics (breed, age, size, color) can form a data members for your object.



3. list out the common behaviors of these dogs
   - like sleep, sit, eat, etc.
   - So these will be the actions of our software objects.
   - <img src="https://i.imgur.com/0fRi8B6.jpg" width="400">

4. So far we have defined following things,

   - **Class**: Dogs
   - **Data member / objects**: size, age, color, breed, etc.
   - **Methods**: eat, sleep, sit and run.

   - <img src="https://i.imgur.com/kBcpE3f.jpg" width="400">





5. for different values of data members (breed size, age, and color) in Java class, you will get different dog objects.
   - <img src="https://i.imgur.com/8TkR0Io.jpg" width="500">



6. after declared a class called Dog, defined an object of the class called "maltese" using a new keyword.  

    ```java
    // Class Declaration
    class Dog {
        // Instance Variables
        String breed;
        String size;
        int age;
        String color;

        // method 1
        public String getInfo() {
            return ("Breed is: "+breed+" Size is:"+size+" Age is:"+age+" color is: "+color);
        }
    }


    public class Execute{
        public static void main(String\[\] args) {
            Dog maltese = new Dog();
            maltese.breed="Maltese";
            maltese.size="Small";
            maltese.age=2;
            maltese.color="white";
            System.out.println(maltese.getInfo());
        }
    }

    // Output:
    // Breed is: Maltese Size is: Small Age is:2 color is: white
    ```

---


# 2.2. Inheritance



---

## 22.1. Introduction: Class Inheritance


In Java, each class can extend exactly one other class.
- Because of this property, Java is said to allow `only single inheritance among classes`.
- We should also note that even if a class definition makes no explicit use of the **extends** clause, it automatically inherits from a class, `java.lang.Object`, which serves as the `universal superclass in Java`.


Constructors are never inherited in Java.
- When a PredatoryCreditCard instance is created, all of its fields must be properly initialized, including any inherited fields.
- For this reason, the `first operation performed within the body of a constructor must be to invoke a constructor of the superclass`, which is responsible for properly initializing the fields defined in the superclass.
- In Java, a constructor of the superclass is invoked by using the keyword super:
  - `super(cust, mk, acnt, lim, initialBal);`


This use of the `super` keyword is very similar to use of the keyword `this` when invoking a different constructor within the same class
- If a constructor for a `subclass` does not make an explicit call to `super` or `this` as its first command,
  - then an implicit call to `super()`,
  - the zero-parameter version of the superclass constructor, will be made.

---

### Class Inheritance example


### example1
```py
from random import randrange

class Pet():
    boredom_decrement = 5
    hunger_decrement = 5
    boredom_threshold = 5
    hunger_threshold = 10
    sounds = ['Mrrp']

    def __init__(self, name = "Kitty", pet_type="dog"):
        self.name = name
        self.hunger = randrange(self.hunger_threshold)
        self.boredom = randrange(self.boredom_threshold)
        self.sounds = self.sounds[:]  # copy the class attribute, so that when we make changes to it, we won't affect the other Pets in the class
        self.pet_type = pet_type

    def mood(self):
            if self.hunger <= self.hunger_threshold and self.boredom <= self.boredom_threshold:
                if self.pet_type == "dog":
                    return "happy"
                elif self.pet_type == "cat":
                    return "happy, probably"
                else:
                    return "HAPPY"
            elif self.hunger > self.hunger_threshold:
                if self.pet_type == "dog":
                    return "hungry, arf"
                elif self.pet_type == "cat":
                    return "hungry, meeeeow"
                else:
                    return "hungry"
            else:
                return "bored"
```


---

### example2


```py
class Animal(object):      # 编写 Animal类
  def run(self):
      print("Animal is running...")

class Dog(Animal):         # Dog类 继承 Amimal类，没有run方法
  pass
  dog = Dog()
  dog.run()

class Cat(Animal):         # Cat类 继承 Animal类，有自己的run方法
  def run(self):
      print("Cat is running...")
  kitty = Cat()
  kitty.run()

class Tortoise(Animal):
    def run(self):
        print('Tortoise is running slowly...')

class Car(object):    # Car类不继承，有自己的run方法
    def run(self):
        print('Car is running...')

class Stone(object):  # Stone类不继承，也没有run方法
    pass
```



---

## Polymorphism and Dynamic Dispatch

**polymorphism**
- In the context of object-oriented design, it refers to the ability of a `reference variable to take different forms`.

```java
CreditCard card;

// Liskov Substitution Principle
// a variable (or parameter) with a declared type can be assigned an instance from any direct or indirect subclass of that type.
// Informally, this is a manifestation of the “is a” relationship modeled by inheritance, as a predatory credit card is a credit card (but a credit card is not necessarily predatory).
CreditCard card = new PredatoryCreditCard(...); // parameters omitted
```






---

## Inheriting Variables and Methods


### Mechanics of Defining a Subclass

Inheritance
- easy and elegant way to represent these differences.
- A natural way to organize various structural components of a software package



Basically, it works by defining a new class, and using a special syntax to show what the new `sub-class` inherits from a `super-class`.

- to define a `Dog` class as a special kind of `Pet`, you would say that the `Dog` type inherits from the `Pet` type.

- In the definition of the inherited class, only need to specify the methods and instance variables that are different from the parent class (the parent class/superclass)


A hierarchical design is useful in software development, as common functionality can be grouped at the most general level, thereby promoting reuse of code, while differentiated behaviors can be viewed as extensions of the general case.

In object-oriented programming
- the mechanism for a modular and hierarchical organization is a technique known as `inheritance`.
  - This allows a new class to be defined based upon an existing class as the starting point.
- the existing class is typically described as the `base class`, `parent class, or super-class`,
- while the newly defined class is known as the `subclass or child class`.
- the subclass extends the superclass.


When inheritance is used, the subclass `automatically inherits all methods from the superclass` (other than constructors).
- The subclass can differentiate itself from its superclass in two ways.
- It may augment the superclass by adding new fields and new methods.
- It may also specialize existing behaviors by providing a new implementation that overrides an existing method.


```py
from random import randrange

# Here's the original Pet class
class Pet():
    boredom_decrement = 4
    hunger_decrement = 6
    boredom_threshold = 5
    hunger_threshold = 10
    sounds = ['Mrrp']
    def __init__(self, name = "Kitty"):
        self.name = name
        self.hunger = randrange(self.hunger_threshold)
        self.boredom = randrange(self.boredom_threshold)
        self.sounds = self.sounds[:]

    def hi(self):
        print(self.sounds[randrange(len(self.sounds))])
        self.reduce_boredom()

class Cat(Pet):
    sounds = ['Meow']
    def chasing_rats(self):
        return "What are you doing, Pinky? Taking over the world?!"

class Cheshire(Cat):
    def smile(self):
        print(":D :D :D")


cat1 = Cat("Fluffy")
cat1.hi()   # Uses the special Cat hello.

new_cat = Cheshire("Pumpkin") # Cheshire cat instance
new_cat.hi()           # same as Cat!
new_cat.chasing_rats() # OK, as Cheshire inherits from Cat
new_cat.smile() # Only for Cheshire instances (and any classes that you make inherit from Cheshire)
# cat1.smile() # This line would give you an error, because the Cat class does not have this method!

# None of the subclass methods can be used on the parent class, though.
p1 = Pet("Teddy")
p1.hi() # just the regular Pet hello
#p1.chasing_rats() # This will give you an error -- this method doesn't exist on instances of the Pet class.
#p1.smile() # This will give you an error, too. This method does not exist on instances of the Pet class.

```


---

### How the interpreter looks up attributes

how the interpreter looks up attributes:

1. First, it checks for an `instance variable/method` by the name.

2. If an `instance variable/method` by that name is not found, it checks for a `class variable`. (See the previous chapter for an explanation of the difference between instance variables and class variables.)

3. If no `class variable` is found, it looks for a `class variable` in the `parent class`.

4. If no `class variable` is found, the interpreter looks for a class variable in THAT class’s parent (the “grandparent” class).

5. This process goes on until the last ancestor is reached, at which point Python will signal an error.

---

```py
new_cat = Cheshire("Pumpkin")
print(new_cat.name)
```

Python looks for the `instance variable` <font color='red'> name </font> in the `new_cat instance`.
- In this case, it exists. The name on this instance of Cheshire is Pumpkin.

---

```py
cat1 = Cat("Sepia")
cat1.hi()
```

The Python interpreter looks for `hi` in the instance of `Cat`.
- It does not find it, because there’s no statement of the form `cat1.hi = ....` (if you had set an instance variable on Cat called hi it would be a bad idea, because you would not be able to use the method that it inherited anymore. We’ll see more about this later.)
- Then it looks for a <font color=DarkCyank> class variable/method `hi` </font> in the <font color='Medblue'> class Cat </font> , and still doesn’t find it.
- Next, it looks for a <font color=DarkCyank> class variable </font> `hi` on the parent of <font color=Medblue> class Cat </font> , <font color=red> Pet class </font>
- It finds that – there’s a <font color=DarkCyank> method  </font> called `hi` on the <font color=red> Pet class </font>. Because of the () after hi, the method is invoked.

---

```py
p1 = Pet("Teddy")
p1.chasing_rats()
```

The Python interpreter looks for an <font color=DarkCyank>  instance variable/method </font> called `chasing_rats` on the <font color='red'> Pet class </font>
- It doesn’t exist. <font color='red'> Pet class </font> has no parent classes, so Python signals an error.

---

```py
new_cat = Cheshire("Pumpkin")
```

Neither Cheshire nor Cat defines an `__init__` constructor method
- so the grandaprent <font color=red> Pet class </font> will have it's `__init__` method called.
- That constructor method sets the instance variables name, hunger, boredom, and sounds.

---


## code implement

### The `__init__()` Method for a Child Class

Child Class:
- The `__init__()` method: takes in the information required to make a Car instance.
- The `super()` function:
    - a special function to call method from the parent class.
    - tells Python to call the __init__() method from Car, which gives an ElectricCar instance all the attributes defined in that method.

```py
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model self.year = year self.odometer_reading = 0

    def get_descriptive_name(self):
        long_name = f"{self.year} {self.manufacturer} {self.model}"
        return long_name.title()

    def read_odometer(self):
        print(f"This car has {self.odometer_reading} miles on it.")

    def update_odometer(self, mileage):
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage else:
            print("You can't roll back an odometer!")

    def increment_odometer(self, miles):
        self.odometer_reading += miles

class ElectricCar(Car):

    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.battery_size = 75

    # Overriding Methods from the Parent Class
    def fill_gas_tank(self):
        print("This car doesn't need a gas tank!")

my_tesla = ElectricCar('tesla', 'model s', 2019)
my_tesla.battery.describe_battery()

print(my_tesla.get_descriptive_name())

```


### Instances as Attributes

```py
class Battery:
    def __init__(self, battery_size=75):
        self.battery_size = battery_size

    def get_range(self):
        if self.battery_size == 75:
            range = 260
        elif self.battery_size == 100:
            range = 315
        print(f"This car can go about {range} miles on a full charge.")

    def describe_battery(self):
        print(f"This car has a {self.battery_size}-kWh battery.")


class ElectricCar(Car):
    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        # Instances as Attributes
        self.battery = Battery()

    # Overriding Methods from the Parent Class
    def fill_gas_tank(self):
        print("This car doesn't need a gas tank!")

my_tesla = ElectricCar('tesla', 'model s', 2019)

my_tesla.battery.describe_battery()
my_tesla.battery.get_range()

# 2019 Tesla Model S
# This car has a 75-kWh battery.
# This car can go about 260 miles on a full charge.
```

---

### `Overrid` 覆写 Methods


`Overrid` 覆写 Methods:

```py

class Parent(Object):

      def samename(self):
          statement1

class child(Parent):

      def samename(self):
          statement2
          # will only performer statement2
```


```py
keep the original Pet class.

make two `subclasses`, Dog and Cat.
- Dogs are always happy unless they are bored and hungry.
- Cats are happy only if they are fed and if their boredom level is in a narrow range and, even then, only with probability 1/2.

# the original Pet class again.
class Cat(Pet):
    sounds = ['Meow']

    def mood(self):
        if self.hunger > self.hunger_threshold:
            return "hungry"
        if self.boredom <2:
            return "grumpy; leave me alone"
        elif self.boredom > self.boredom_threshold:
            return "bored"
        elif randrange(2) == 0:
            return "randomly annoyed"
        else:
            return "happy"

class Dog(Pet):
    sounds = ['Woof', 'Ruff']
    def mood(self):
        if (self.hunger > self.hunger_threshold) and (self.boredom > self.boredom_threshold):
            return "bored and hungry"
        else:
            return "happy"

c1 = Cat("Fluffy")
d1 = Dog("Astro")

c1.boredom = 1
print(c1.mood())    # grumpy; leave me alone
c1.boredom = 3
for i in range(10):
    print(c1.mood())
print(d1.mood())

```

---

## 22.4. Invoke 调用 the Parent Class’s Method

`Invoke` 覆写 Methods:

```py

class Parent(Object):

      def samename(self):
          statement1

class child(Parent):

      def samename(self):
          Parent.samename(self)
          statement2
          # will performer both statement1&2
```


```py

class superclass():

    def __init__(self,x):
        self.x=x

    def method(self):
        print(1)


class childclass(superclass):

    def __init__(self,x,y=2):
        superclass.__init__(self,x)
        self.y=y

    def method(self):
        superclass.method(self)
        print(2)
```

Sometimes the parent class has a useful method,
- just need to execute a little extra code when running the subclass’s method.
- override the parent class’s method in the subclass’s method with the same name, or invoke the parent class’s method.

```py

# the original Pet class again.

class Pet():
    boredom_decrement = 4
    hunger_decrement = 6
    boredom_threshold = 5
    hunger_threshold = 10
    sounds = ['Mrrp']
    def __init__(self, name = "Kitty"):
        self.name = name
        self.hunger = randrange(self.hunger_threshold)
        self.boredom = randrange(self.boredom_threshold)
        self.sounds = self.sounds[:]  # copy the class attribute, so that when we make changes to it, we won't affect the other Pets in the class

    def feed(self):
        self.reduce_hunger()

// wanted the Dog subclass of Pet to say “Arf! Thanks!” when the feed method is called

class Dog(Pet):
    sounds = ['Woof', 'Ruff']

    def feed(self):
        Pet.feed(self)
        print("Arf! Thanks!")

# if the Pet.feed(self) line was deleted?
# no longer calling the parent Pet class's method in the Dog subclass's method definition, the class definition will override the parent method.
# the actions defined in the parent method feed will not happen, and only Arf! Thanks! will be printed.
# The string would print but d1 would not have its hunger reduced.

d1 = Dog("Astro")
d1.feed()
#
Arf! Thanks!
```

here’s a subclass that overrides feed() by invoking the the parent class’s feed() method;
- it then also executes an extra line of code. Note the somewhat inelegant way of invoking the parent class’ method.
- We explicitly refer to `Pet.feed` to get the method/function object. We invoke it with parentheses. However, since we are not invoking the method the normal way, with `<obj>.methodname`, we have to explicitly pass an instance as the first parameter.
- In this case, the variable `self` in `Dog.feed()` will be bound to an instance of Dog, and so just pass `self`: `Pet.feed(self)`.


This technique is very often used with the `__init__` method for a subclass.
- some extra instance variables are defined for the subclass.
- When you invoke the constructor, you pass all the regular parameters for the parent class, plus the extra ones for the subclass.
- The subclass’ `__init__` method then stores the extra parameters in instance variables and calls the parent class’ `__init__` method to store the common parameters in instance variables and do any other initialization that it normally does.

```py
class Pet():

    def hi(self):
        print(self.sounds[randrange(len(self.sounds))])
        self.reduce_boredom()

class Bird(Pet):
    sounds = ["chirp"]

    def __init__(self, name="Kitty", chirp_number=2):
        Pet.__init__(self, name) # call the parent class's constructor
        # basically, call the SUPER -- the parent version -- of the constructor, with all the parameters that it needs.
        self.chirp_number = chirp_number # now, also assign the new instance variable

    def hi(self):
        for i in range(self.chirp_number):
            print(self.sounds[randrange(len(self.sounds))])
            print(8)
        self.reduce_boredom()

b1 = Bird('tweety', 5)
b1.teach("Polly wanna cracker")
b1.hi()
# overwrite
Polly wanna cracker
8
Polly wanna cracker
8
chirp
8
Polly wanna cracker
8
chirp
8
```

---

## 继承和多态

### 继承
在Object Oriented Programming OOP程序设计中，当我们定义一个class的时候，可以从某个现有的class继承，新的class称为`子类 Subclass`，而被继承的class称为`基/父/超类（Base / Super class）`。

比如

```py

1.  编写了一个名为`Animal`的`class`
    - 有一个run()方法可以直接打印：

      class Animal(object):
          def run(self):
              print('Animal is running...')


2.  需要编写Dog和Cat类时，就可以直接从Animal类继承：
    - 于Dog来说，Animal就是它的父类
    - 对于Animal来说，Dog就是它的子类。Cat和Dog类似。

      class Dog(Animal):
          pass

      class Cat(Animal):
          pass
```

### 继承好处

```py

1.  最大的好处是子类获得了父类的全部功能。
    - 由于Animial实现了run()方法，因此，Dog和Cat作为它的子类，什么事也没干，就自动拥有了run()方法：

        dog = Dog()
        dog.run()

        cat = Cat()
        cat.run()
        # 结果
        Animal is running...
        Animal is running...


2.  可以对子类增加一些方法，比如Dog类：

        class Dog(Animal):

            def eat(self):
                print('Eating meat...')


3.  可以对代码做改进

        class Dog(Animal):

            def run(self):
                print('Dog is running...')

        class Cat(Animal):

            def run(self):
                print('Cat is running...')
        # 结果
        Dog is running...
        Cat is running...
```

当子类和父类都存在相同的run()方法时，
- 子类的run()覆盖了父类的run()
- 在代码运行的时候，总是会调用子类的run()。
- 这样，我们就获得了继承的另一个好处：`多态`。


### 多态

- 当定义一个`class`的时候，我们实际上就定义了一种数据类型。
- 我们定义的数据类型和Python自带的数据类型，比如str、list、dict没什么两样

```py
a = list()    # a是list类型
b = Animal()  # b是Animal类型
c = Dog()     # c是Dog类型

判断一个变量是否是某个类型可以用 isinstance() 判断：
>>> isinstance(a, list)
True
>>> isinstance(b, Animal)
True
>>> isinstance(c, Dog)
True
# a、b、c确实对应着list、Animal、Dog这3种类型。
>>> isinstance(c, Animal)
True
# c不仅仅是Dog，c还是Animal！
```

因为Dog是从Animal继承下来的，当创建一个Dog的实例c，c的数据类型是Dog，同时也是Animal，Dog本来就是Animal的一种

所以，在继承关系中，如果一个实例的数据类型是某个子类，那它的数据类型也可以被看做是父类。但是，反过来就不行：

```py
>>> b = Animal()
>>> isinstance(b, Dog)
False
Dog可以看成Animal，但Animal不可以看成Dog。
```


### 多态的好处


```py
编写一个函数接受一个Animal类型的变量：

def run_twice(animal):
    animal.run()
    animal.run()

# 当我们传入Animal的实例时，run_twice()就打印出：
>>> run_twice(Animal())
Animal is running...
Animal is running...

# 当我们传入Dog的实例时，run_twice()就打印出：
>>> run_twice(Dog())
Dog is running...
Dog is running...

# 当我们传入Cat的实例时，run_twice()就打印出：
>>> run_twice(Cat())
Cat is running...
Cat is running...

如果再定义一个Tortoise类型，也从Animal派生：

class Tortoise(Animal):
    def run(self):
        print('Tortoise is running slowly...')
# 调用run_twice()
>>> run_twice(Tortoise())
Tortoise is running slowly...
Tortoise is running slowly...

新增一个Animal的子类，不必对run_twice()做任何修改
- 任何依赖Animal作为参数的函数或者方法都可以不加修改地正常运行，原因就在于多态。
```


1.  多态的好处就是，当我们需要传入Dog、Cat、Tortoise……时，我们只需要接收Animal类型就可以了，因为Dog、Cat、Tortoise……都是Animal类型，然后，按照Animal类型进行操作即可。
    - 由于Animal类型有run()方法，因此，传入的任意类型，只要是Animal类或者子类，就会自动调用实际类型的run()方法，这就是多态的意思：

对于一个变量，只需要知道它是Animal类型，无需确切地知道它的子类型，就可以调用run()方法，
- 而具体调用的run()方法是作用在Animal、Dog、Cat还是Tortoise对象上，由运行时该对象的确切类型决定，这就是多态真正的威力：
- 调用方只管调用，不管细节，
- 而当新增一种Animal的子类时，只要确保run()方法编写正确，不用管原来的代码是如何调用的。这就是著名的`“开闭”原则`：

对扩展开放：允许新增Animal子类；

对修改封闭：不需要修改依赖 Animal类型的 run_twice() 等函数。

继承还可以一级一级地继承下来，就好比从爷爷到爸爸、再到儿子这样的关系。而任何类，最终都可以追溯到根类object，这些继承关系看上去就像一颗倒着的树。比如如下的继承树：

```py
                ┌───────────────┐
                │    object     │
                └───────────────┘
                        │
           ┌────────────┴────────────┐
           │                         │
           ▼                         ▼
    ┌─────────────┐           ┌─────────────┐
    │   Animal    │           │    Plant    │
    └─────────────┘           └─────────────┘
           │                         │
     ┌─────┴──────┐            ┌─────┴──────┐
     │            │            │            │
     ▼            ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│   Dog   │  │   Cat   │  │  Tree   │  │ Flower  │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

## 静态语言 vs 动态语言
对于静态语言（例如Java）来说，如果需要传入Animal类型，则传入的对象必须是Animal类型或者它的子类，否则，将无法调用run()方法。

对于Python这样的动态语言来说，则不一定需要传入Animal类型。我们只需要保证传入的对象有一个run()方法就可以了：

class Timer(object):
    def run(self):
        print('Start...')

这就是动态语言的“鸭子类型”，它并不要求严格的继承体系，一个对象只要“看起来像鸭子，走起路来像鸭子”，那它就可以被看做是鸭子。

Python的`“file-like object“`就是一种鸭子类型。对真正的文件对象，它有一个read()方法，返回其内容。但是，许多对象，只要有read()方法，都被视为“file-like object“。许多函数接收的参数就是“file-like object“，你不一定要传入真正的文件对象，完全可以传入任何实现了read()方法的对象。

---

继承可以把父类的所有功能都直接拿过来，这样就不必重零做起，子类只需要新增自己特有的方法，也可以把父类不适合的方法覆盖重写。

动态语言的鸭子类型特点决定了继承不像静态语言那样是必须的。

---

## Importing Classes

### Importing a Single Class

```py
car.py:

class Car:
"""A simple attempt to represent a car."""
```

1. Importing a Single Class

```py
from car import Car

my_new_car = Car('audi', 'a4', 2019)
my_new_car.odometer_reading = 23
my_new_car.read_odometer()
```

2. Storing Multiple Classes in a Module

```py
from car import ElectricCar
my_tesla = ElectricCar('tesla', 'model s', 2019)
```


3. Importing Multiple Classes from a Module

```py
from car import Car, ElectricCar
```

4. Importing an Entire Module

```py
import car
```

5. Using Aliases

```py
from car import ElectricCar as EC
```


.


# Key Concepts

## Horizontal vs. Vertical Scaling

* Vertical scaling
  * increasing the resoures of a specific node.
  * For example
  * add additional memory to a server to improve its ability to handle load changes.

* Horizontal scaling
  * increasing the number of nodes.
  * For example
  * add additional servers, thus decreasing the load on any one server.

Vertiacal scaling is generally easer than horizontal scaling, but it's limited.

---

## Load Balancer

Typically, some frontend parts of a scalable website will be thrown behind a load balancer. This allows a system to distribute the load evenly so that one server doesn't crash and take down the whole system.
- To do so, of course, you have to build out a network of cloned servers that all have essentially the same code and access to the same data.

---

## Database Denormalization and NoSQL

Joins in a relational database such as SQL can get very slow as the system grows bigger. For this reason, you would generally avoid them.

- **Denormalization** is one part of this.
  - adding redundant information into a database to speed up reads.
  - For example, imagine a database describing projects and tasks (in addition to the project table).
- Or, you can go with a NoSQL database.
  - A NoSQL database does not support joins and might structure data in a different way.
  - It is designed to scale better..



---



## Database Partitioning (Sharding)

- Sharding means splitting the data across multiple machines
- while ensuring you have a way of figuring out which data is on which machine.

A few common ways of partitioning include:

* **Vertical Partitioning**:
  * This is basically `partitioning by feature`.

* **Key-Based (or Hash-Based) Partitioning**:
  * This uses some part of the data to partition it.
  * A very simple way to do this is to allocate N servers and put the data on `mode(key, n)`.
  * One issue with this is that the number of servers you have is effectively fixed.
  * Adding additional servers means reallocating all the data -- a very expensive task.

* **Directory-Based Partitioning**:
  * In this scheme, you maintain a lookup table for where the data can be found.
  * This makes it relatively easy to add additional servers, but it comes with two major drawbacks.
    * First the lookup table can be a single point of failure.
    * Second, constantly access this table impacts performance.


---



## Caching

- An in-memory cache can deliver very rapid results.
- It is a simple key-value pairing and typically sits between your application layer and your data store.

---


## Asynchronous Processing & Queues

- Slow operations should ideally be done asynchronously.
- Otherwise, a user might get stuck waiting and waiting for a process to complete.

---

## Networking Metrics

* **Bandwidth**: This is the maximum amount of data that can be transferred in a unit of time. It is typically expressed in bits per seconds.
* **Throughput**: Whereas bandwidth is the maximum data that can be transferred in a unit of time, throughput is the actual amoutn of data that is transferred.
* **Latency**: This is how long it takes data to go from one end to the other. That is, it is the delay between the sender sending information (even a very small chunk of data) and the receiver receiving it.

---

## MapReduce

A MapReduce program is typically used to process large amounts of data.

* Map takes in some data and emits a pair
* Reduce takes a key and a set of associated values and reduces them in some way, emitting a new key and value.

MapReduce allows us to do a lot of processing in parallel, which makes processing huge amounts of data more scalable.


---

## Considerations

* **Failures**: Essentially any part of a system can fail. You'll need to plan for many or all of these failures.
* **Availability and Reliability**:
  * Availability is a function of `the percentage of time` the system is operatoinal.
  * Redliability is a function of `the probability` that the system is operational for a certain unit of time.
* **Read-heavy vs. Write-heavy**:
  * Whether an application will do a lot of reads or a lot of writes implacts the design.
  * If it's write-heavy, you could consider queuing up the writes (but think about potential failure here!).
  * If it's read-heavy, you might want to cache.
* **Security**:
  * Security threats can, of course, be devastating for a system.
  * Think about the tyupes of issues a system might face and design around thos.


  
