---
title: "JavaScript「创建型」设计模式"
date: 2018-10-23
permalink: /2018-10-23-singleton-pattern/
---
> 创建型模式：创建对象的模式，抽象了实例化的过程


## 单例模式


### 什么是单例模式？


单例模式定义：保证一个类仅有一个实例，并提供访问此实例的全局访问点。


### 单例模式用途


如果一个类负责连接数据库的线程池、日志记录逻辑等等，**此时需要单例模式来保证对象不被重复创建，以达到降低开销的目的。**


## 代码实现


需要指明的是，**以下实现的单例模式均为“惰性单例”：只有在用户需要的时候才会创建对象实例。**


### python3 实现


```python
class Singleton:
  # 将实例作为静态变量
  __instance = None
  @staticmethod
  def get_instance():
    if Singleton.__instance == None:
      # 如果没有初始化实例，则调用初始化函数
      # 为Singleton生成 instance 实例
      Singleton()
    return Singleton.__instance
  def __init__(self):
    if Singleton.__instance != None:
      raise Exception("请通过get_instance()获得实例")
    else:
      # 为Singleton生成 instance 实例
      Singleton.__instance = self
if __name__ == "__main__":
  s1 = Singleton.get_instance()
  s2 = Singleton.get_instance()
  # 查看内存地址是否相同
  print(id(s1) == id(s2))

```


### javascript 实现


```javascript
const Singleton = function() {};
Singleton.getInstance = (function() {
    // 由于es6没有静态类型,故闭包: 函数外部无法访问 instance
    let instance = null;
    return function() {
        // 检查是否存在实例
        if (!instance) {
            instance = new Singleton();
        }
        return instance;
    };
})();
let s1 = Singleton.getInstance();
let s2 = Singleton.getInstance();
console.log(s1 === s2);
```


## 工厂模式


### 什么是工厂模式？


工厂方法模式的实质是“定义一个创建对象的接口，但让实现这个接口的类来决定实例化哪个类。工厂方法让类的实例化推迟到子类中进行。”


简单来说：_就是把_ _`new`_ _对象的操作包裹一层，对外提供一个可以根据不同参数创建不同对象的函数_。


### 工厂模式的优缺点


优点显而易见，可以隐藏原始类，方便之后的代码迁移。调用者只需要记住类的代名词即可。


由于多了层封装，会造成类的数目过多，系统复杂度增加。


### ES6 实现


调用者通过向工厂类传递参数，来获取对应的实体。在这个过程中，具体实体类的创建过程，由工厂类全权负责。


```javascript
/**
 * 实体类：Dog、Cat
 */
class Dog {
    run() {
        console.log("狗");
    }
}
class Cat {
    run() {
        console.log("猫");
    }
}
/**
 * 工厂类：Animal
 */
class Animal {
    constructor(name) {
        name = name.toLocaleLowerCase();
        switch (name) {
            case "dog":
                return new Dog();
            case "cat":
                return new Cat();
            default:
                throw TypeError("class name wrong");
        }
    }
}
/**
 * 以下是测试代码
 */
const cat = new Animal("cat");
cat.run();
const dog = new Animal("dog");
dog.run();

```


## 抽象工厂模式


抽象工厂模式就是：围绕一个超级工厂类，创建其他工厂类；再围绕工厂类，创建实体类。


相较于传统的工厂模式，它多出了一个**超级工厂类**。


### 什么是抽象工厂模式？


抽象工厂模式就是：围绕一个超级工厂类，创建其他工厂类；再围绕工厂类，创建实体类。


相较于传统的工厂模式，它多出了一个**超级工厂类**。


它的优缺点与工厂模式类似，这里不再冗赘它的优缺点，下面直接谈一下实现吧。


### 如何实现抽象工厂模式？


为了让目标更清晰，就实现下面的示意图：


![006tNbRwgy1gamtor9r2kj30en07zdgq.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2018-10-23-singleton-pattern/006tNbRwgy1gamtor9r2kj30en07zdgq.jpg)


### 准备实体类


按照之前的做法，这里我们实现几个实体类：Cat 和 Dog 一组、Male 和 Female 一组。


```javascript
class Dog {
    run() {
        console.log("狗");
    }
}
class Cat {
    run() {
        console.log("猫");
    }
}
/*************************************************/
class Male {
    run() {
        console.log("男性");
    }
}
class Female {
    run() {
        console.log("女性");
    }
}

```


### 准备工厂类


假设 Cat 和 Dog，属于 Animal 工厂的产品；Male 和 Female 属于 Person 工厂的产品。所以需要实现 2 个工厂类：Animal 和 Person。


由于工厂类上面还有个超级工厂，为了方便工厂类生产实体，工厂类应该提供生产实体的方法接口。


为了更好的约束工厂类的实现，先实现一个抽象工厂类：


```typescript
class AbstractFactory {
    getPerson() {
        throw new Error("子类请实现接口");
    }
    getAnimal() {
        throw new Error("子类请实现接口");
    }
}

```


接下来，Animal 和 Dog 实现抽象工厂类(AbstractFactory)：


```typescript
class PersonFactory extends AbstractFactory {
    getPerson(person) {
        person = person.toLocaleLowerCase();
        switch (person) {
            case "male":
                return new Male();
            case "female":
                return new Female();
            default:
                break;
        }
    }
    getAnimal() {
        return null;
    }
}
class AnimalFactory extends AbstractFactory {
    getPerson() {
        return null;
    }
    getAnimal(animal) {
        animal = animal.toLocaleLowerCase();
        switch (animal) {
            case "cat":
                return new Cat();
            case "dog":
                return new Dog();
            default:
                break;
        }
    }
}

```


### 实现“超级工厂”


超级工厂的实现没什么困难，如下所示：


```typescript
class Factory {
    constructor(choice) {
        choice = choice.toLocaleLowerCase();
        switch (choice) {
            case "person":
                return new PersonFactory();
            case "animal":
                return new AnimalFactory();
            default:
                break;
        }
    }
}
```


### 看看怎么使用超级工厂


实现了那么多，还是要看用例才能更好理解“超级工厂”的用法和设计理念：


```javascript
/**
 * 以下是测试代码
 */
// 创建person工厂
const personFactory = new Factory("person");
// 从person工厂中创建 male 和 female 实体
const male = personFactory.getPerson("male"),
    female = personFactory.getPerson("female");
// 输出测试
male.run();
female.run();
// 创建animal工厂
const animalFactory = new Factory("animal");
// 从animal工厂中创建 dog 和 cat 实体
const dog = animalFactory.getAnimal("dog"),
    cat = animalFactory.getAnimal("cat");
// 输出测试
dog.run();
cat.run();
```


