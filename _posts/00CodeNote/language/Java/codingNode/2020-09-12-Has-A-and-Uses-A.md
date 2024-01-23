---
title: Java - Has-A and Uses-A Relationship
date: 2020-09-12 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---

# Java - Has-A and Uses-A Relationship

[toc]

---

In Java, reuse code using an `Is-A relationship` or `Has-A relationship`.
- Is-A relationship is also known as inheritance
- Has-A relationship is also known as composition in Java.
- both used for code reusability in Java.


## Is-A Relationship in Java
- In Java, Is-A relationship depends on inheritance.
  - Further inheritance is of two types, `class inheritance` and `interface inheritance`.
- **inheritance is unidirectional in nature**.
  - a Potato is a vegetable, a Bus is a vehicle.
  - a house is a building.
  - But not all buildings are houses.
- determine an Is-A relationship in Java.
  - `extends` or `implement` keyword in the class declaration in Java
  - then the specific class is said to be following the Is-A relationship.


## Has-A Relationship in Java
- In Java, Has-A relationship is also known as composition.
- Has-A relationship: an instance of one class has a reference to an instance of another class or an other instance of the same class.
  - For example, a car has an engine, a dog has a tail and so on.
- In Java, there is no such keyword that implements a Has-A relationship. But we mostly use new keywords to implement a Has-A relationship in Java.


![IS-A and HAS-A relationship](https://github.com/ocholuo/ocholuo.github.io/tree/master/assets/img/Javaimg/IS-A-and-HAS-A-relationship.jpg)

![IS-A and HAS-A relationship](/assets/img/Javaimg/IS-A-and-HAS-A-relationship.jpg)


## code example

```java
package relationsdemo;

public class Bike  {
    private String color;
    private int maxSpeed;
    public void bikeInfo()  {
        System.out.println("Bike Color= "+color + " Max Speed= " + maxSpeed);
    }
    public void setColor(String color)  {
        this.color = color;
    }
    public void setMaxSpeed(int maxSpeed)  {
        this.maxSpeed = maxSpeed;
    }
}

public class Engine  {
    public void start() {
        System.out.println("Started:");
    }
    public void stop() {
        System.out.println("Stopped:");
    }
}


// ====================================================================


public class Pulsar extends Bike {
// Pulsar is a type of bike that extends the Bike class that shows that Pulsar is a Bike.
// All the methods like setColor( ), bikeInfo( ), setMaxSpeed( ) are used because of the Is-A relationship of the Pulsar class with the Bike class.

    public void PulsarStartDemo() {
        Engine PulsarEngine = new Engine();
        // Pulsar also uses an Engine's method, stop, using composition.
        PulsarEngine.stop();
    }
}


// ====================================================================

public class Demo  {
    public static void main(String[] args)  {
        Pulsar myPulsar = new Pulsar();
        myPulsar.setColor("BLACK");
        myPulsar.setMaxSpeed(136);
        myPulsar.bikeInfo();
        myPulsar.PulsarStartDemo();
    }
}


```
