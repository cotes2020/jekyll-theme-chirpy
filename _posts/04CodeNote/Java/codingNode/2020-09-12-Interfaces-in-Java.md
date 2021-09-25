---
title: Java - Interfaces in Java
date: 2020-09-12 11:11:11 -0400
description:
categories: [04CodeNote, JavaNote]
tags: [Java]
toc: true
---

# Java - Interfaces in Java


```java
interface In1 { 
	final int a = 10;    // public, static and final 
	void display();      // public and abstract 
} 

// A class that implements the interface. 
class TestClass implements In1 {        // Implementing the capabilities of interface. 
	public void display() { 
		System.out.println("Geek"); 
	} 
	// Driver Code 
	public static void main (String[] args) { 
		TestClass t = new TestClass(); 
		t.display(); 
		System.out.println(a); 
	} 
} 
```

---

## basic

To declare an interface, use interface keyword. It is used to provide total abstraction. That means all the methods in an interface are declared with an empty body and are public and all fields are public, static and final by default. 

A class that implements an interface must implement all the methods declared in the interface.

Why use interface
- to achieve total abstraction.
- to achieve multiple inheritance .
  - java does not support multiple inheritance in case of class, 
- to achieve loose coupling.
- to implement abstraction. 
  - abstract classes may contain non-final variables, 
  - whereas variables in interface are final, public and static.


New features added in interfaces in JDK 8
1. Prior to JDK 8, interface could not define implementation. 
   - can now add default implementation for interface methods. 
   - This default implementation has special use and does not affect the intention behind interfaces.
   - Suppose we need to add a new function in an existing interface. Obviously the old code will not work as the classes have not implemented those new functions. So with the help of default implementation, we will give a default body for the newly added functions. Then the old codes will still work.

```java
// interfaces can have methods from JDK 1.8 onwards 
interface In1 { 
	final int a = 10; 
	default void display() { 
		System.out.println("hello"); 
	} 
} 

// A class that implements the interface. 
class TestClass implements In1 { 
	// Driver Code 
	public static void main (String[] args) { 
		TestClass t = new TestClass(); 
		t.display(); 
	} 
} 
// Output :
// hello
```

1. can now define static methods in interfaces which can be called independently without an object. Note: these methods are not inherited.

```java
// interfaces can have methods from JDK 1.8 onwards 
interface In1 { 
    final int a = 10; 
    static void display() { 
        System.out.println("hello"); 
    } 
} 
  
// A class that implements the interface. 
class TestClass implements In1 { 
    // Driver Code 
    public static void main (String[] args) { 
        In1.display(); 
    } 
} 
// Output :
// hello
```

---

## code


1. real-world example

```java
import java.io.*; 

interface Vehicle {  
	void changeGear(int a);     // all are the abstract methods. 
	void speedUp(int a); 
	void applyBrakes(int a); 
} 

class Bicycle implements Vehicle{ 
	int speed; 
	int gear; 
	
	// to change gear 
	@Override
	public void changeGear(int newGear){ 
		gear = newGear; 
	} 
	// to increase speed 
	@Override
	public void speedUp(int increment){ 
		speed = speed + increment; 
	} 
	// to decrease speed 
	@Override
	public void applyBrakes(int decrement){ 
		speed = speed - decrement; 
	} 
    
	public void printStates() { 
		System.out.println("speed: " + speed + " gear: " + gear); 
	} 
} 

class Bike implements Vehicle { 
	int speed; 
	int gear; 
	
	// to change gear 
	@Override
	public void changeGear(int newGear){ 
		gear = newGear; 
	} 
	// to increase speed 
	@Override
	public void speedUp(int increment){ 
		speed = speed + increment; 
	} 
	// to decrease speed 
	@Override
	public void applyBrakes(int decrement){ 
		speed = speed - decrement; 
    } 
    
	public void printStates() { 
		System.out.println("speed: " + speed + " gear: " + gear); 
	} 
} 


class GFG { 
	public static void main (String[] args) { 
		Bicycle bicycle = new Bicycle();   // creating an inatance of Bicycle doing some operations 
		bicycle.changeGear(2); 
		bicycle.speedUp(3); 
		bicycle.applyBrakes(1); 
		System.out.println("Bicycle present state :"); 
		bicycle.printStates(); 
		
		
		Bike bike = new Bike();   // creating instance of the bike. 
		bike.changeGear(1); 
		bike.speedUp(4); 
		bike.applyBrakes(3); 
		System.out.println("Bike present state :"); 
		bike.printStates(); 
	} 
} 
```








.
