---
title: Java - Note
date: 2020-09-14 11:11:11 -0400
description:
categories: [04CodeNote, JavaNote]
tags: [Java]
toc: true
---

- [Interfaces and Abstract Classes](#interfaces-and-abstract-classes)
  - [Commonalities in Code](#commonalities-in-code)
  - [Developing an Interface](#developing-an-interface)
  - [Interface: Utility and Flexibility](#interface-utility-and-flexibility)
- [abstract keyword in java](#abstract-keyword-in-java)
  - [Capture commonality in Abstract Base Class](#capture-commonality-in-abstract-base-class)

---



# Interfaces and Abstract Classes




## Commonalities in Code


```java

Method MarkovRunner.runMarkovN()

    FileResource fr = new FileResource("data/confucius.txt");
	String st = fr.asString();
	st = st.replace('\n', ' ');

public void runMarkovN() {
    // MarkovZero markov = new MarkovZero();
    // MarkovOne markov = new MarkovOne();
    // MarkovFour markov = new MarkovFour();
	MarkovModel markov = new MarkovModel();
    markov.setModel(8);
    markov.setTraining(st);
    markov.setRandom(365);

	for(int k=0; k < 2; k++){
        String text = markov.getRandomText(500);
    	printOut(text);
    }
}

public class MarkovModel {
    private String myText;
    private Random myRandom;
    private int n;
    public MarkovModel(){}

    public void setRandom(int seed){}

    public void setTraining(String s){}
    public void setModel(int N){}
    public String getRandomText(int numChars) {}

    public ArrayList<String> getFollows(String key) {}
}

```

## Developing an Interface

```java

public interface IMarkovModel {
    public void setTraining(String text);
    public String getRandomText(int numChars);
}

public class MarkovOne implements IMarkovModel{
    private String myText;
	private Random myRandom;
    public MarkovOne() {}
    public void setRandom(int seed){}
}
	
public class MarkovTwo implements IMarkovModel{}         // MarkovTwo is <IMarkovModel> objecy

public class MarkovFour implements IMarkovModel{}


```

## Interface: Utility and Flexibility


```java
public void runModel(IMarkovModel markov, String text, int size){
    markov.setTraining(text);
    System.out.println("running with "+markov);
    for(int k=0; k < 3; k++){
        String st = markov.getRandomText(size);
        printOut(st);
    }
}

MarkovZero mz = new MarkovZero();
runModel(mz,text,800);

MarkovTwo m2 = new MarkovTwo();
runModel(m2,text,800);
```


---

# abstract keyword in java

abstract is a non-access modifier in java applicable for classes, methods but not variables. 
It is used to achieve abstraction which is one of the pillar of Object Oriented Programming(OOP).

1. Abstract classes
   - having **partial implementation** 
   - not all methods present in the class have method definition
   - Due to partial implementation, cannot instantiate abstract classes.
   - Any subclass of an abstract class must either implement all of the abstract methods in the super-class, or be declared abstract itself.
   - Some of the predefined classes in java are abstract. They depends on their sub-classes to provide complete implementation. 
   - For example, java.lang.Number is a abstract class. 

```java
abstract class class-name{
    //body of class
}
```

2. Abstract methods
   - Sometimes, require just method declaration in super-classes.
   - This can be achieve by specifying the abstract type modifier. 
   - These methods are sometimes referred to as subclasser responsibility because they have no implementation specified in the super-class. 
   - Thus, a subclass must override them to provide method definition. 

```java
abstract type method-name(parameter-list);
```

3. demo

```java
abstract class A  { 
    // abstract with method it has no body 
    abstract void m1(); 
      
    // concrete methods are still allowed in abstract classes 
    void m2() { 
        System.out.println("This is a concrete method."); 
    } 
} 
  
// concrete class B 
class B extends A { 
    // class B must override m1() method 
    // otherwise, compile-time exception will be thrown 
    void m1() { 
        System.out.println("B's implementation of m2."); 
    }
} 

public class AbstractDemo  { 
    public static void main(String args[])  { 
        B b = new B(); 
        b.m1(); 
        b.m2(); 
    } 
} 
// Output:
// B's implementation of m2.
// This is a concrete method.
```



---

## Capture commonality in Abstract Base Class

`AbstractMarkovModel`

```java
// Class marked as abstract
public abstract class AbstractMarkovModel implements IMarkovModel {

    protected String myText;     // Shared state is protected, not private
    protected Random myRandom;

    public AbstractMarkovModel() {
        myRandom = new Random();
    }
    
    public void setTraining(String text) {
        myText = text;
    }

    abstract public String getRandomText(int numChars);

    protected ArrayList<String> getFollows(String key){
        // code not shown
    }
}



public class MarkovModel extends AbstractMarkovModel {

    private int myOrder;

    public MarkovModel(int order) {
        myOrder = order;
    }

    public String getRandomText(int length) {
        StringBuffer sb = new StringBuffer();
        int index = myRandom.nextInt(myText.length() - myOrder);
        String current = myText.substring(index, index + myOrder);
        sb.append(current);
        
        for(int k=0; k < length-myOrder; k++){
            ArrayList<String> follows = getFollows(current);
        }
    }
}



```











.