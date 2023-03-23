---
title: Java - POJO and JavaBeansrview Questiones
date: 2020-09-12 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---

- [Java - POJO and JavaBeans](#java---pojo-and-javabeans)
  - [Plain Old Java Objects - POJO](#plain-old-java-objects---pojo)
    - [Reflection with a POJO](#reflection-with-a-pojo)
  - [JavaBeans](#javabeans)
    - [EmployeePojo as a JavaBean](#employeepojo-as-a-javabean)
    - [Reflection with a JavaBean](#reflection-with-a-javabean)

---


# Java - POJO and JavaBeans

---

## Plain Old Java Objects - POJO
- a straightforward type with no references to any particular frameworks.
- A POJO has no naming convention for our properties and methods.

```java
a basic employee POJO. three properties; first name, last name, and start date:

public class EmployeePojo {

    public String firstName;
    public String lastName;
    private LocalDate startDate;

    public EmployeePojo(String firstName, String lastName, LocalDate startDate) {
        this.firstName = firstName;
        this.lastName = lastName;
        this.startDate = startDate;
    }

    public String name() {
        return this.firstName + " " + this.lastName;
    }

    public LocalDate getStart() {
        return this.startDate;
    }
}
```

- This class can be used by any Java program as it's not tied to any framework.
- But, we aren't following any real convention for constructing, accessing, or modifying the class's state.

lack of convention causes two problems:
- increases the learning curve to understand how to use it.
- may limit a framework's ability to favor convention over configuration, understand how to use the class, and augment its functionality.

---

### Reflection with a POJO

add the commons-beanutils dependency to project:

```java
<dependency>
    <groupId>commons-beanutils</groupId>
    <artifactId>commons-beanutils</artifactId>
    <version>1.9.4</version>
</dependency>
```

And now, let's inspect the properties of our POJO:

```java
List<String> propertyNames = PropertyUtils.getPropertyDescriptors(EmployeePojo.class).stream()
    .map(PropertyDescriptor::getDisplayName)
    .collect(Collectors.toList());
```

If we were to print out propertyNames to the console, we'd only see:

`[start]`

Here, we see that we only get start as a property of the class. PropertyUtils failed to find the other two.

We'd see the same kind of outcome were we to use other libraries like Jackson to process EmployeePojo.

Ideally, we'd see all our properties: firstName, lastName, and startDate. And the good news is that many Java libraries support by default something called the JavaBean naming convention.

---

## JavaBeans
- A JavaBean is still a POJO but introduces a strict set of rules around how we implement it:
- Access levels – our properties are private and we expose getters and setters
- Method names – our getters and setters follow the getX and setX convention (in the case of a boolean, isX can be used for a getter)
- Default Constructor – a no-argument constructor must be present so an instance can be created without providing arguments, for example during deserialization
- Serializable – implementing the Serializable interface allows us to store the state

### EmployeePojo as a JavaBean

```java
public class EmployeeBean implements Serializable {

    private static final long serialVersionUID = -3760445487636086034L;
    private String firstName;
    private String lastName;
    private LocalDate startDate;

    public EmployeeBean() {
    }

    public EmployeeBean(String firstName, String lastName, LocalDate startDate) {
        this.firstName = firstName;
        this.lastName = lastName;
        this.startDate = startDate;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    //  additional getters/setters
}
```

---

### Reflection with a JavaBean
When we inspect our bean with reflection, now we get the full list of the properties:

`[firstName, lastName, startDate]`


1. Tradeoffs When Using JavaBeans
- When we use JavaBeans we should also be mindful of some potential disadvantages:
- Mutability – JavaBeans are mutable due to their setter methods – this could lead to concurrency or consistency issues
- Boilerplate – we must introduce getters for all properties and setters for most, much of this might be unnecessary
- Zero-argument Constructor – we often need arguments in our constructors to ensure the object gets instantiated in a valid state, but the JavaBean standard requires us to provide a zero-argument constructor
- Given these tradeoffs, frameworks have also adapted to other bean conventions over the years.
