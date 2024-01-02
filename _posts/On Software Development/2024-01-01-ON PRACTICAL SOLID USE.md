# On practical SOLID use

In this post I'll go about my personal interpretation of the SOLID principle. What I understand and take from it, and
how I believe it can be applied practically.

Numerous discussions have been made about both the significance of each of these principles and their respective
utility. I stand by that these principles make sense in specific scenarios and within these situations they are both
useful and efficient.

There are dozens of posts and web pages explaining the acronym,
with a few showing examples and in-depth explanations of each section.
My motivation with this post is to provide a simple interpretation that can be taken into account with minimal effort.
This was made with the intention of helping both myself and the *improbable* r eader with keeping these principles in
mind during all steps taken in developing software.

## The SOLID principles

For memoization purposes, the acronym SOLID stands for:

- Single-responsibilityThe SRP defines the reason for which an entity exists principle (or ***SRP***): "There should
  never be more than one reason for a class to change."
- Open-closed principle (or ***OCP***): "Entities... should be open for extension, but closed for modification."
- Liskov substitution principle (or ***LSP***): "Functions that use pointers or references to base classes must be able
  to use objects of derived classes without knowing it."
- Interface segregation principle (or ***ISP***): "Clients should not be forced to depend upon interfaces that they do
  not use."
- Dependency inversion principle (or ***DIP***): "Depend upon abstractions, \[not\] concretes."

## My interpretation

For the useful section of this post, my humble interpretation and exemplification of these principles:

### SRP

The SRP defines the reason for which an entity exists

- This entity exists solely to perform, maintain, create, or other verb with respect to that reason.
- If the entity does anything that affects anything else, that action is part of another responsibility that does not
  concern that entity.

### OCP

The OCP defines the way in which an entity can be modified

- This entity can be modified by adding new functionality, but not by changing existing functionality.
- If the entity is modified by changing existing functionality, then it is not following the OCP.
  Extension means adding new functionality, modification means changing existing functionality.

| Module                                 | Open | Closed |
|----------------------------------------|------|--------|
| Available for use by other modules     | Yes  | Yes    |
| Source code available for modification | No   | Yes    |
| Available for extension                | Yes  | No     |

### LSP

The LSP defines the way in which an entity can be used

- An entity can be used in place of any other entity that it inherits from.
- If the entity cannot be used in place of any other entity that it inherits from, then it is not following the LSP.
  This means that the entity is not a subtype of the other entity.

An example of this is the `Rectangle` and `Square` classes.
A `Square` can be used in place of a `Rectangle`, but a `Rectangle` cannot be used in place of a `Square`.

### ISP

The ISP describes the design of an entity relative to its dependencies

- An entity should not depend on any functionality that it does not use.
- Clients should not be forced to depend on interfaces they do not use.

For example, if you have an interface with ten methods and a class only needs three of them,
adhering to the ISP would suggest breaking down the interface into smaller, more specific interfaces,
so that the class can implement only what it requires.

### DIP

The DIP, or Dependency Inversion Principle, deals with the direction of dependency.

- High-level modules should not depend on low-level modules. Both should depend on abstractions.
- Abstractions should not depend on details. Details should depend on abstractions.

This principle encourages the use of abstractions to decouple
high-level modules from low-level modules, promoting flexibility and easier maintenance.

Instead of a high-level module directly depending on the implementation details of a low-level module, both
should depend on an interface or abstract class. This allows for easier substitution of components without
affecting the overall system.

## References:

1. [Martin, Robert C.](https://en.wikipedia.org/wiki/Robert_C._Martin "Robert C. Martin")["Principles Of OOD"](http://butunclebob.com/ArticleS.UncleBob.PrinciplesOfOod) - 2003.
