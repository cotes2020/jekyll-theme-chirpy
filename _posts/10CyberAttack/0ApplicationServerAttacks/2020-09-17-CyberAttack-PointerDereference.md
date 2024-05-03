---
title: Meow's CyberAttack - Application/Server Attacks - Pointer Dereference
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack]
tags: [CyberAttack]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Pointer Dereference](#meows-cyberattack---applicationserver-attacks---pointer-dereference)
	- [Pointer Dereference](#pointer-dereference)

book:
- S+ ch7

---

# Meow's CyberAttack - Application/Server Attacks - Pointer Dereference

---

## Pointer Dereference

Programming languages such as C, C++, and Pascal commonly use pointers, which simply store a reference to something. Some languages such as Java call them <font color=OrangeRed> references </font>.

Example:
- imagine an application has multiple modules.
- When a new customer starts an order, the application invokes the CustomerData module.
- This module needs to populate the city and state in a form after the user enters a zip code.
- How does the module get this array?
- One way is to pass the entire array to the module when invoking it. However, this consumes a lot of memory.
- The second method is to pass a reference to the data array, which is simply a pointer to it.
  - This consumes very little memory and is the preferred method.
  - This method uses a <font color=OrangeRed> pointer dereference </font>.

Dereferencing 非关联化 is the process of using the pointer to access the data array.
- Imagine the pointer is named `ptrZip` and the name of the full data array is named `arrZip`.
- The value within `ptrZip` is `arrZip`, which references the array.
- What is this thing that the pointer points to? There isn’t a standard name, but some developers refer to it as a pointee.

What’s the point?
- A failed dereference operation can cause an application to crash.
- In some programming languages, it can subtly corrupt memory, which can be even worse than a crash.
- The subtle, random changes result in the application using incorrect data.
- This can often be difficult to troubleshoot and correct.

The cause of a <font color=OrangeRed> failed dereference operation </font> is a <font color=LightSlateBlue> pointer that references a nonexistent pointee </font>.
- Admittedly, this programming error would be quickly discovered because the CustomerData module wouldn’t correctly populate the city and state.
- However, other pointer dereferencing problems aren’t so easy to discover.
