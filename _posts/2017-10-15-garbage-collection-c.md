---
title: 'Garbage Collection in C#'
date: 2017-10-15T17:03:22+02:00
author: Wolfgang Ofner
categories: [Programming]
tags: ['C#']
---
The garbage collection is a great feature which makes the life of a programmer was easier. It releases unused variables and doing so frees memory on the heap.  If you know C++ or C, you know what a pain it can be to release all unused object and variables and how easily you can forget one which will lead to a memory leak. A memory leak is caused by a variable which has a value but no reference. This means that the variable can&#8217;t be deleted and remains in the memory until the computer is restarted.

In C# the garbage collection works fully automatically and the programmer doesn&#8217;t have to worry about it. But how does it work?

## Garbage Collection and variable generations

When a new variable is generated it will be placed in generation 0. The garbage collector is invoked after the generation 0 filled up. The unused variable will be deleted and the memory will be released from the heap. Variables which survive this process will be pushed into generation 1. If generation 1 is full the process will be repeated. C# knows the generations 0, 1, 2. Lower generations will be checked more frequently.

The programmer can&#8217;t start the garbage collection process. Going through all variables is an expensive task and comes with the cost of lower performance.

### Further reading

In this short post, I only covered the basics on how the garbage collection works. For more detailed information see the provided links.

<a href="https://docs.microsoft.com/en-us/dotnet/standard/garbage-collection/" target="_blank" rel="noopener">https://docs.microsoft.com/en-us/dotnet/standard/garbage-collection/</a>

<a href="http://aspalliance.com/828" target="_blank" rel="noopener">http://aspalliance.com/828</a>

<a href="https://www.codeproject.com/Articles/1060/Garbage-Collection-in-NET" target="_blank" rel="noopener">https://www.codeproject.com/Articles/1060/Garbage-Collection-in-NET</a>