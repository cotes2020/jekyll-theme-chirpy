---
title: How I learned programming
date: 2017-10-15T23:07:59+02:00
author: Wolfgang Ofner
categories: [Programming]
tags: [.Net, 'C#', learning]
---
I learned programming in university in Austria. Before going to university, I only knew that there is the for loop and the if condition. This was all I knew. Unsurprisingly you have many theoretical classes like requirements engineering, software design and introduction to computer systems in your first semesters. This classes also helped to understand the bigger picture of programming but in this post, I will focus on the practical parts.

## The journey begins

In my first semester, I had a class called logic and logic programming. The logic programming was programming with <a href="https://en.wikipedia.org/wiki/Prolog" target="_blank" rel="noopener noreferrer">Prolog</a>. If you have never heard about this programming language, be happy. In Prolog, everything works with lists and recursion. This is probably the worst language to start learning programming since recursion is not that easy to understand in the beginning and so you can&#8217;t understand what&#8217;s going on during debugging either. For me, there was a lot of magic happening and I couldn&#8217;t tell why either.

Also, our assignments were not beginner friendly at all. In class we learned that with a logical conclusion you get to the result. For example, a BMW Z4 is faster than a Ford Focus and the Starship Enterprise is faster than the BMW. Therefore, the Enterprise is faster than the Ford. So far so good. The assignment then was that user enters the plan of a building and how they are connected and a starting point. The program should calculate all possible exits and store them in a list. Every exit could be selected only once. Maybe the assignment was a bit different but it was something like that.

This class is at least in the top 3 of the biggest bullshit classes of 5 years studying. I think it&#8217;s a terrible way to start learning programming and it made me almost quit. Honestly no idea how I passed. In the second semester I had C# instead and from this moment on I loved programming.

## C#: The start of a love story

In this chapter and in the next part I will describe what I learned in theory in class and what assignments I had. I will also show my implementations of these assignments. Bear in mind that I would do them, with the knowledge of today, completely different. I will link my GitHub repository.

As already mentioned, in my second semester I had programming in C#. I had one class a week for seven hours (from 10 to 5). During this class, we had a big theory block and then lab. In the first theory block, I learned that different kinds of loops and the if and switch condition exists. I also learned that it&#8217;s possible to read user input in the console and print out something in different colors to the console. The last thing I learned was that a data storage called array exists and that I can save different values there.

After these around 150 slides, I had my first lab. The assignment for this lab was to program a console program which paints the sinus curve and lets the user move to the left and right and always displaying the cursor at the current location of the curve. I had no idea how to do that at all and the only answer from our instructor was &#8220;You have to figure that out by yourself&#8221;. Since this lab didn&#8217;t give any grades I didn&#8217;t worry too much and looked forward to the first of five assignments.

## My first assignment: The Matrix Calculator

The first assignment was to program a console application which lets the user enter two-dimensional matrices. Each matrix had a name which had to be unique. The user could enter up to 10 different matrices. Every matrix started with opening square brackets and ended with closed square brackets. The values were separated by a comma and a row was ended by a semicolon. An example for a valid matrix is A=\[1, 2, 3; 7, 8, 9; ]. An invalid matrix would be A = [1, 2, 3; 7, 8;\] (The name is already taken and the second row has fewer values than the first one).

After the user finished his input, he could perform different calculations with the entered matrices. For example, A + B should add the previously entered matrices A and B and then display the result. The program must not crash at any point when the user entered invalid inputs. This was fastidiously tested which in my eyes is not useful at all because it&#8217;s way more important to understand what you are doing than testing every single input (at least when you start learning how to program).

### The implementation

The program had to be handed in within two weeks and it took me around 60 hours to finish it. Within this two weeks, I spent a couple nights in front of my computer and tried to figure out how to solve this problem. For the data input, I used a three-dimensional array. Up to today, this is the only time I used one. At that time I didn&#8217;t know better than using the first dimension for the name and the second and third dimension for the data. I also wasted a lot of time with testing every single input for all the possible wrong user inputs. Programming the calculations was done quickly because in my first semester I already did this in Matlab. So I could basically copy that. Otherwise, the assignment would have been even a couple hours longer.

Another useless guideline from my school was that we had to use Style Cop with all comment settings activated. Handing in an assignment with a single Style Cop warning would lead to zero points. Comments can be good and necessary but in that excess, we had to do it and was counterproductive and lowered the readability and was a huge pain in the ass.

Today I know that objective programming and lists exist and therefore my solution today would look completely different than back in the days. If you are interested in my solution you can find it on <a href="https://github.com/WolfgangOfner/UNI-MatrixCalculator" target="_blank" rel="noopener noreferrer">GitHub</a>.

## The second assignment: Flight Route Calculation

Prior to the next assignment, I had two theory classes. In this classes I learned that it’s possible to overload methods, to catch exceptions with a try catch block and to start a program with command line arguments. I also heard the first time about object-oriented programming which allows you to create classes in your program. To be honest, I didn’t understand the concept fully at this time (which you can see in my assignment because I only had one class which had over 450 lines of code). Another great feature I learned was List. Finally, I could store data dynamically at run time.

Equipped with this new knowledge I got my second assignment. The program should take flight routes from the user and then calculate all possible flight routes between two destinations. This program was similar to the flight plan calculator in Prolog which I described earlier.

### My implementation

After figuring out how recursion works, this assignment wasn’t too hard anymore. It was again a lot of input testing and printing the menus. Additionally, to the user input, it is possible to start the program with some command line arguments. I can’t remember the syntax of the arguments but in the program, it was again just input testing.

An example for the user input is:

  * Paris-London
  * Paris-Vienna
  * Vienna-Zurich
  * London-Zurich

After the routes are entered the user can enter his query, for example, Paris-Zurich. Now the program searches all possible routes from Paris to Zurich and Prints:

  * Paris-Vienna-Zurich
  * Paris-London-Zurich

The tricky part here is to know if the route was already found and when to stop. Otherwise, you will end in an infinite loop.

You can find my original solution on <a href="https://github.com/WolfgangOfner/Uni-FlightRouteCalculator" target="_blank" rel="noopener noreferrer">GitHub</a>. Again, today my solution would look completely different. Back then I didn’t understand object-oriented programming and had never heard of clean code or TDD.

This was the first part of how I learned programming. In the next part, you can read about the second part of the semester.

Next: <a href="/how-i-learned-programming-part-2/" target="_blank" rel="noopener noreferrer">Part 2</a>