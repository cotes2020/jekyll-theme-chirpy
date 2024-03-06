---
title: Java - Arrays stream() method in Java
date: 2020-09-12 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---


## Arrays stream() method in Java


stream(T[] array)

to get a Sequential Stream from the array passed as the parameter with its elements.

It returns a sequential Stream with the elements of the array, passed as parameter, as its source.

Syntax:`public static <T> Stream<T> stream(T[] array)`

- Parameters: This method accepts a mandatory parameter array which is the array of whose elements are to be converted into a sequential stream.
- Return Value: This method returns a Sequential Stream from the array passed as the parameter.



Program 1: Arrays.stream() to convert string array to stream.

```java
import java.util.*;
import java.util.stream.*;

class GFG {
    public static void main(String[] args) {
        // Creating a String array
        String[] arr = { "Geeks", "for", "Geeks" };

        // Using Arrays.stream() to convert array into Stream
        Stream<String> stream = Arrays.stream(arr);

        // Displaying elements in Stream
        stream.forEach(str -> System.out.print(str + " "));
    }
}
// Output:
// Geeks for Geeks
```


Program 2: Arrays.stream() to convert int array to stream.

filter_none
edit
play_arrow

brightness_4
// Java program to demonstrate Arrays.stream() method

import java.util.*;
import java.util.stream.*;

class GFG {

    public static void main(String[] args)
    {
        // Creating an integer array
        int arr[] = { 1, 2, 3, 4, 5 };

        // Using Arrays.stream() to convert
        // array into Stream
        IntStream stream = Arrays.stream(arr);

        // Displaying elements in Stream
        stream.forEach(str -> System.out.print(str + " "));
    }
}
Output:
1 2 3 4 5
stream(T[] array, int startInclusive, int endExclusive)
The stream(T[] array, int startInclusive, int endExclusive) method of Arrays class in Java, is used to get a Sequential Stream from the array passed as the parameter with only some of its specific elements. These specific elements are taken from a range of index passed as the parameter to this method. It Returns a sequential Stream with the specified range of the specified array as its source.

Syntax:

public static <T> Stream<T>
    stream(T[] array, int startInclusive, int endExclusive)
Parameters: This method accepts three mandatory parameters:

array which is the array of whose elements are to be converted into a sequential stream.
startInclusive which is the first index to cover, inclusive
endExclusive which is the index immediately past the last index to cover
Return Value: This method returns a Sequential Stream formed from the range of elements of array passed as the parameter.

Below are the example to illustrate Arrays.stream() method:

Program 1: Arrays.stream() to convert string array to stream.

filter_none
edit
play_arrow

brightness_4
// Java program to demonstrate Arrays.stream() method

import java.util.*;
import java.util.stream.*;

class GFG {
    public static void main(String[] args)
    {

        // Creating a String array
        String[] arr = { "Geeks", "for", "Geeks",
                         "A", "Computer", "Portal" };

        // Using Arrays.stream() to convert
        // array into Stream
        Stream<String> stream = Arrays.stream(arr, 3, 6);

        // Displaying elements in Stream
        stream.forEach(str -> System.out.print(str + " "));
    }
}
Output:
A Computer Portal
Program 2: Arrays.stream() to convert int array to stream.

filter_none
edit
play_arrow

brightness_4
// Java program to demonstrate Arrays.stream() method

import java.util.*;
import java.util.stream.*;

class GFG {

    public static void main(String[] args)
    {
        // Creating an integer array
        int arr[] = { 1, 2, 3, 4, 5 };

        // Using Arrays.stream() to convert
        // array into Stream
        IntStream stream = Arrays.stream(arr, 1, 3);

        // Displaying elements in Stream
        stream.forEach(str -> System.out.print(str + " "));
    }
}
Output:
2 3
Attention reader! Don’t stop learning now. Get hold of all the important DSA concepts with the DSA Self Paced Course at a student-friendly price and become industry ready.
