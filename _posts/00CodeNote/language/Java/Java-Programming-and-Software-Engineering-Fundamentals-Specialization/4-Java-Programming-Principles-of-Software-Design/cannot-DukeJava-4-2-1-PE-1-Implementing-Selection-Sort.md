---
title: Java - DukeJava - 4-2-1 Programming Exercise 1 Implementing Selection Sort
date: 2020-09-13 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---

[toc]

---

# DukeJava - 4-2-1 Programming Exercise 1 Implementing Selection Sort

Java-Programming-and-Software-Engineering-Fundamentals-Specialization.
- 4.Java-Programming-Principles-of-Software-Design
  - 4-2 Earthquakes: Sorting Algorithms
    - 4-2-1 Programming Exercise 1 Implementing Selection Sort

Resource Link: http://www.dukelearntoprogram.com/course4/index.php

ProjectCode: https://github.com/ocholuo/language/tree/master/0.project/javademo

---

- The class Location, from the Android platform and revised for this course, a data class representing a geographic location. One of the constructors has parameters latitude and longitude, and one of the public methods is distanceTo.

- The class QuakeEntry, from the lesson, which has a constructor that requires latitude, longitude, magnitude, title, and depth. It has several get methods and a toString method.

- The class EarthQuakeParser, from the lesson, which has a read method with one String parameter that represents an XML earthquake data file and returns an ArrayList of QuakeEntry objects.

- The QuakeSortWithTwoArrayLists
  - to sort using two ArrayLists of QuakeEntry.

- The QuakeSortInPlace class
  - it implements the selection sort algorithm that sorts earthquakes by magnitude in place, in one ArrayList of QuakeEntry.


## Assignment 1: Sort by Depth

add methods in the QuakeSortInPlace class to sort the quakes by depth, from largest depth to smallest depth. This will mean any positive depth values will be first, followed by depths with increasingly negative values, e.g., 200.00, 0.00, -20000.00, -100000.00.


1. Write the method getLargestDepth
   - has two parameters, an ArrayList of type QuakeEntry named quakeData and an int named from representing an index position in the ArrayList.
   - This method returns an integer representing the index position of the QuakeEntry with the largest depth considering only those QuakeEntry from position from to the end of the ArrayList.


2. Write the void method sortByLargestDepth
   - has one parameter, an ArrayList of type QuakeEntry named in.
   - This method sorts the QuakeEntry in the ArrayList by depth using the selection sort algorithm, but in reverse order from largest depth to smallest depth (the QuakeEntry with the largest depth should be in the 0th position in the ArrayList).
   - This method should call the method getLargestDepth repeatedly until the ArrayList is sorted.


3. Modify the testSort method
   - to comment out the line sortByMagnitude and add below this line a call to sortByLargestDepth. Run your program on any data file and you should see the earthquakes in sorted order from largest depth to smallest depth.

---

## Assignment 2: Bubble Sort

1. Write the void method onePassBubbleSort
   - has two parameters, an ArrayList of type QuakeEntry named quakeData and an int named numSorted that represents the number of times this method has already been called on this ArrayList and thus also represents the number of the elements that are guaranteed to already be where they belong when the ArrayList is sorted by magnitude.
   - This method makes one pass of bubble sort on the ArrayList.
   - It should take advantage of the fact that the last numSorted elements are already in sorted order.

2. Write the void method sortByMagnitudeWithBubbleSort that has one parameter, an ArrayList of type QuakeEntry named in.
   - If the ArrayList in has N elements in it,
   - this method should call onePassBubbleSort N – 1 times to sort the elements in in.


3. Modify the testSort method to comment out the line sortByLargestDepth, and add below this line a call to sortByMagnitudeWithBubbleSort


---

## Assignment 3: Check for Completion



1. Write the method checkInSortedOrder
   - has one parameter, an ArrayList of type QuakeEntry named quakes.
   - This method returns true if the earthquakes are in sorted order by magnitude from smallest to largest.
   - Otherwise this methods returns false.
   - You’ll need to loop through the ArrayList and check adjacent earthquakes to see if any are out of order.


2. Write the void method sortByMagnitudeWithBubbleSortWithCheck
   - has one parameter, an ArrayList of type QuakeEntry named in.
   - If the ArrayList in has N elements in it, this method should call onePassBubbleSort at most N – 1 times.
   - This method should call checkInSortedOrder and stop early if the ArrayList is already sorted. This method should print how many passes were needed to sort the elements.


3. Modify the testSort method to call to sortByMagnitudeWithBubbleSortWithCheck
   - Run your program on any data files earthquakeDataSampleSix1.atom (should sort after 2 passes) and earthquakeDataSampleSix2.atom (should sort after 3 passes). Both of these files have five earthquakes.


4. Write the void method sortByMagnitudeWithCheck
   - has one parameter, an ArrayList of type QuakeEntry named in.
   - This method sorts earthquakes by their magnitude from smallest to largest using selection sort similar to the sortByMagnitude method.
   - However, this method should call checkInSortedOrder and stop early if the ArrayList is already sorted. This method should print how many passes were needed to sort the elements. For selection sort, one pass has exactly one swap.


5. Modify the testSort method to call to sortByMagnitudeWithCheck
   - Run your program on any data files earthquakeDataSampleSix1.atom (should sort after 3 passes) and earthquakeDataSampleSix2.atom (should sort after 4 passes). Both of these files have five earthquakes.
