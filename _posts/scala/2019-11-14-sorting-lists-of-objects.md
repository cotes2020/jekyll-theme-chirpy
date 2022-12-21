---
title: Sorting Lists of Objects in Scala
description: Learn how to set sort the lists in Scala. The blog talks about the usage of sortWith and sortBy along with coding examples
tags: ["scala", "jvm"]
category: ["programming", "tutorial"]
date: 2019-11-14
permalink: '/scala/sorting-lists-of-objects/'
---

<hr>
### Introduction

One of the most common ADT that a developer uses in their day-to-day coding is List. And one of the most common operations a developer performs on a list is to order it or sort it with given criteria. In this article, I will focus on sorting a list of objects in Scala.

Mainly, there are two ways of sorting a list in Scala, i.e.

* ```sortWith```
* `sortBy`

Let's consider the popular example of sorting IMDB ratings. Below is my IMDB class.

```scala
case class ImdbRating(name: String, ratings: Double)
```

Here is a list of the top five rated movies of all time (source).


```scala
val ratings = List(
  ImdbRating("The Shawshank Redemption", 9.3),
  ImdbRating("The Godfather ", 9.2),
  ImdbRating("The Dark Knight", 9.1),
  ImdbRating("The Godfather: Part II", 9.0),
  ImdbRating("The Lord of the Rings: The Return of the King", 8.9)
)
```

If you observe closely, the above list is ordered by ratings, but my requirement is to order movies by length of their names.

### sortWith
First, I will try to order the list using `sortWith`. `sortWith` sorts a given list based on the comparison function that is provided to it. It is a stable sort, which means that an item will not lose its original position if two elements are equal. Here is the code to sort the list by length of names:

```scala
val sortedRatings = ratings.sortWith(_.name.size < _.name.size)
```

If I want to order the list in descending order, then all I need to do is to reverse the `<` operator. 

### sortBy
Another way to order the above list is to use `sortBy`. `sortBy` sorts a given sequence according to the implicitly defined natural `Ordering`. Like `sortWith`, this sort is stable as well. Here is the code to order the list.

```scala
val sortedRatings = ratings.sortBy(_.name.size)
```

There is a third way to sort the list of objects that I have not discussed here, and that is to extend the `Ordered` trait. The trait forces you to implement the `compare` method. `Ordered` trait is somewhat like `Comparable` interfaces in java.

If you want more details about sorting or Ordered trait, you can refer to the following videos in which I provide additional examples.

<iframe src="https://www.youtube.com/embed/hjY_mC7dPxc"></iframe>
