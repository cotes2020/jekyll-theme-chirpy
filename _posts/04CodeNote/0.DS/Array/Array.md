---
title: Data Structure - Array
date: 2020-09-10 11:11:11 -0400
description: IT Blog Pool
categories: [Data Structure, Array]
tags: [DataStructure, Array]
---


# Data Structure - Array

- [Data Structure - Array](#data-structure---array)
- [Arrays](#arrays)
- [Arrays in Java](#arrays-in-java)
  - [declare Array](#declare-array)
  - [Instantiate Array](#instantiate-array)
  - [Array Literal](#array-literal)
  - [Access Array Elements with for Loop](#access-array-elements-with-for-loop)
  - [access java arrays using foreach loops ???](#access-java-arrays-using-foreach-loops-)
  - [Arrays of Objects](#arrays-of-objects)
  - [error: access element outside the array size](#error-access-element-outside-the-array-size)
  - [Multidimensional Arrays](#multidimensional-arrays)
  - [Passing Arrays to Methods](#passing-arrays-to-methods)
  - [Returning Arrays from Methods](#returning-arrays-from-methods)
  - [Class Objects for Arrays](#class-objects-for-arrays)
  - [Array Members](#array-members)
  - [Cloning of arrays](#cloning-of-arrays)


---

# Arrays


array
- a collection of items stored at 连续的 **contiguous memory locations**.
- The idea is to `store multiple items of the same type together`. 
- This makes it easier to calculate the `position of each element` by simply adding an 抵消 offset to a base value, 
- i.e., the memory location of the first element of the array (generally denoted by the name of the array).

![array-2](https://i.imgur.com/0lvYfk8.png)

- The above image can be looked as a top-level view of a staircase where you are at the base of the staircase. 
- Each element can be uniquely identified by their `index` in the array.

Advantages of using arrays:
- `allow random access` of elements, makes **accessing elements by position** faster.
- have **better cache locality**, make a pretty big difference in performance.

```java
// A character array in C/C++/Java
char arr1[] = {'g', 'e', 'e', 'k', 's'};

// An Integer array in C/C++/Java
int arr2[] = {10, 20, 30, 40, 50};
```

> Usually, an array of characters is called a ‘string’,
whereas an array of ints or floats is called simply an array.


---


# Arrays in Java


An array is a group of like-typed variables that are referred to by a common name.Arrays in Java work differently than they do in C/C++. Following are some important point about Java arrays.

- In Java all arrays are `dynamically allocated`.
- arrays are objects in Java, we can find their length using member `length`.
  - different from C/C++ where we find length using `sizeof`.
- A Java array variable can also be declared like other variables with `[]` after the data type.
- The variables in the array are **ordered** and each have an `index` beginning from 0.
- Java array can be also be used as a static field, a local variable or a method parameter.
- The size of an array must be specified by an int value and not long or short.
- The direct superclass of an array type is Object.
- Every array type implements the interfaces `Cloneable` and `java.io.Serializable`.


Array can contains primitives (int, char, etc) as well as object (or non-primitives) references of a class depending on the definition of array.
- In case of primitives data types, the actual values are stored in contiguous memory locations.
- In case of objects of a class, the actual objects are stored in heap segment.

---

## declare Array

One-Dimensional Arrays :
- The general form of a one-dimensional array declaration is

```java
type var-name[];
OR
type[] var-name;
```

An array declaration has two components:
- the type
- the name.

the element `type` for the array determines what type of data the array will hold.

```java
// both are valid declarations
int intArray[];
int[] intArray;

byte byteArray[];
short shortsArray[];
boolean booleanArray[];
long longArray[];
float floatArray[];
double doubleArray[];
char charArray[];

// an array of references to objects of
// the class MyClass (a class created by
// user)
MyClass myClassArray[];

Object[]  ao,        // array of Object
Collection[] ca;     // array of Collection of unknown type
```

Although declaration establishes the fact that intArray is an array variable,
- no array actually exists.
- It simply tells to the compiler that this(intArray) variable will hold an array of the integer type.
- To link intArray with an actual, physical array of integers, you must allocate one using `new` and assign it to intArray.

---

## Instantiate Array

When an array is declared, only a reference of array is created.

To actually create or give memory to array,

`var-name = new type [size];`

- `type` specifies the type of data being allocated
- `size` specifies the number of elements in the array
- `var-name` is the name of array variable that is linked to the array.
- to use new to allocate an array, must specify the type and number of elements to allocate.

```java
int intArray[];          //declaring array
intArray = new int[20];  // allocating memory to array

OR

int[] intArray = new int[20]; // combining both statements in one
```

1. The elements in the array allocated by `new` will automatically be initialized to `zero (for numeric types)`, `false (for boolean)`, or `null (for reference types)`.
Refer Default array values in Java
2. Obtaining an array is a two-step process.
   - First, must declare a variable of the desired array type.  
   - Second, must allocate the memory that will hold the array, using `new`, and assign it to the array variable. Thus, in Java all arrays are dynamically allocated.

---

## Array Literal

In a situation, where the size of the array and variables of array are already known, array literals can be used.

```java
int[] intArray = new int[]{ 1,2,3,4,5,6,7,8,9,10 };
// Declaring array literal
```

- The length of this array determines the length of the created array.
- There is no need to write the new int[] part in the latest versions of Java

---

## Access Array Elements with for Loop

Each element in the array is accessed via its index.
- The index begins with 0 and ends at `array.length-1`.
- All the elements of array can be accessed using Java for Loop.

```java
for (int i = 0; i < arr.length; i++){
  System.out.println(arr[i]);
}
```

---

## access java arrays using foreach loops ???

---


## Arrays of Objects

An array of objects is created just like an array of primitive type data items in the following way.

```java
Class[] arr = new Class[7];
```

- The student Array contains seven memory spaces each of size of student class in which the address of seven Student objects can be stored.
- The Student objects have to be instantiated using the constructor of the `Student` class and their references should be assigned to the array elements in the following way.

```java
Student[] arr = new Student[5];

class Student {
    public int roll_no;
    public String name;

    Student(int roll_no, String name) {
        this.roll_no = roll_no;
        this.name = name;
    }
}

// Elements of array are objects of a class Student.
public class GFG {

    public static void main (String[] args) {
        Student[] arr;
        arr = new Student[5];

        // initialize the first elements of the array
        arr[0] = new Student(1,"aman");
        arr[1] = new Student(2,"vaibhav");
        arr[2] = new Student(3,"shikar");
        arr[3] = new Student(4,"dharmesh");
        arr[4] = new Student(5,"mohit");

        // accessing the elements of the specified array
        for (int i = 0; i < arr.length; i++)
            System.out.println("Element at " + i + " : " +
                        arr[i].roll_no +" "+ arr[i].name);
    }
}
```

---

## error: access element outside the array size

JVM throws `ArrayIndexOutOfBoundsException` to indicate that array has been accessed with an illegal index.
- The index is either negative or greater than or equal to size of array.

```java
class GFG {
    public static void main (String[] args) {
        int[] arr = new int[2];
        arr[0] = 10;
        arr[1] = 20;

        for (int i = 0; i <= arr.length; i++)
            System.out.println(arr[i]);
    }
}
// Runtime error
// Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: 2
//     at GFG.main(File.java:12)
// Output:
10
20
```


---


## Multidimensional Arrays

Multidimensional arrays are arrays of arrays with each element of the array holding the reference of other array.
- These are also known as `Jagged Arrays`.
- A multidimensional array is created by appending one set of square brackets ([]) per dimension.

```java
int[][] intArray = new int[10][20];       //a 2D array or matrix
int[][][] intArray = new int[10][20][10]; //a 3D array


class multiDimensional {
    public static void main(String args[]) {

        // declaring and initializing 2D array
        int arr[][] = { {2,7,9},{3,6,1},{7,4,2} };

        // printing 2D array
        for (int i=0; i< 3 ; i++) {
            for (int j=0; j < 3 ; j++)
                System.out.print(arr[i][j] + " ");
            System.out.println();
        }
    }
}
// Output:
2 7 9
3 6 1
7 4 2
```

---

## Passing Arrays to Methods

```java
// pass array to method sum for calculating sum of array’s values.

class Test {     

    public static void sum(int[] arr)  {
        int sum = 0;
        for (int i = 0; i < arr.length; i++){
            sum+=arr[i];
        }
        System.out.println("sum of array values : " + sum);
    }


    public static void main(String args[])  {
        int arr[] = {3, 1, 2, 5, 4};
        sum(arr);
    }


}
```

---

## Returning Arrays from Methods

- a method can return an array.
- For example, below program returns an array from method m1.

```java
class Test {     

    // Driver method
    public static void main(String args[])  {
        int arr[] = m1();
        for (int i = 0; i < arr.length; i++)
            System.out.print(arr[i]+" ");
    }

    public static int[] m1()  {
        // returning  array
        return new int[]{1,2,3};
    }
}
// Output:
1 2 3
```

---

## Class Objects for Arrays

Every array has an associated Class object, shared with all other arrays with the same component type.

```java
class Test
{  
    public static void main(String args[])  
    {
        int intArray[] = new int[3];
        byte byteArray[] = new byte[3];
        short shortsArray[] = new short[3];

        // array of Strings
        String[] strArray = new String[3];

        System.out.println(intArray.getClass());
        System.out.println(intArray.getClass().getSuperclass());
        System.out.println(byteArray.getClass());
        System.out.println(shortsArray.getClass());
        System.out.println(strArray.getClass());
    }
}
// Output:
// class [I
// class java.lang.Object
// class [B
// class [S
// class [Ljava.lang.String;

// Explanation :
// The string “[I” is the run-time type signature for the class object “array with component type int“.
// The only direct superclass of any array type is java.lang.Object.
// The string “[B” is the run-time type signature for the class object “array with component type byte“.
// The string “[S” is the run-time type signature for the class object “array with component type short“.
// The string “[L” is the run-time type signature for the class object “array with component type of a Class”. The Class name is then followed.
```

---


## Array Members

arrays are object of a class and direct superclass of arrays is class Object.
The members of an array type are all of the following:
- The public final field `length`, which contains the number of components of the array. length may be positive or zero.
- All the members inherited from class `Object`; the only method of Object that is not inherited is its `clone` method.
- The public method `clone()`, which overrides clone method in class Object and throws no checked exceptions.


---

## Cloning of arrays

When you clone a single dimensional array, such as `Object[]`
- a “deep copy” is performed with the new array containing copies of the original array’s elements as opposed to references.

```java
class Test{
    public static void main(String args[])  {
        int intArray[] = {1,2,3};
        int cloneArray[] = intArray.clone();
        // will print false as deep copy is created
        // for one-dimensional array
        System.out.println(intArray == cloneArray);

        for (int i = 0; i < cloneArray.length; i++) {
            System.out.print(cloneArray[i]+" ");
        }
    }
}
// Output:
// false
// 1 2 3
```

A clone of a `multidimensional array` (like `Object[][]`) is a “shallow copy” 
- it creates only a single new array with each element array a reference to an original element array but subarrays are shared.

```java
class Test{     
    public static void main(String args[])  {
        int intArray[][] = {{1,2,3},{4,5}};
        int cloneArray[][] = intArray.clone();

        // will print false
        System.out.println(intArray == cloneArray);

        // will print true
        // as shallow copy is created
        // i.e. sub-arrays are shared
        System.out.println(intArray[0] == cloneArray[0]);
        System.out.println(intArray[1] == cloneArray[1]);
    }
}
```
