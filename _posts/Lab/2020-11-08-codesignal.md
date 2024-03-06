---
title: Lab - CodeSignal Arcade Universe
date: 2020-11-08 11:11:11 -0400
description: Learning Path
categories: [Lab, codegame]
# img: /assets/img/sample/rabbit.png
tags: [Lab, codegame]
---

[toc]

---

# CodeSignal Arcade Universe

---

## Intro

---

## Smooth Sailing

1. All Longest Strings

```java
// Given an array of strings, return another array containing all of its longest strings.
// Example
// For inputArray = ["aba", "aa", "ad", "vcd", "aba"], the output should be
// allLongestStrings(inputArray) = ["aba", "vcd", "aba"].

String[] allLongestStrings(String[] inputArray){
    String l = ""; //full string with "-" separator
    for( String s : inputArray ){
        //length is first index of substring
        //if list has same size strings, add this one
        if( l.indexOf("-") == s.length() ) {
            l += s + "-";
        }
        //reset if list has smaller strings
        else if ( l.indexOf("-") < s.length() ) {
            l = s + "-";
        }
    }
    return l.split( "-" );
}


String[] allLongestStrings(String[] inputArray) {
    int longest = 0;
    for (int i = 0; i < inputArray.length; i++) {
        longest = inputArray[i].length() > longest ? inputArray[i].length():longest;
        }
    }
    final int longestLength = longest;
    return Stream.of(inputArray)
        .filter(s -> s.length() == longestLength)
        .toArray(String[]::new);
}


String[] allLongestStrings(String[] inputArray) {
    int size = 0;
    int max = 0;
    for (String wrd : inputArray){
        max = wrd.length() > max?wrd.length():max;
    }
    for (String wrd : inputArray){
        if(wrd.length() == max){
            size++;
        }
    }
    String[] ans = new String[size];
    int count = 0;
    for (String wrd : inputArray){
        if(wrd.length() == max){
            ans[count] = wrd;
            count++;
        }
    }
    return ans;
}


String[] allLongestStrings(String[] inputArray) {
    ArrayList al = new ArrayList();
    int max = 0;
    for(String wrd : inputArray){
        max = wrd.length() > max?wrd.length():max;
    }
    for(String wrd : inputArray){
        if(wrd.length() == max){
            al.add(wrd);
        }
    }
    String[] outans = new String[al.size()];
    for(int i=0 ; i < al.size(); i++){
        outans[i] = al.get(i).toString();
    }
    return outans;
}


String[] allLongestStrings(String arr[]) {
    int max_len = Arrays.stream(arr).max(Comparator.comparingInt(String::length)).get().length();
    return Arrays.stream(arr).filter(x -> x.length() == max_len).toArray(String[]::new);
}


```

1. All Longest Strings

```java

```

1. All Longest Strings

```java

```

1. All Longest Strings

```java

```
1. All Longest Strings

```java

```
1. All Longest Strings

```java

```

1. All Longest Strings

```java

```
1. All Longest Strings

```java

```
1. All Longest Strings

```java

```
1. All Longest Strings

```java

```
1. All Longest Strings

```java

```



---

## Edge of the Ocean

1. adjacentElementsProduct

```java
int adjacentElementsProduct(int[] inputArray) {
    Arrays.sort(inputArray);
    return inputArray[-1] + inputArray[-2];
}

int adjacentElementsProduct(int[] arr) {
    int prev = arr[0] * arr[1];
    for (int i = 0; i < arr.length-1; i++) {
        int curr = arr[i] * arr[i + 1];
        prev = curr > prev?curr:prev;
        // if (curr > prev) {
        //     prev = curr;
        // }
    }
    return prev;
}

int adjacentElementsProduct(int[] inputArray) {
    int ans=Integer.MIN_VALUE;
    for(int i=0; i<inputArray.length-1; i++){
        ans = inputArray[i]*inputArray[i+1]>ans?inputArray[i]*inputArray[i+1]:ans;
    }
    return ans;
}
```

2. shapeArea

```java
int shapeArea(int n) {
  return (n*n)+( (n-1)*(n-1) );
}


int shapeArea(int n) {
    int sum = 1;
    while(int i = 0; i < n; i++){
        sum += 2^n;
    }
    return sum;
}

int shapeArea(int n) {
    if(n == 1) return 1;
    return 4*(n-1) + shapeArea(n-1);
}
```


3. Make Array Consecutive 2

```java
Input: [6, 2, 3, 8]
Expected Output: 3

int makeArrayConsecutive2(int[] statues) {
    int min = statues[0];
    int max = statues[0];
    for (int i=1; i<statues.length; ++i) {
        min = Math.min(min, statues[i]);
        max = Math.max(max, statues[i]);
    }
    return (max-min+1) - statues.length;
}

int makeArrayConsecutive2(int[] statues) {
	Arrays.sort(statues);
	return (statues[statues.length-1] - statues[0] + 1) - statues.length;
}
```

4. almostIncreasingSequence

```java
Note: sequence a0, a1, ..., an is considered to be a strictly increasing if a0 < a1 < ... < an. Sequence containing only one element is also considered to be strictly increasing.
Example
For sequence = [1, 3, 2, 1], the output should be
almostIncreasingSequence(sequence) = false


boolean almostIncreasingSequence(int[] sequence) {
    int numErr = 0;
    for (int i = 0; i < sequence.length - 1; i++) {
        if (sequence[i] >= sequence[i+1]) {
            numErr += 1;
            if (i - 1 >= 0 && i + 2 <= sequence.length - 1 && sequence[i] >= sequence[i+2] && sequence[i-1] >= sequence[i+1]) {
                return false;
            }
        }
    }
    return numErr <= 1;
}

boolean almostIncreasingSequence(int[] sequence) {
    boolean flag=true;
    int seq1=0;
    int seq2=0;
    for(int i=0;i<sequence.length-1;i++){
        if(sequence[i]>=sequence[i+1]) seq1++;
    }
    for(int k=0;k<sequence.length-2;k++){
        if(sequence[k]>=sequence[k+2]) seq2++;
    }
    if(seq1+seq2>2) flag=false;
    return flag;
}
```


5. matrixElementsSum

```java
// After becoming famous, the CodeBots decided to move into a new building together. Each of the rooms has a different cost, and some of them are free, but there's a rumour that all the free rooms are haunted! Since the CodeBots are quite superstitious, they refuse to stay in any of the free rooms, or any of the rooms below any of the free rooms.

// Given matrix, a rectangular matrix of integers, where each value represents the cost of the room, your task is to return the total sum of all rooms that are suitable for the CodeBots (ie: add up all the values that don't appear below a 0).

int matrixElementsSum(int[][] matrix) {
    int sum = 0;
    for (int c = 0; c < matrix[0].length; ++c)
        for (int r = 0; r < matrix.length; ++r) {
            if (matrix[r][c] > 0)
                sum += matrix[r][c];
            else break;
        }
    return sum;
}

int matrixElementsSum(int[][] matrix) {
    int rooms = matrix[0].length;
    int floors = matrix.length;
    int sum = 0;
    for(int i = 0; i < rooms; i++) {
        for(int j = 0; j < floors && matrix[j][i] > 0; j++) {
            sum += matrix[j][i];
        }
    }
    return sum;
}
```

---


### The Journey Begins

1. add

```java
int add(int param1, int param2) {
    return param1 + param2;
}
```

2. centuryFromYear

```java
    int centuryFromYear(int year) {
        return 1 + (year - 1) / 100;
    }

    int centuryFromYear(int year) {
        return (year+99)/100;
    }

    int centuryFromYear(int year) {
        int century = 0;
        int modnum = year % 100;
        if(modnum != 0){
            century = year/100 + 1;
        }
        else {
            century = year/100;
        }
        return century;
    }

    int centuryFromYear(int year) {
        int century = year/100;
        if(year % 100 != 0){
            century++;
        }
        return century;
    }
```


3. checkPalindrome

```java
    boolean checkPalindrome(String inString) {
        return inString.equals(new StringBuilder(inString).reverse().toString());
    }

    boolean checkPalindrome(String s) {
        return new StringBuilder(s).reverse().toString().equals(s);
    }

    boolean checkPalindrome(String inString) {
        for(int i = 0; i < inString.length()/2; i++){
            if(inString.charAt(i) != inString.charAt(inString.length()-i-1))
                return false;
        }
        return true;
    }

    boolean checkPalindrome(String inString) {
        int i = 0;
        int j = inString.length()-1;
        while(i<j){
            if(inString.charAt(i) == inString.charAt(j)){
                i++;
                j--;
            }
            else{
                return false;
            }
        }
        return true;
    }
```
