---
title: Java - DukeJava - 4-3-1 Programming Exercise 1 Generating Random Text
date: 2020-09-14 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---

# DukeJava - 4-3-1 Programming Exercise 1 Generating Random Text

[toc]

Java-Programming-and-Software-Engineering-Fundamentals-Specialization
- 4.Java-Programming-Principles-of-Software-Design
  - N-Grams: Predictive Text
    - 4-3-1 Programming Exercise 1 Generating Random Text

Resource Link: http://www.dukelearntoprogram.com/course4/index.php

ProjectCode: https://github.com/ocholuo/language/tree/master/0.project/javademo

---

```
- The void method `setRandom` has one integer parameter named seed.
  - Using this method will allow to generate the same random text each time, which will help in testing your program.

- The void method `setTraining` has one String parameter named s.
  - The String s is used to initialize the training text.
  - It is important that you DO NOT change this line or it may affect your output: myText = s.trim();

- The `getRandomText` method has one integer parameter named numChars.
  - This method generates and returns random text that is numChars long.
  - Remember, for MarkovZero, this class generates each letter by randomly choosing a letter from the training text.

- The void method `runMarkovZero` has no parameters.
  - This method reads in a file the user chooses, creates a MarkovZero object, and then generates three sets of randomly generated text using the file read in to choose the random characters from.

- The void method `printOut` is called by runMarkovZero to print out the random text that was generated with around 60 characters per line.
  - DO NOT CHANGE THIS METHOD. You’ll need output generated in this format for some of the quiz questions.
```

---

## Assignment 1: MarkovZero and MarkovOne

```
1. Create `MarkovZero` generated texts by running the method `runMarkovZero` in `MarkovRunner`.
   - Run the program twice and note that the output is different each time you run it.


2. Modify the `runMarkovZero` method
   - call the `setRandom` method with the `seed 42`. Run this method at least twice.
   - change to `seed to 101`. Run it at least twice.
   - You should get different text than you got with the `seed 42`, but every time you run it you get the same text.

3. Create a new class called `MarkovOne`.
   - Copy the body of `MarkovZero` into `MarkovOne`. You’ll only need to change the name of the constructor to `MarkovOne` and add the same import that `MarkovZero` had, and then it should compile.
   - Right now, `MarkovOne` is only doing what `MarkovZero` did, since it is a copy of it. We will fix it shortly to use one character to predict text.

4. In the class `MarkovRunner`, make a copy of the method `runMarkovZero`, and name this method `runMarkovOne`.
   - Then change the line `MarkovZero markov = new MarkovZero();` to `MarkovOne markov = new MarkovOne();`


5. In the class `MarkovOne`, write the method getFollows that has one String parameter named key. This method should find all the characters from the private variable myText in `MarkovOne` that follow key and put all these characters into an ArrayList and then return this ArrayList. This algorithm for this method was described in “Finding Follow Sets.”
   - For example, if myText were “this is a test yes this is a test.”
   - call `getFollows(“t”)` should return an ArrayList with the Strings `“h”, “e”, “ “, “h”, “e”, “.”` as “t” appears 6 times.
   - The call `getFollows(“e”)` should return an ArrayList with the Strings `“s”, “s”, “s”`.
   - Your method should work even if key is a word. Thus, `getFollows(“es”)` should return an ArrayList with the Strings `“t”, “ “, “t”`.

6. Create a new class `Tester` and a void method in this class named `testGetFollows` with no parameters.
   - This method should create a `MarkovOne` object,
   - set the training text as “this is a test yes this is a test.”.
   - Then have it call `getFollows` and print out the resulting ArrayList and also its size.
   - Be sure to test it on the three examples above and also on the Strings “.” and “t.”, which occur at the end of the String.

7. test `getFollows` on a file.
   - In the Tester class, write the void method `testGetFollowsWithFile` with no parameters.
   - This method should create a `MarkovOne` object, set the training text to a file the user selects (similar to the methods in `MarkovRunner`), and then call `getFollows`.
   - Run your program on `confucius.txt` and look for the characters that follow “t”. You should get 11548.


8. In the class `MarkovOne` modify the method getRandomText so that it works for the way it should for `MarkovOne`.
   - It should predict the next character, by finding all the characters that follow the current character in the training text, and then randomly picking one of them as the next character.

9. modified the `runMarkovOne` method in the class `MarkovRunner`.
   - Run this method with the random seed as 42 and the file confucius.txt. The first line of MarkovOne random text generated starts with:
   - nd are, Prevedowalvism n thastsour tr ndsang heag ti. the ffinthe
```


---

## Assignment 2: MarkovFour and MarkovModel


1. Create the class MarkovFour to use four characters to predict the next character. Copy and paste in MarkovOne and then modify it. You can watch the video “Implementing Order-Two” on how to create MarkovTwo from MarkovOne.

2. In the MarkovRunner class, create the method runMarkovFour to generate random text using the MarkovFour class. If you set the random seed with 25 and run this method on confucius.txt, the first line of text should start with:
ouses the people Minister said the that a many Project of it

3. Create the class MarkovModel to use N characters to predict the next character. An integer should be passed in with the constructor to specify the number of characters to use to predict the next character. Copy and paste in MarkovFour and then modify it.

4. In the MarkovRunner class, create the method runMarkovModel to generate random text using the MarkovModel class. If you set the random seed with 38 and run this method with N = 6 on confucius.txt, the first line of text should start with:
sters I could thrice before downloading, and his lord, might









.
