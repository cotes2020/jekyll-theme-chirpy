---
layout: post
title: "Integer Overflow bugs"
categories: explained
tags: [under the hood, linux, security]
---

## Overview:

Hello everyone, In this Post we will take a look at one of the most troublesome yet not very known type of software bugs: Integer overflows

## back to school:

### Bits, bytes, and words
The most fundamental unit of computer memory is the **bit**. A bit can be a tiny magnetic region on a hard disk, a tiny dent in the reflective material on a CD or DVD, or a tiny transistor on a memory stick. Whatever the physical implementation, the important thing to know about a bit is that, like a switch, it can only take one of two values: it is either “on” or “off”.

A collection of 8 bits is called a **byte** and (on the majority of computers today) a collection of 4 bytes, or 32 bits, is called a **word**. Each individual data value in a data set is usually stored using one or more bytes of memory, but at the lowest level, any data stored on a computer is just a large collection of bits.

The number of bytes and words used for an individual data value will vary depending on the storage format, the operating system, and even the computer hardware, but in many cases, a single letter or character of text takes up one byte and an integer, or whole number, takes up one word. A real or decimal number takes up one or two words depending on how it is stored.

For example, the text  “hello”  would take up 5 bytes of storage, one per character. The text  “12345”  would also require 5 bytes. The integer 12,345 would take up 4 bytes (1 word), as would the integers 1 and 12,345,678. The real number 123.45 would take up 4 or 8 bytes, as would the values 0.00012345 and 12345000.0.

### what is an integer ?
An integer, in the context of computing, is a variable capable of
representing a real number with no fractional part.  Integers are typically
the same size as a pointer on the system they are compiled on (i.e. on a 32
bit system, an integer is 32 bits long, on a 64 bit system,
such as SPARC, an integer is 64 bits long).

Integers, like all variables are just regions of memory.  When we talk
about integers, we usually represent them in decimal, as that is the
numbering system humans are most used to.  Computers, being digital, cannot deal with decimal, so internally to the computer integers are stored in
binary.  Binary is another system of representing numbers which uses only
two numerals, 1 and 0, as opposed to the ten numerals used in decimal.  As
well as binary and decimal, hexadecimal (base sixteen) is often used in
computing as it is very easy to convert between binary and hexadecimal.

Integers are commonly stored using a word of memory, which is 4 bytes or 32 bits, so integers from 0 up to 4,294,967,295 (this is the maximum of unsigned integers)

Since it is often necessary to store negative numbers, there needs to be a
mechanism to represent negative numbers using only binary.  The way this is
accomplished is by using the most significant bit (MSB) of a variable to
determine the sign: if the MSB is set to 1, the variable is interpreted as
negative; if it is set to 0, the variable is positive.

not all variables are signed, meaning they do not all use the MSB to
determine whether they are positive or negative.  These variable are known
as unsigned and can only be assigned positive values, whereas variables
which can be either positive or negative are called signed.

A signed integer ranges from -2147483648 to 2147483647. An unsigned integer is a 32-bit ranges from 0 to 4294967295. The signed integer is represented in twos complement notation.

## What is an integer overflow:

We have two unsigned integers, a and b, both of which are 32 bits long.  We
assign to a the maximum value a 32 bit integer can hold, and to b we assign
1 We add a and b together and store the result in a third unsigned 32 bit
integer called r:

    a = 0xffffffff
    b = 0x1
    r = a + b

Now, since the result of the addition cannot be represented using 32 bits,
the result, in accordance with the ISO standard, is reduced modulo
0x100000000.

    r = (0xffffffff + 0x1) % 0x100000000
    r = (0x100000000) % 0x100000000 = 0

Reducing the result using modulo arithmetic basically ensures that only the
lowest 32 bits of the result are used, so integer overflows cause the
result to be truncated to a size that can be represented by the variable.
This is often called a "wrap around", as the result appears to wrap around
to 0.

To make it easier, imagine that you are driving you car and you already made 999999 kilometres with it . now when you drive another kilometre the counter will be reset to 0. what happened ?? Same as the previous example the counter wasn't supposed to hold value of 1000000 so it went back to 0.

Since an integer is a fixed size there is a fixed maximum value it can store.  When an attempt is made to store a value greater than this maximum value it is known as an integer overflow.  an **integer overflow** occurs when an arithmetic operation attempts to create a numeric value that is outside of the range that can be represented with a given number of digits – either higher than the maximum or lower than the minimum representable value. The ISO C99 standard says that an **integer overflow** causes "undefined behavior", meaning that compilers conforming to the standard may do anything they like from completely ignoring the overflow to aborting the program.  Most compilers seem to ignore the overflow, resulting in an unexpected or erroneous result being stored.

## examples

### Example 1

in this first example we will see how an integer overflow can lead to another vulnerability which is a heap overflow.
let's check is code
```
#include <stdio.h>

void *AllocateFileMemory(unsigned int file_size, unsigned int name)
{
unsigned int total_size = file_size + name_size;
void *buffer = malloc(total_size);
printf("Allocated space: %d\n", total_size);
return buffer;
}

int main()
{
int file_size = 12;
int name_size = 10;
AllocateFileMemory(file_size, name_size);
return 0;
}
```
this code will allocate space to a file(let's say it is code in a file transfer app) by adding it's size and name so if the file_size is 16 and name_size is 4 the code will allocate 20 bytes.

But if file_size is 4294967295 and we want to add 4 the allocated space will become 3. So the application allocated 3 bytes of space for a  file with over 4294967295 bytes in size. this will create a heap overflow in our program which can lead to a huge security problem.

### Example 2

now this is another example from a CTF challenge:
When we connect to the server, we're given a number and we're prompted for a number to make it negative
```
$ nc vuln2014.picoctf.com 50000
Your number is 2088572. Can you make it negative by adding a positive integer?
12345
Almost... the sum was 2100917.

Thanks for playing.
```
The solution is to cause an integer overflow. 2,147,483,647 is the maximum positive value for a 32-bit signed integer, so if we enter that it should cause an integer overflow when added to the server's number:
```
$ nc vuln2014.picoctf.com 50000
Your number is 440902. Can you make it negative by adding a positive integer?
2147483647
Congratulations! The sum is -2147042747. Here is the flag: That_was_easssy!

Thanks for playing.
```
so when we added 2147483647 to 440902 the cpu was unable to store the result because it's the greater than a singed integer could store what happened is that variable went back -2147483648 ( remember your car counter ??) and started calculating from there...

### Example 3

Another challenge:
```
$ ./bonuspoints
Hello, here you can get some bonus points for the competition.
You cannot get more than 100 bonus points.
If you go above 1 000 you win.
Your score is currently 43
How many bonus points do you want ?
>>> -44
Your new score is 4294967295
Congratulations !
```

The vulnerability here is an unsigned integer overflow. The challenge does not allow us to add more than 100 bonus points, but we can take away more than 100. Therefore, if we take away our current score we get to `0`, and if we take away one more point we get to `0xffffffff`
We can see that we manage to put our score to the value `4294967295` (`0xffffffff` in hex) and therefore validate the test!

## Real world scenarios
[The Explosion of the Ariane 5](https://hownot2code.com/2016/09/02/a-space-error-370-million-for-an-integer-overflow/).
[Boeing 787 software crash](https://www.engadget.com/2015-05-01-boeing-787-dreamliner-software-bug.html)
[year 2038 Problem](https://en.wikipedia.org/wiki/Year_2038_problem)
[Casino slot machine](https://www.reddit.com/r/softwaregore/comments/dqqfq7/woman_wins_4294967276_on_a_slot_machine_but/)

## Conclusion
Integer overflows can be extremely dangerous, partly because it is
impossible to detect them after they have happened.  If an integer overflow
takes place, the application cannot know that the calculation it has
performed is incorrect, and it will continue under the assumption that it
is.  Even though they can be difficult to exploit, and frequently cannot be
exploited at all, they can cause unepected behaviour, which is never a good
thing in a secure system.