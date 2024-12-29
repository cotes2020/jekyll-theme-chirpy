---
title: "Data Representation - Floating Point Numbers"
subtitle: "「数据表示」浮点数"
layout: post
author: "Hux"
header-style: text
hidden: true
tags:
  - 笔记
  - 基础
  - C
  - C++
---

In the last episode we talked about the data representation of integer, a kind
of fixed-point numbers. Today we're going to learn about floating-point numbers.

Floating-point numbers are used to _approximate_ real numbers. Because of the
fact that all the stuffs in computers are, eventually, just a limited sequence
of bits. The representation of floating-point number had to made trade-offs
between _ranges_ and _precision_.

Due to its computational complexities, CPU also have a dedicated set of
instructions to accelerate on floating-point arithmetics.


Terminologies
-------------

The terminologies of floating-point number is coming from the
[_scientific notation_](https://en.wikipedia.org/wiki/Scientific_notation),
where a real number can be represented as such:

```
1.2345 = 12345 × 10 ** -4
         -----   --    --
  significand^   ^base  ^exponent
```

- _significand_, or _mantissa_, 有效数字, 尾数
- _base_, or _radix_ 底数
- _exponent_, 幂

So where is the _floating point_? It's the `.` of `1.2345`. Imaging the dot
can be float to the left by one to make the representation `.12345`.

The dot is called _radix point_, because to us it's seem to be a _decimal point_,
but it's really a _binary point_ in the computers.

Now it becomes clear that, to represent a floating-point number in computers,
we will simply assign some bits for _significand_ and some for _exponent_, and
potentially a bit for _sign_ and that's it.


IEEE-754 32-bits Single-Precision Floats 单精度浮点数
----------------------------------------

- <https://en.wikipedia.org/wiki/Single-precision_floating-point_format>

It was called **single** back to IEEE-754-1985 and now **binary32** in the
relatively new IEEE-754-2008 standard.

```cpp
       (8 bits)             (23 bits)
sign   exponent             fraction
  0   011 1111 1    000 0000 0000 0000 0000 0000

 31   30 .... 23    22 ....................... 0
```

- The _sign_ part took 1 bit to indicate the sign of the floats. (`0` for `+`
and `1` for `-`. This is the same treatment as the [sign magnitute](2020-06-19-data-rep-int.md##sign-magnitude-原码).
- The _exponent_ part took 8 bits and used [_offset-binary (biased) form_](2020-06-19-data-rep-int.md#offset-binary-移码) to represent a signed integer.
It's a variant form since it took out the `-127` (all 0s) for zero and `+128`
(all 1s) for non-numbers, thus it ranges only `[-126, 127]` instead of
`[-127, 128]`. Then, it choose the zero offset of `127` in these 254 bits (like
using `128` in _excess-128_), a.k.a the _exponent bias_ in the standard.
- The _fraction_ part took 23 bits with an _implicit leading bit_ `1` and
represent the actual _significand_ in total precision of 24-bits.

Don't be confused by why it's called _fraction_ instead of _significand_!
It's all because that the 23 bits in the representation is indeed, representing
the fraction part of the real significand in the scientific notation.

The floating-point version of "scientific notation" is more like:

```cpp
(leading 1)
   1. fraction  ×  2 ^ exponent   ×  sign
      (base-2)           (base-2)
```

So what number does the above bits represent?

```cpp
S     F   ×  E  =  R
+  1.(0)  ×  0  =  1
```

Aha! It's the real number `1`!
Recall that the `E = 0b0111 1111 = 0` because it used a biased representation!

We will add more non-trivial examples later.


Demoing Floats in C/C++
-----------------------

Writing sample code converting between binaries (in hex) and floats are not
as straightforward as it for integers. Luckily, there are still some hacks to
perform it:

### C - Unsafe Cast

We unsafely cast a pointer to enable reinterpretation of the same binaries.

```cpp
float f1 = 0x3f800000; // C doesn't have a floating literal taking hex.
printf("%f \n", f1);   // 1065353216.000000 (???)

uint32_t u2 = 0x3f800000;
float* f2 = (float*)&u2;   // unsafe cast
printf("%f \n", *f2);      // 1.000000
```

### C - Union Trick

Oh I really enjoyed this one...Union in C is not only untagged union, but also
share the exact same chunk of memory. So we are doing the same reinterpretation,
but in a more structural and technically fancier way.

```cpp
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

float pi = (float)M_PI;
union {
    float f;
    uint32_t u;
} f2u = { .f = pi };  // we took the data as float

printf ("pi : %f\n   : 0x%" PRIx32 "\n", pi, f2u.u);  // but interpret as uint32_t
pi : 3.141593
   : 0x40490fdb
```

N.B. this trick is well-known as [type punning](https://en.wikipedia.org/wiki/Type_punning):

> In computer science, type punning is a common term for any programming technique that subverts or circumvents the type system of a programming language in order to achieve an effect that would be difficult or impossible to achieve within the bounds of the formal language.

### C++ - `reinterpret_cast`

C++ does provide such type punning to the standard language:

```cpp
uint32_t u = 0x40490fdb;
float a = *reinterpret_cast<float*>(&u);
std::cout << a;  // 3.14159
```

N.B. it still need to be a conversion between pointers,
see <https://en.cppreference.com/w/cpp/language/reinterpret_cast>.

Besides, C++ 17 does add a floating point literal that can take hex, but it
works in a different way, using an explicit radix point in the hex:

```cpp
float f = 0x1.2p3;  // 1.2 by 2^3
std::cout << f;     // 9
```

That's try with another direction:

```cpp
#include <iostream>
#include <stdint.h>
#include <inttypes.h>

int main() {
  double qNan = std::numeric_limits<double>::quiet_NaN();
  printf("0x%" PRIx64 "\n", *reinterpret_cast<uint64_t*>(&qNan));
  // 0x7ff8000000000000, the canonical qNaN!
}
```


Representation of Non-Numbers
-----------------------------

There are more in the IEEE-754!

Real numbers doesn't satisfy [closure property](https://en.wikipedia.org/wiki/Closure_(mathematics))
as integers does. Notably, the set of real numbers is NOT closed under the
division! It could produce non-number results such as **infinity** (e.g. `1/0`)
and [**NaN (Not-a-Number)**](https://en.wikipedia.org/wiki/NaN) (e.g. taking
a square root of a negative number).

It would be algebraically ideal if the set of floating-point numbers can be
closed under all floating-point arithmetics. That would made many people's life
easier. So the IEEE made it so! Non-numeber values are squeezed in.

We will also include the two zeros (`+0`/`-0`) into the comparison here,
since they are also special by being the only two demanding an `0x00` exponent:

```cpp
             binary                |    hex    |
--------------------------------------------------------
0 00000000 00000000000000000000000 = 0000 0000 = +0
1 00000000 00000000000000000000000 = 8000 0000 = −0

0 11111111 00000000000000000000000 = 7f80 0000 = +infinity
1 11111111 00000000000000000000000 = ff80 0000 = −infinity

_ 11111111 10000000000000000000000 = _fc0 0000 = qNaN (canonical)
_ 11111111 00000000000000000000001 = _f80 0001 = sNaN (one of them)
```

```cpp
      (8 bits)  (23 bits)
sign  exponent  fraction
  0      00     0 ...0 0  = +0
  1      00     0 ...0 0  = -0
  0      FF     0 ...0 0  = +infinity
  1      FF     0 ...0 0  = -infinity
  _      FF     1 ...0 0  = qNaN (canonical)
  _      FF     0 ...0 1  = sNaN (one of them)
```

Encodings of qNaN and sNaN are not specified in IEEE 754 and implemented
differently on different processors. Luckily, both x86 and ARM family use the
"most significant bit of fraction" to indicate whether it's quite.

### More on NaN

If we look carefully into the IEEE 754-2008 spec, in the _page35, 6.2.1_, it
actually defined anything with exponent `FF` and not a infinity (i.e. with
all the fraction bits being `0`), a NaN!

> All binary NaN bit strings have all the bits of the biased exponent field E set to 1 (see 3.4). A quiet NaN bit string should be encoded with the first bit (d1) of the trailing significand field T being 1. A signaling NaN bit string should be encoded with the first bit of the trailing significand field being 0.

That implies, we actually had `2 ** 24 - 2` of NaNs in a 32-bits float!
The `24` came from the `1` sign bit plus `23` fractions and the `2` excluded
were the `+/- inf`.

The continuous 22 bits inside the fraction looks quite a waste, and there
would be even 51 bits of them in the `double`! We will see how to made them useful
in later episodes (spoiler: they are known as _NaN payload_).

It's also worth noting that it's weird that the IEEE choose to use the MSB
instead of the sign bit for NaN quiteness/signalness:

> It seems strange to me that the bit which signifies whether or not the NaN is signaling is the top bit of the mantissa rather than the sign bit; perhaps something about how floating point pipelines are implemented makes it less natural to use the sign bit to decide whether or not to raise a signal.
> -- <https://anniecherkaev.com/the-secret-life-of-nan>

I guess it might be something related to the CPU pipeline? I don't know yet.


### Equality of NaNs and Zeros.

The spec defined a comparison with NaNs to return an **unordered result**, that
means any comparison operation except `!=`, i.e. `>=, <=, >, <, =` between a
NaN and any other floating-point number would return `false`.

No surprised that most (if not every) language implemented such behaviours, e.g.
in JavaScript:

```js
NaN !== NaN   // true
NaN === NaN   // false
NaN >  1      // false
NaN <  1      // false
```

Position and negative zeros, however, are defined to be equal!

```js
+0 === -0  // true, using the traditional JS equality
Object.is(+0, -0)  // false, using the "SameValue" equality
```

In Cpp, we can tell them apart by looking at its sign bit:

```cpp
#include <cmath>   // signbit

cout << (+0.0f == -0.0f);     // 1
cout << std::signbit(-0.0f);  // 1
cout << std::signbit(+0.0f);  // 0
```




IEEE-754 64-bits Double-Precision Floats
----------------------------------------

- <https://en.wikipedia.org/wiki/Double-precision_floating-point_format>

Now, the 64-bit versions floating-point number, known as `double`, is just a
matter of scale:

```cpp
       (11 bits)            (52 bits)
sign   exponent             fraction
  0

 63   62 .... 52    51 ....................... 0
```


IEEE-754-2008 16-bits Short Floats
----------------------------------------

The 2008 edition of IEEE-754 also standardize the `short float`, which is
neither in C or C++ standard. Though compiler extension might include it.

It looks like:

```cpp
1 sign bit | 5 exponent bits | 10 fraction bits
S            E E E E E         M M M M M M M M M M
```



References
----------

- <https://en.wikipedia.org/wiki/Floating-point_arithmetic>
- <https://www3.ntu.edu.sg/home/ehchua/programming/java/datarepresentation.html>
