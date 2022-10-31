---
title: Cryptography - White box Cryptography
date: 2022-01-18 11:11:11 -0400
categories: [13Cryptography]
tags: [cryptography]
toc: true
image:
---

- [Cryptography - White box Cryptography](#cryptography---white-box-cryptography)
  - [Kerckhoffs's Principle](#kerckhoffss-principle)
  - [basic](#basic)
  - [attacks](#attacks)


[link](https://www.youtube.com/watch?v=A9md7ONv7tI)

---


# Cryptography - White box Cryptography

> In penetration testing, white-box testing is where the testers (or attackers) have access to the source code and internal workings of the system.

White-box cryptography:
- attackers have access to the compiled code where the keys exist.
- The difficult problem that it aims to solve is how to keep those keys safe while using them in execution.
- remove the distinction between keys and crypto algorithm code


## Kerckhoffs's Principle

Kerckhoffs's principle
- revolutionized the way we think about cryptography
- we should allow the attacker to know everything about a crypto implementation, except the key.
- If a cryptosystem can stand up to that level of scrutiny it will be the better for it.

White-box crypto
- kind-of takes this one step further.
- we technically give the attacker access to the key, we just hide/encrypt it well enough that they can't find it.

---


## basic

In order to secure a program using white-box cryptography, we assume the attacker has complete access to the system. This includes:
- Access to executable binary
- Access to execution memory
- CPU call intercepts


to successfully hide the keys given this scenario, take the steps for white-box a block cipher:
- Partial Evaluation: When performing an operation, we alter the operation based on the key. For example, in the substitution phase of a block cipher, we would change the lookup table to be dependent on the key. Note that if someone were to see this table, they could derive the key (solved in step 3)

![Screen Shot 2022-01-20 at 08.13.23](https://i.imgur.com/3iuvE4n.png)

- Tabularizing: Transform all other operations to also use lookup tables. This is possible because lookup tables can describe any function.
- Randomization and Delinearization: We create an encoded chain of lookup tables that has the same functionality as the original chain, but hides the key. Now, using this new chain, we have an obfuscated algorithm. For reading on the details of this operation, see here.

- encoding:

![Screen Shot 2022-01-20 at 08.14.38](https://i.imgur.com/1VRv4qd.png)



![Screen Shot 2022-01-20 at 08.07.52](https://i.imgur.com/bmwDRDn.png)



![Screen Shot 2022-01-20 at 08.10.30](https://i.imgur.com/EcEHXvZ.png)


![Screen Shot 2022-01-20 at 08.11.09](https://i.imgur.com/l3vhDOH.png)


## attacks


- Fault injection attacks: modifying the white box execution so the output cipher text is wrong, not expected
-
-




.
