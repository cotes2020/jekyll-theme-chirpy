---
title: Integrity Attacks
# author: Grace JyL
date: 2021-04-05 11:11:11 -0400
description:
excerpt_separator:
categories: [15NetworkSec, NetworkAttacks]
tags: [NetworkSec]
math: true
# pin: true
toc: true
# image: https://wissenpress.files.wordpress.com/2019/01/a1bb1-16oVQ0409lk5n3C2ZPMg8Rg.png
---

# Integrity Attacks
- [Integrity Attacks](#integrity-attacks)
  - [Integrity Attacks](#integrity-attacks-1)
  - [Integrity attack methods](#integrity-attack-methods)
    - [man-in-the-middle attack](#man-in-the-middle-attack)
    - [Salami attack](#salami-attack)
    - [Data diddling](#data-diddling)
    - [Trust relationship exploitation](#trust-relationship-exploitation)
    - [Password attack](#password-attack)

---

## Integrity Attacks

Integrity attacks: `alter data (compromise the integrity of the data)`.

---

## Integrity attack methods

### man-in-the-middle attack

- a network stream is `intercepted, modified, and retransmitted`, and computer viruses, which modify critical system files so as to perform some malicious action and to replicate themselves.

### Salami attack

- `a collection of small attacks`, result in a larger attack when combined.
- Example:
  - attacker had a collection of stolen credit-card numbers
  - the attacker withdraw small amounts from each credit card (possibly unnoticed by the card holders).
  - Although each individual withdrawal was small, the combination of the multiple withdrawals results in a significant sum for the attacker.

### Data diddling

- `change data` before it is stored in a computing system.
- Malicious code in an input application or a virus could perform data diddling.
- Example:
  - a `virus, Trojan horse, worm` could be written to intercept keyboard input
  - while displaying the appropriate characters onscreen (so the user does not see an issue), manipulated characters could be entered into a database application or sent over a network.

- `virus`: is a piece of code (like a program or a script) that an end user executes.

- `worm` 蠕虫: can infect a system or propagate to other systems without any intervention from the end user.

- `Trojan horse`: is a program that appears to be for one purpose (like game), but secretly performs another task (like collecting a list of contacts from an end user’s e-mail program).

### Trust relationship exploitation

- Different devices in a network might have a trust relationship between themselves.
- example:
  - a certain host might be trusted to communicate through a firewall using specific ports, while other hosts are denied passage through the firewall using those same ports.
    - If an attacker were able to compromise the host that had a trust relationship with the firewall, then the attacker could use the compromised host to pass normally denied data through a firewall.
  - `web server --- database server` mutually trusting one another.
    - if an attacker gained control of the web server,
    - he might be able to leverage that trust relationship to compromise the database server.

### Password attack

- attempts to determine the password of a user.
  - Once the attacker gains the username and password credentials.
  - he can attempt to log into a system as that user and inherit that user’s set of permissions.

- Various approaches are available to determine passwords.
  - if a password is an arbitrary string of at least eight printable characters.
  - then the number of potential passwords is at least `94^8 = 6 095 689 385 410 816`, that is, at least 6 quadrillion.
  - Even if a computer could test one password every nanosecond, faster than any computer could, then it would take, on average, at least 3 million seconds to break one such password, that is, at least 1 month of nonstop attempts.

- Example:
  - **Trojan horse**:
    - a program that appears to be a useful application, but might capture a user’s password and then make it available to the attacker.
  - **Packet capture**:
    - a utility can capture packets seen on a PC’s NIC.
    - If the PC can see a copy of a plain-text password being sent over a link, the packet-capture utility can be used to glean the password.
  - **Keylogger**:
    - A program that runs in a computer’s background, and it logs keystrokes that a user makes.
    - after a user enters a password, the password is stored in the log created by the keylogger.
    - An attacker can then retrieve the log of keystrokes to determine the user’s password.
  - **Brute force Decryption Attack**:
    - This attack tries all possible password combinations until a match is made.
    - Example:
      - valid messages, English text of up to `t` characters
        - with the standard 8-bit ASCII encoding
          - n = 8:
          - a t-byte array.
          - the total number of possible `t`-byte arrays: `(2^8)^t = 2^n`.
          - a message is a binary string of length `n = 8t`
      - But, each character of English text carries about 1.25 bits of information
        - the number of `t`-byte arrays that correspond to English text: `(2^1.25)^t = 2^1.25t`.
        - the bit length n, the number of n-bit arrays corresponding to English text is approximately 2^0.16n.
      - 待细看
      - the brute-force attack might start with the letter a and go through the letter z.
      - Then, the letters aa through `zz` are attempted, until the password is determined.
      - Therefore, using a mixture of upper- and lowercase, in addition to special characters and numbers, can help mitigate a brute-force attack.
  - **Dictionary attack**:
    - Similar to a brute-force attack, multiple password guesses are attempted.
    - the dictionary attack is based on a dictionary of commonly used words, rather than the brute-force method of trying all possible combinations.
      - Example
        - English language, there are less than `50,000` common words, `1,000` common human first names, `1,000` typical pet names, and `10,000` common last names.
        - In addition, there are only `36,525` birthdays and anniversaries for almost all living humans on the planet, that is, everyone who is 100 years old or younger.
    - So an attacker can compile a dictionary of all these common passwords and have a file that has fewer than 100,000 entries.
    - If an attcker can try the words in his dictionary at the full speed of a modern computer, he can attack a password-protected object and break its protections in just a few minutes.
      - if a computer can test one password every millisecond, which is probably a gross overestimate for a standard computer with a clock speed of a gigahertz,
      - then it can complete the dictionary attack in 100 seconds, which is less than 2 minutes.
    - Picking a password that is not a common word helps mitigate a dictionary attack.
  - **Botnet 僵尸网络**:
    - A software robot is typically thought of as an application on a machine that can be controlled remotely (like a Trojan horse or a backdoor in a system).
    - If a collection of computers are infected with such software robots, called bots, this collection of computers is called a botnet (zombie).
    - Because of the potentially large size of a botnet, it might compromise the integrity of a large amount of data.
  - **Hijacking a session**: 劫持
    - An attacker could hijack a `TCP session`
    - Example:
      - by completing the third step in the three-way TCP handshake process between an authorized client and a protected server.
      - If an attacker successfully hijacked a session of an authorized device, he might be able to maliciously manipulate data on the protected server.
