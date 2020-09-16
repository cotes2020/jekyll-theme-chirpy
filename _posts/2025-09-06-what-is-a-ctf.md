---
layout: post
title: 'What is a CTF?'
categories:
- Security
tags:
- ctf
- introduction

---


## Introduction

---

I am sure you have heard me talk about "participating in a CTF" and wondered to yourself: What is a CTF and how does it work?

**CTF** stands for **Capture The Flag**, where the flag in this context is usually a string of characters that is predefined by the contest creators. For example, an example of a flag could be the password of the admin user on a machine, or a hidden piece of text in an image. You most probably will have a rough idea of what the flag is or what it looks like, otherwise it's like looking for a needle in a haystack.

Your goal as a hacker is to find these flags and submit them for points. If the target is a website, your goal is to hack into it or break it in some way that allows you to reveal the flag. If the target is a system, your goal is also to hack into it and retrieve a flag. Sometimes the target is an encrypted piece of data, or just a sound file. Using cryptographic or forensic techniques, you have to reverse the encryption or encoding and retrieve the flag hidden inside. Get the point?

## Capturing a flag

---

Let's take a look at a simple challenge and how a flag looks like. In this example, the target is a simple webpage and we are asked to figure out how we can view a page that should only be viewed by an admin user.
