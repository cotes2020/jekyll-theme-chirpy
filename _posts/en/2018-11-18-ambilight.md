---
title: 'Ambilight monitor : Adding backlight to a screen'
description: A low-cost solution to add ambilight technology to a monitor.
date: 2018-11-18
categories:
  - Embedded
media_subpath: /assets/img/posts/ambilight
tags:
  - arduino
  - c
lang: en
image: ambilight.jpg
---

> The project was realized years before the redaction of this post, so it may lack details and in-depth explanation.
{: .prompt-info }

Ambilight is a type of ambient lighting that adjusts the amount of light behind the monitor according on what is shown on it. The idea was to employ this technology in conjunction with my standard monitor. 

I accomplished this by cutting an LED strip into four pieces, soldering them together, and then adhering them all around the rear of my screen. Then, in order to record the color that was now displayed on the monitor, an Arduino Nano that was directly connected to my computer via USB controlled the LED light.
I used an old 5V phone charger that had a broken connector, which I took out and kept just the power wire for powering all the LEDs. 

I was rather pleased with the outcome, even if there was a slight delay and some color lag.
Below is a brief demonstration :

{% include embed/youtube.html id='vdvZXc05rZM' %}
