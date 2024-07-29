---
title: "Ambilight monitor : Adding backlight to a screen"
description: "A cheap solution to add ambilight technology to a monitor."
date: 2018-11-18
categories: [Embedded]
media_subpath: /assets/img/posts/ambilight
tags: [arduino, C] 
lang: en
---

> The project was realized years before the redaction of this post, so it may lack details and in-depth explanation.
{: .prompt-info }

My first personal embedded eletrical project was an LED ambilight monitor. Ambilight is an ambient lighting that creates light behind the monitor based on what is displayed on it. The goal was to use my regular monitor and this technology to it. 

To do that I used LED strip cutted in 4 parts and soldered together to be glued all around the back of my screen. Then the LED light was controlled by an Arduino Nano directly connected by USB to my computer to capture the color currently showed on the monitor.
For powering all the LED I reused an old 5V phone charger with a broken connector which I removed and kept only the power wire. 

Event though a little delay is noticeable and the color lag a little behind, I was pretty satisfied with the result. You can see a little demo below.

{% include embed/youtube.html id='vdvZXc05rZM' %}

