---
layout: post
title:  "Installing Taillight"
categories: [Empennage, ~rudder]
tags: [taillight, avionics]
minutes: 60
mermaid: true
---

## LET THERE BE LIGHT!

In the first week of the new year I am jumping between a few different projects, and today I installed
the tail light, and the rudder is officially complete.

![light](/assets/img/20240111/tail_light.jpg)

I've done soldering at Arudino level before, but this is the first time I use crimper to connect wires. I
wanted to make sure I absolutely do everything in the right way, so this is a bit of learning process
for me.

## Tools

For wire connection, after some research, I decided to crimp rather than soldering. I choose this way because
it's the standard practice in automotive and aviation. And I really like the consistency from using
crimpers.

Here are the tools involved.

* Automatic wire stripper
* Wire crimper
* Heat shrink crimp connector
* Extra heat shrink

I spent some time and spare wires to learn how to use wire stripper and crimpers, then moved onto the real thing.

## Connecting the wires

The light I am using is Aveo Minimax Ariel. It comes with 4 wires: black, red, blue, yellow.

However the 4 core wire Sling has is: white, green, orange, blue.

I am  not sure if there is a standard color code mapping, and I couldn't find any. So decided to match them in this way:

| Light wire | Sling wire |
| -----------|------------|
| Black      |  White     |
| Blue       |  Blue      |
| Yellow     |  Orange    |
| Red        |  Green     |


## Things that could have gone better

I installed the rudder tip and the wire that came with Sling before I installed the light. It was not esay to work
the wires when half of it is already in the rudder. I had to be really careful to not pull it out.

If I had to do it again, I would have waited to install rudder tip until the light is installed. It would have provided
more space to work with.

I also drilled the hole to feed wire on rudder tip too small. When I measured the hole I only make it slightly bigger than
the wire that came with Sling. I forgot that the wire would become thicker, and it no longer fit the hole. So I had to drill
another hole right next to the original wire feeder hole, and connect the two holes together to provide more space. Again, if
I connected the wires before installing rudder tip this would have been unnecessary.
