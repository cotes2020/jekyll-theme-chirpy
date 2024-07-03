---
title: "Retro handheld I : A minimalist game console"
description: "A STM32 board equiped with a mini OLED screen and a Xbox Controller to recreate some old memories..."
date: 2024-06-08
categories: [Embedded, Retro handheld]
tags: [ble, reverse, st, C]     # TAG names should always be lowercase
lang: en
---

I was brainstorming for a project idea to test my just bought NUCLEO-WB55 evaluation board and its BLE features when my eyes landed on my Xbox controller lying on my desk. This controller can connect remotely to a device via a proprietary 5 GHz protocol requiring a USB adapter or by the famous standardized protocol : BLE (Bluetooth Low Energy). 

By continuing my productive brainstorming session, I remembered having a small unused OLED display screen in my stock, originally bought for my Bus Tracker project. I thought it could be a nice occasion to also test this screen.

> If you want to learn more about my **Bus Tracker** project, you can read more about it [here]({% post_url en/2023-02-10-bustracker %}).
{: .prompt-tip }

To resume I have : 
- a gaming controller
- a mini-OLED screen 
- a ST development board waiting to be flashed with its next software

I was just missing an idea of what the final objective of this project could be, until the moment I had a flashback by looking at my little OLED screen. It reminded me of
the mini handheld game console which was popular when I was a child. It was a cheap electronics toy, which was sometimes given away at McDonald, with just one or two buttons, a bad monochrome screen and a battery lasting only a few days.

Now the goal of my next project was clear : create a similar handheld game console based on a STM32 microcontroller but with a nice colorful screen and playable with a wireless Xbox Controller !

***

#### Read more about this project :
- [x] [Retro handheld I : A minimalist game console]({% post_url en/2024-06-08-miniconsole_part1 %})
- [ ] [Retro handheld II : Connecting an Xbox controller to a STM32 microcontroller]({% post_url en/2024-06-09-miniconsole_part2 %})
- [ ] And soon more...

