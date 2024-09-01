---
title: "Retro handheld II : Connecting an Xbox controller to a STM32 microcontroller"
date: 2024-06-09
categories: [Embedded, Retro handheld]
tags: [ble, reverse, st, C] 
media_subpath: /assets/img/posts/miniconsole
lang: en
---

The first goal of this project is to connect the Xbox controller to my NUCLEO-WB55 board. The two communicate via BLE, but an appropriate driver should be written for the microcontroller to realize the connection handshake and understand the controller inputs. 

So before starting programming on my board, I need to know how the controller communicate with the Xbox driver for my PC. To do that I am helped by Wireshark, a network packet analysis software which can also capture and dissect BLE packets.

# The controller input

By connecting the controller to my PC and intercepting the BLE packets between the two devices, I can see all the frame sent by the controller when I realize certain action (joystick moved, button pressed, trigger pushed, ...).

For example, if I press the button 'B', the controller send the following frame : 

![BLE packet received when button B is pressed](xbox_ble_packet.png){: w="1000" h="700"}
_BLE packet received when button B is pressed_

The most important part of this packet is the BLE attribute value, containing information about all the controller inputs.
So, I filled an Excel document with the attribute value intercepted for each performed action and split it by byte. Note that to make it easier to reverse the protocol, I only do one action at a time. 
This way I am able to identify the purpose of each byte field in the BLE attribute value. 
Once all the packets captured, I deduced the following table :

![Table of packets for each action](xbox_parse_table.png){: w="1000" h="700"}
_Table of packets for each action_

# Writing a Wireshark plugin

To check I didn't do any mistake and to better visualize the packet received, I developed a Wireshark dissector. Dissectors are meant to analyze some part of a packet's data and I choose to integrate one into Wireshark via a plugin.

I did that in 3 steps : 
- I downloaded the Wireshark source code and installed all the compilation tools
- I wrote my plugin code in C by following the Wireshark documentation on dissectors (https://github.com/wireshark/wireshark/blob/master/doc/README.dissector)
- I recompiled the application and the plugin together

The Xbox controller protocol is then automatically detected when a BLE packet coming from the controller is received. You can find an example on the screenshot below :

TODO screenshot wireshark

My Xbox controller dissector for Wireshark is available on Github : 

# The controller handshake

The controller needs to be able to connect to the board and the board needs to tell the controller the connection is sucessfull. To do that I also sniffed the BLE packet echanged when the controller is connecting to my PC.

***

#### Read more here about this project :
- [ ] [Retro handheld I : A minimalist game console]({% post_url en/2024-06-08-miniconsole_part1 %})
- [x] [Retro handheld II : Connecting an Xbox controller to a STM32 microcontroller]({% post_url en/2024-06-09-miniconsole_part2 %})
- [ ] And soon more...