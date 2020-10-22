---
title: 'How I learned programming - Part 2'
date: 2017-10-16T13:40:17+02:00
author: Wolfgang Ofner
categories: [Programming]
tags: ['C#', learning]
---
Welcome to part 2 of how I learned programming. In my last post, I wrote about how I started learning C# in university in Austria. You can find this post <a href="/how-i-learned-programming/" target="_blank" rel="noopener">here</a>.

In this post, I will continue my story about how I learned C#.

## Learning inheritance and more OOP concepts

After my second assignment, I had another two theory blocks. There I heard the first time about inheritance and abstract classes. With this new information, I still didn’t understand the concept to its fullest but it started to make more and more sense.

The next assignment was designed to understand classes and inheritance more. The task was to create a console application in which the user can create different objects. These objects were a circle, rectangle and diamond. Every object had a border color, body color, name, starting coordinates and specific attributes according to the object type. The circle had a radius, the rectangle length and width and the diamond had rows.

After the user entered his objects, these objects were painted. Additionally, the user could select one and move it around in the console. The user could switch between the objects and also change their Z- level. The Z-level decided which object was in front and covered the other ones.

The program also could display the entered attributes of the object plus calculated attributes like the area or diameter.

This assignment was my favorite of this semester and the source can be found on <a href="https://github.com/WolfgangOfner/Uni-GeometricObjects" target="_blank" rel="noopener">GitHub</a>.

## Learning about interfaces and improving the previous assignment

Before my fourth assignment, I had one theory class where I learned about interfaces. To be honest it took me way longer than this semester to understand interfaces but during the fourth assignment, I could at least see why they are useful.

The fourth assignment was an extension of the third one. The task was to implement interfaces to enable the user to use a color or monochrome renderer. Additionally, the user should be able to sort the objects by their area using the IComparer interface.

The source code to my solution can be found on <a href="https://github.com/WolfgangOfner/Uni-GeometricObjectsMonochrom" target="_blank" rel="noopener">GitHub</a>.

The last assignment was something about creating a file structure in the console using event.

## Network programming and source control

After the programming class, I had a two-week intense class called project week. This class was Monday to Friday from 8 to 6. The idea of this class was to see how it feels to work in a team on a “real” project. It started with gathering the requirements, designing the ideas, refining them with the stakeholders and then implementing it.

The project goal was to implement Pacman which can be played over LAN. The player started the application on his computer and other clients could join his game. After the game start, the map was split between all logged in clients. The player could always see the Pacman on his screen but the Pacman went from one computer to another when moving around the map. We set up a whole computer room and at the end of the project, we could watch the Pacman move from one screen to the next one. This was pretty cool to see what we achieved in such a short time.

Before we started this project we had a short introduction to network programming with TCP and UDP and also about source control. We only learned that TFS is a source control and with that, we can work as a team together. We have to check out a file if we want to work on it and then check it in after we are done. This was all we knew and all we needed at that time. You can find the source code <a href="https://github.com/WolfgangOfner/Uni-Pacman" target="_blank" rel="noopener">here</a>.

### Teach yourself

If you are interested in network programming you can start with reading the documentation to TCP <a href="https://msdn.microsoft.com/en-us/library/system.net.sockets.tcpclient(v=vs.110).aspx" target="_blank" rel="noopener">here</a>. Start with implementing a simple chat tool with one server and clients which can connect to the server. If one client sends a message, all connected clients get this message. As next step try to implement the same program using <a href="https://msdn.microsoft.com/en-us/library/system.net.sockets.udpclient(v=vs.110).aspx" target="_blank" rel="noopener">UDP</a>.

After the second assignment, I implemented my own Snake game in the console. It was actually easier than expected and only took me around two hours (obviously there is much to improve). It was a nice confidence boost to see that I could make my own game and also fun playing it afterwards. You can find the source code [here](https://github.com/WolfgangOfner/Snake).

I highly recommend to sit down at least once a week and implement something. It doesn&#8217;t even matter what it is, as long as it&#8217;s a challenge.

In the next post, I will talk about new experiences on how to get feedback from a tutor and what I learned in Canada and later how I got into web development.

Next: <a href="/how-i-learned-programming-part-3/" target="_blank" rel="noopener">Part 3</a>

Previous: <a href="/how-i-learned-programming-part-1/" target="_blank" rel="noopener">Part 1</a>