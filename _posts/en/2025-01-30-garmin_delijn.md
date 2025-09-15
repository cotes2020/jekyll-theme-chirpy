---
title: "Garmin Bus Tracker"
description: "A simple bus tracker widget compatible with Garmin watches to follow all DeLijn buses in Flanders."
date: 2025-01-30
categories: [Embedded]
tags: [c, network, garmin, watch]
media_subpath: /assets/img/posts/garminDeLijn
lang: en
image:
    path: garmin_screen.png
---

A year after my last **Bus Tracker** project for buses in Toulouse, I moved to Belgium and decided to create a similar tool but for **Garmin** watches.

> If you want to learn more about my **Bus Tracker** project, you can read more about it [here]({% post_url en/2023-02-10-bustracker %}).
{: .prompt-tip }

I personally wear a **Forerunner 245** and have always wanted to know exactly when the bus is coming. So I created this app to track all **DeLijn** buses in Flanders and display the next bus’s arrival time.

## DeLijn Bus Data API
The first step was to get access to the **DeLijn API** to receive all the real-time bus information.
Fortunately, it is free to use and only requires to create an account on the [De Lijn Open Data portal](https://data.delijn.be/).
Once my keys were in hand, DeLijn provided three different APIs:
- **GTFS Static** : Standard API to get static public transport info
- **GTFS Realtime** : Standard API to get real-time public transport info
- **Open Data Services** : Custom API to get real-time and static public transport info

### GTFS

**General Transit Feed Specification (GTFS)** is a standard format for public transport schedules created by Google in 2005. Initially created to incorporate transit data into Google Maps, it has since been adopted by most other navigation services (Apple Maps, Moovit, ...) and public transit services.
It allows transit agencies to publish their schedule, route, and stop data in a format that navigation services can easily integrate.

A **GTFS feed** is a collection of files that describe the public transport network, including routes, stops, and schedules. However a GTFS feed is not enough to get real-time information because it contains only the static/planned schedules and not the real-time delays or trip updates.

To have this additional real-time information, you need to fetch another feed, **GTFS-RT** (GTFS Realtime), which is an extension of GTFS. This feed will contain only relative delays for each trip and not the absolute arrival time. So you need to combine the results of both **GTFS APIs** to compute the absolute arrival/departure time of public transport vehicles.

> See [GTFS official website](https://gtfs.org) for more information.
{: .prompt-info }

### Open Data Services
The **Open Data Services** API is a custom API by DeLijn that provides real-time and static public transport information. It is not standard but offers the same information as the **GTFS APIs** in a **JSON** format and does not require to make additional requests to fetch the real-time information.
> See [DeLijn Data website](https://data.delijn.be/product#product=5978abf6e8b4390cc83196ad) for more information.
{: .prompt-info }

On an embedded environment with limited memory and processing power, the **Open Data Services** API seemed to be the best choice. The response being a **JSON** object, it can be easily parsed and does not require to navigate between different files and execute different HTTP requests like the **GTFS** APIs.

## Garmin SDK
To develop an application for a **Garmin** product, you have to use the **Garmin** [Connect IQ SDK](https://developer.garmin.com/connect-iq/overview/).
The SDK allows to build native applications, widgets and data fields for all **Garmin** smartwatches.

### Monkey C

**Garmin** developed its own programming language called **Monkey C** to use the SDK. The syntax is derived from multiple languages (C, C#, Java, Javascript, ...) and is quite easy to understand.

![Monkey C](monkeyc.png){: w="300"}
_Monkey C_

> See [Garmin Monkey C](https://developer.garmin.com/connect-iq/monkey-c/) for more information.
{: .prompt-info }

One of the first things I made is to install the official [Monkey C language support extension for VS Code](https://marketplace.visualstudio.com/items?itemName=garmin.monkey-c).

As well as offering syntax highlighting and code completion, it also eases the interaction with the **Connect IQ SDK** by providing a few useful commands (create a new project, build, open samples, open SDK manager, ...).

### Emulator

The SDK manager also comes with a device emulator to directly run the application from the computer.
It is quite handy to quickly run the application without having to install it on the watch or to be able to test it on different watches.

![Garmin Emulator](emulator.png){: w="300"}
_Garmin Emulator_

## Application

### Architecture

The application is quite simple, it fetches the time of arrival of the next bus from the **DeLijn Open Data Services API** and displays a countdown on the watch.
The countdown is updated every second and a new API request is made every minute or, if the refresh button is pressed, to correct the countdown if the bus is delayed or advanced. This way, the information displayed is always up to date.

![Architecture Diagram](schema.png){: w="300"}
_Architecture Diagram_

### Settings

In the settings, accessible from the **Connect IQ** application, you can select which bus stop and bus lines you want to track. You should also set your own **DeLijn API** key, so you don't have to worry about rate limits.

![Settings](settings.png){: w="200"}
_Settings_

The API request interval can be changed as well.

### Interface
The countdown (in minutes) to the next bus is displayed. The interface can show up to two bus stop information, one on each line.
The countdown color indicates if the estimated bus arrival time (based on the GPS position) is on schedule, late, or early compared to the static timetable information :
- <span style="color:purple;font-weight:bold">Purple</span> : Bus is estimated to arrive earlier than expected
- <span style="color:green;font-weight:bold">Green</span>: Bus is on time.
- <span style="color:red;font-weight:bold">Red</span> : Bus is estimated to arrive later than expected.

![Interface](interface.png){: w="200"}
_Interface_

> The code is available on Github here : [https://github.com/nicopaulb/Garmin-DeLijn-Bus-Tracker](https://github.com/nicopaulb/Garmin-DeLijn-Bus-Tracker).
{: .prompt-tip }

## Garmin Connect IQ Store
Publishing the application was quite straightforward if you compare it to a lot of other stores and platforms. Simply create an account, upload the application, update the store details and you're done.
> The **Garmin** application is available on the **Connect IQ** Store here : [https://apps.garmin.com/fr-FR/apps/1d2b5826-ae2e-4bb9-a6e7-76e3e6b1ef5a](https://apps.garmin.com/fr-FR/apps/1d2b5826-ae2e-4bb9-a6e7-76e3e6b1ef5a).
{: .prompt-tip }
