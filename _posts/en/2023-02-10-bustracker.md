---
title: 'Bus Tracker : Real-time information display'
description: A ESP32 and a TFT screen to never be late again.
date: 2023-02-10T00:00:00.000Z
categories:
  - Embedded
media_subpath: /assets/img/posts/bustracker
tags:
  - arduino
  - esp
  - network
  - c
lang: en
image:
  path: bustracker_pcb.png
---

> The project was realized years before the redaction of this post, so it may lack details and in-depth explanation.
{: .prompt-info }

The inspiration for this project came from my frequent use of public transportation to get to work. Every day, I only had to take a 5 minute ride to work because I lived close to it. However, the bus had the annoying habit of being either late or early, but rarely on time.

I found myself constantly checking the Tisséo (Toulouse's public transportation company) application for real-time updates on the next bus route. A cumbersome routine to start the day, and sometimes the app would even crash.

That's why I wanted a compact display that shows when the next bus is coming, similar to the ones found at bus stops.

![Bus real-time information display](bustracker_example.jpg){: w="400" h="150"}
_Bus real-time information display_

It had to be smaller to fit my apartment and budget, but I would still be able add some extra features !

## Obtaining real-time bus data

First, I needed to ensure that I could access all real-time information about Toulouse's public transportation. So I searched the internet and discovered an official API provided by Tisseo. 

To use it, a private key was required, so I politely asked Tisseo if I could get one for free. Fortunately, they agreed, and I was able to try out the API a few days later.

Despite sparse documentation, using the API was straightforward: a simple HTTP GET request with the key, line ID, stop ID, and filters returns the needed JSON data, including the upcoming bus' waiting time.

So for example by sending this GET request (I censored my key, so it will not work for you): 

`https://api.tisseo.fr/v2/stops_schedules.json?timetableByArea=1&lineId=line:170&stopPointId=stop_point:SP_1423&number=2&key=fffffff-ffff-ffff-ffff-ffffffffffff`

The response is the following JSON data:

```json
{
  "departures": {
    "stopAreas": [
      {
        "cityId": "admin:fr:31446",
        "cityName": "RAMONVILLE-SAINT-AGNE",
        "id": "stop_area:SA_1813",
        "name": "Ramonville Sud",
        "schedules": [
          {
            "destination": {
              "cityId": "admin:fr:31446",
              "cityName": "RAMONVILLE-SAINT-AGNE",
              "id": "stop_area:SA_213",
              "name": "Ramonville",
              "way": "backward"
            },
            "journeys": [
              {
                "dateTime": "2023-06-13 22:12:00",
                "realTime": "1",
                "waiting_time": "00:16:23"
              },
              {
                "dateTime": "2023-06-13 22:42:00",
                "realTime": "1",
                "waiting_time": "00:46:23"
              }
            ],
            "line": {
              "bgXmlColor": "#ff671b",
              "color": "(255,103,27)",
              "fgXmlColor": "#ffffff",
              "id": "line:170",
              "name": "Ramonville / Castanet-Tolosan",
              "network": "Tisséo",
              "shortName": "L6",
              "style": "orange"
            },
            "stop": {
              "handicappedCompliance": "1",
              "id": "stop_point:SP_1423",
              "name": "Ramonville Sud",
              "operatorCode": "20831"
            }
          }
        ],
        "uniqueStopId": "stop_point:SP_1423"
      }
    ]
  },
  "expirationDate": "2023-06-13 21:56"
}
```

The next step was to use this API from a microcontroller and only show the 'waiting_time' value from the JSON on the screen.

## Choosing the hardware

### Microcontroller

My choice of a microcontroller for this project was based on two criteria :
- Be able to connect to internet without needing an extra module
- Having an easy framework to use to get a prototype up and running quickly


I chose an ESP32 (or ESP8266) for its built-in Wi-Fi and compatibility with the Arduino framework, perfect for quick prototyping. I already had a Wemos D1 Mini Pro (ESP8266) on hand, so I put it to use.

![ESP32](bustracker_WEMOSD1.jpg){: w="150" h="50"}
_ESP32_

### Display

The information needed to be displayed on a screen as well. The display needed to be small, readable, and be powered by the ESP32's internal voltage regulator. The microcontroller should also be able to control it via I2C or SPI.

I found two good options for my limited budget: a mini 128x64 OLED and a 128x160 TFT screen.

![128x64 OLED screen](bustracker_OLED.jpg){: w="150" h="50"}
_128x64 OLED screen_
![128x160 TFT screen](bustracker_TFT.jpg){: w="150" h="50"}
_128x160 TFT screen_

While the OLED screen was smaller, I hoped that the higher contrast would allow me to see it from a greater distance. 
Because I couldn't decide, I bought both to try them out and eventually chose the TFT display.

### Button

A button was required to interact with the screen and turn it off when not in use. Because I wanted to use as few components as possible, I only used one rotary encoder with a push button. I was able to navigate the menu by rotating the encoder and then validate my selection by pressing it.

![Rotary encoder button](bustracker_BUTTON.jpg){: w="150" h="50"}
_Rotary encoder button_

### Bonus

I thought another useful feature would be to play a small ringtone when it's time to leave for the bus. So, I added a piezo speaker to my shopping list.

## Writing the code

As I previously stated, the goal of this project was to quickly develop a working prototype. Because a large number of Arduino libraries were already available for the screen and rotary encoder, I chose to create the ESP32 software using the Arduino framework.

To improve code readability, I separated each functionality into its respective class :
- **Controller** : Decode rotary encoder signals and generate callbacks for user events (button pressed, rotated, etc.).
- **Display** : Communicate with the TFT display and update its graphical interface.
- **TimeTable** : Connect to a WIFI network and perform HTTP request to obtain the bus real-time information.
- **Alarm** : Using the piezo speaker, play sound effects for both the graphical interface and the alarm.

Each class has an initialization function called during setup and a tick function (which updates internal state) called from the main loop.

> The code is available on Github here : [https://github.com/nicopaulb/BusTracker/tree/main](https://github.com/nicopaulb/BusTracker/tree/main).
{: .prompt-tip }

## Building a prototype

Connecting the peripherals to the ESP was straightforward :
- The TFT displays communication pins to the SPI ports (SCK, MOSI, MISO, CS) and the RST and DC pins to some digital outputs. I also added a transistor between the backlight power of the screen (LED pin) and the VCC of the ESP to be able to switch off the screen when needed via one of the output ports.
- The rotary encoder pins (CLK/DT for the encoder and SW for the button) to some input port.
- The piezo speaker to a PWM port to play different tonality by varying the duty cycle.

![Bus Tracker schematic](bustracker_schematic.png){: w="500" h="500"}
_Bus Tracker schematic_

For development, I simply connected the components with a breadboard and some jumper wires.

![Bus Tracker protoype](bustracker_photo.jpg){: w="500" h="500"}
_Bus Tracker prototype_

I was pleased with the results and used it in this state for several weeks. 

Then I wanted to use perfboard to make the whole montage smaller and more attractive. I couldn't find a suitable tool to plan the component positioning and tracks for a perfboard, so I used standard PCB design software. It wasn't practical, but I was able to draw a preliminary version of the schematic.

![Bus Tracker perfboard schematic](bustracker_pcb.png){: w="500" h="500"}
_Bus Tracker perfboard schematic_

Unfortunately, I got distracted by another project and never fully finished the final build. Still, it served me well for months and I never missed a bus again!
