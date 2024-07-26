---
title: "Bus Tracker : A real-time information display"
description: "A ESP32 and a TFT screen to never be late again."
date: 2023-02-10
categories: [Embedded]
media_subpath: /assets/img/posts/bustracker
tags: [arduino, esp, network, C] 
lang: en
---

> The project was realized years before the redaction of this post, so it may lack details and in-depth explanation.
{: .prompt-info }

The idea behind this project came from my regular use of public transport to get to work. Since I lived near my job, I only needed to take a 5-minute ride to go to work every day. However, the bus had this annoying habit of either being late or in advance but rarely on time.

So to never arrive late or wait too long for the bus, I had no choice but to use the Tisseo (Toulouse's public transport company) application to have a real-time update of the next bus passage.

Always having to look at my phone in the morning to know when I need to go to work was not practical and that's without mentioning the bugs that sometimes made the application unusable. 

That's why I wanted a display showing when the next bus is coming, exactly like the one often at a bus stop.

![Bus real-time information display](bustracker_example.jpg){: w="400" h="150"}
_Bus real-time information display_

It had to be smaller to fit my apartment and my budget, but I can also add some extra functionalities!

## Obtaining real-time bus data

First, to achieve that, I had to be sure it was possible to access all the real-time information about the public transport in Toulouse. So I searched on internet and found an official API provided by Tisseo. 

To use it a private key was needed, so I asked Tisseo nicely if it was possible to get one for free. Thankfully they agreed, and I was able to play with the API a few days later.

Although documentation was limited, using the API was relatively straightforward. By just sending a GET HTTP request with the key, bus line, bus stop ID, and some additional parameters to filter out useless information, I was able to get everything I wanted.

So for example by sending this GET request (I censored my key, so it will not work for you): 

`https://api.tisseo.fr/v2/stops_schedules.json?timetableByArea=1&lineId=line:170&stopPointId=stop_point:SP_1423&number=2&key=fffffff-ffff-ffff-ffff-ffffffffffff`

The answer is the below JSON data :

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
The next step was to use this API from a microcontroller and display only the `waiting_time` value from the JSON on a screen.

## Choosing the hardware

### Microcontroller
My choice of a microcontroller for this project was based on two criteria :
- Be able to connect to internet without needing an extra module
- Having an easy framework to use to get a prototype up and running quickly

The ESP32/ESP8266 meets all the criteria with its built-in WIFI functionality and Arduino framework. 
Also, I already some WEMOS D1 Mini Pro boards (ESP8266) bought in bulk in the past, so it was time to use them.

![ESP32](bustracker_WEMOSD1.jpg){: w="150" h="50"}
_ESP32_

### Screen
Then to display the next bus stop, a screen was also important. It had to be not too small to be able to read quickly the information on it but not too large because I wanted it to be powered through the ESP32 internal voltage regulator. The microcontroller should also be able to control it via I2C or SPI.

I found two nice candidates for my small budget: a mini 128x64 OLED and a 128x160 TFT screen.

![128x64 OLED screen](bustracker_OLED.jpg){: w="150" h="50"}
_128x64 OLED screen_
![128x160 TFT screen](bustracker_TFT.jpg){: w="150" h="50"}
_128x160 TFT screen_

While the OLED screen was smaller, I thought the better contrast could help to see the screen from a little further. 
Because I couldn't make a decision I just bought the two to try them, and finally went for the TFT display.

### Button
A button was required to interact with the screen and to switch it off when not used. Because I wanted to have as few components as possible, I used only one rotary encoder with a push button. This way, I was able to navigate through the menu by rotating the encoder and validate my choice by pressing it.

![Rotary encoder button](bustracker_BUTTON.jpg){: w="150" h="50"}
_Rotary encoder button_

### Bonus
I thought another nice functionality, will be to play a small ringtone when it is time to go to work. So I also added a piezo speaker to my shopping list.

## Writing the code
As I said previously, the goal of this project was to have a working prototype quickly. Because a lot of Arduino library was already available for the screen and the rotary encoder, I chose to develop the ESP32 software with the Arduino framework.

I separated each functionality in its class to improve the code readability :
- **Controller** : Decode signal from the rotary encoder and create callback for user events (button pressed, rotated, ...).
- **Display** : Communicate with the TFT display and update the graphical interface.
- **TimeTable** : Connect to a WIFI network and perform HTTP request to get the bus real-time information.
- **Alarm** : Play sound effects for the graphical interface and the alarm thanks to the piezo speaker.

Each class exposes an initialization function called during setup and a tick function (updating internal state) called from the main loop. 

> The code is available on Github here : [https://github.com/Fantomos/BusTracker/tree/main](https://github.com/Fantomos/BusTracker/tree/main).
{: .prompt-tip }

## Building a prototype
Connecting the peripherals to the ESP was straightforward :
- The TFT displays communication pins to the SPI ports (SCK, MOSI, MISO, CS) and the RST and DC pins to some digital outputs. I also added a transistor between the backlight power of the screen (LED pin) and the VCC of the ESP to be able to switch off the screen when needed via one of the output ports.
- The rotary encoder pins (CLK/DT for the encoder and SW for the button) to some input port.
- The piezo speaker to a PWM port to play different tonality by varying the duty cycle.

![Bus Tracker schematic](bustracker_schematic.png){: w="500" h="500"}
_Bus Tracker schematic_

For development purposes, I just used a breadboard and some jumper wires to connect the components. 

![Bus Tracker protoype](bustracker_photo.jpg){: w="500" h="500"}
_Bus Tracker prototype_

I was pretty satisfied with the results and used it in this state for several weeks. 

Then I wanted to use a perfboard to make the whole montage smaller and prettier. I didn't find an appropriate tool to plan the component's positioning and tracks for a perfboard so I just used a standard PCB design software. It was not practical, but I managed to draw some first version of the schematic.

![Bus Tracker perfboard schematic](bustracker_pcb.png){: w="500" h="500"}
_Bus Tracker perfboard schematic_

Sadly, the conclusion of this project is a little rough, because I got distracted by another project and never finished it. 
Nevertheless, it was an interesting project and I used it every day for several months without ever being late for work!