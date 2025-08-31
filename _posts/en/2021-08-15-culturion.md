---
title: 'Culturion : The quiz about France'
description: A successful mobile application to learn and play with French culture.
date: 2021-08-15T00:00:00.000Z
categories:
  - Mobile
media_subpath: /assets/img/posts/culturion
tags:
  - mobile
  - game
  - android
  - java
lang: en
image: culturion_banner.png
---

> The project was realized years before the redaction of this post, so it may lack details and in-depth explanation.
{: .prompt-info }

I'm going to introduce you to one of my biggest projects: **Culturion**. 

You can find a short video describing this Android app below :

{% include embed/youtube.html id='7KrYA6vUp8Q' %}

This video was used as an advertisement on YouTube and social media.

> Try the app for yourself from the Play Store: [https://play.google.com/store/apps/details?id=com.culturion.culturion](https://play.google.com/store/apps/details?id=com.culturion.culturion).
{:.prompt-info}

## Moovic : a movie blind test

Moovic was my very first native Android app, developed with Android Studio in 2018. 

![Moovic](moovic.png){: w="700"}
_Moovic_

The concept was a movie blind-test featuring three fast-paced rounds:
- Guess a movie from its **synopsis**
- Guess a movie from an **image**
- Guess a movie from its **original soundtrack**

Each round was timed, and faster answers meant higher scores. It even allowed multiplayer with friends by creating custom games designed so that several players could challenge each other on the same movie rounds.

This game was released on the Play Store in early 2019 but is no longer available. However, Moovic was primarily a preliminary experiment before Culturion, which allowed me to gain a better understanding of the languages, tools, and development processes on Android and the Android Studio IDE.

## Culturion : a more ambitious application

With Culturion, I wanted to raise the bar a little higher, because I had a clear idea of the next game I wanted to develop and I knew there was a real demand for it.

Still interested in quiz and general knowledge games, in 2020 I started working on Culturion: a quiz on different aspects of French heritage.

![Culturion](culturion.jpg){: w="700"}
_Culturion_

The development of this game would experience ups and downs as I lost motivation throughout the year, but it really took shape at the end of summer 2020 thanks to a new idea for the user interface. The concept was to display a blank map of France and fill it in as the player gave correct answers. For example, by guessing which city a photo was taken in, its location would be revealed on the map.

I then extended this concept to other elements related to France: department numbers, gastronomic specialties, and the birthplaces of celebrities. I worked on this project for several months, spending time developing the application, but also gathering all the necessary data (photos of cities, culinary specialties, etc.). All this data (more than 500 different questions) then had to be added to a MySQL database so that it could be easily used by the app.

I also had to think of a name, and **Culturion** simply came from the contraction of **Culture** and **Region** because the game is based on cultural knowledge divided into French regions.

Once the game was playable, my family and friends helped me a lot to fix errors and bugs and improve certain features. Given the very positive feedback, I published the app on the Play Store in January 2021.

## Monetization and marketing

Culturion was also my first step into the world of advertising and marketing promotion.

The economic model of my app is based on a free game but with advertising in the form of banners, interstitials, and rewarded videos to obtain additional clues. Ads can be removed by purchasing an **Explorer Pack** (€0.99) or **VIP Pack** (€1.79), which also gives access to minor benefits. 

The real challenge was to balance the amount of advertising and the price of these in-app purchases in order to finance the advertising campaign launched in parallel to promote the game. This campaign was managed with Google Ads and reported a conversion rate of 7.10% for approximately 12,900 conversions.

The result exceeded my expectations, as I am proud that the app has been downloaded by over 15,000 people and rated 4.5/5 based on over 200 reviews! 

![Culturion](culturion_stats.png){: w="300"}
_Google Play Store metrics_

**UPDATE (2025-05-24)** : I recently updated the Android SDK to be able to compile Culturion for all recent Android smartphones and republish it on the Play Store. It was a tedious effort, having to fix outdated and conflicting dependencies and thousands of warnings after 6 years.

<br>
<p align="center">
  <a href="https://play.google.com/store/apps/details?id=com.culturion.culturion"><img src="playstore.png" alt="Culturion"/></a>
</p>
