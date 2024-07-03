---
title: "Retro handheld I : Une mini-console de jeu"
description: "Une carte STM32 équipée d'un mini écran OLED et d'une manette Xbox pour recréer de vieux souvenirs..."
date: 2024-06-08
categories: [Embedded, Retro handheld]
tags: [ble, reverse, st, C]     # TAG names should always be lowercase
lang: fr-FR
---

Je cherchais une idée de projet pour tester la carte d'évaluation NUCLEO-WB55 que je venais d'acheter lorsque mes yeux se sont posés sur ma manette Xbox posée sur mon bureau. Cette manette peut se connecter à distance à un appareil via un protocole propriétaire 5 GHz nécessitant un adaptateur USB ou par le fameux protocole standardisé : BLE (Bluetooth Low Energy). 

En continuant ma séance de brainstorming, je me suis souvenu que j'avais un petit écran OLED inutilisé dans mon stock, acheté à l'origine pour mon projet Bus Tracker. J'ai pensé que ce serait une bonne occasion de tester également cet écran.

> Si vous voulez en savoir plus sur mon projet **Bus Tracker**, vous pouvez le lire [ici]({% post_url fr-FR/2023-02-10-bustracker %}).
{ : .prompt-tip }

Pour résumer, j'ai donc : 
- une manette de jeu
- un écran mini-OLED 
- une carte de développement ST (NUCLEO-WB55) qui attend d'être flashée avec son prochain logiciel

Il me manquait seulement un objectif pour ce projet, jusqu'au moment où j'ai eu un flashback en regardant mon petit écran OLED. Il me rappelait
la mini-console de jeu portable qui était populaire lorsque j'étais enfant. Ce jouet électronique bon marché qui était parfois offert à McDonald avec seulement un ou deux boutons, un écran monochrome de mauvaise qualité et une batterie qui ne durait que quelques jours.

L'objectif de mon prochain projet était maintenant clair : créer une console de jeu portable similaire basée sur un microcontrôleur STM32 mais avec un bel écran coloré et jouable avec une manette Xbox sans fil !

***

#### En savoir plus sur ce projet :
- [x] [Retro handheld I : Une mini-console de jeu]({% post_url en/2024-06-08-miniconsole_part1 %})
- [ ] [Retro handheld II : Connecting an Xbox controller to a STM32 microcontroller]({% post_url en/2024-06-09-miniconsole_part2 %})
- [ ] Et bientôt plus...

