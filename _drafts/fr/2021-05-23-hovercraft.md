---
title: "Aéroglisseur radiocommandé : Une course qui ne manque pas d'air"
description: "Un aéroglisseur commandé par un smartphone."
date: 2018-11-18
categories: [Embedded]
media_subpath: /assets/img/posts/hovercraft
tags: [C, PIC] 
lang: fr-FR
---

> Le projet a été réalisé des années avant la rédaction de cet article, il peut donc manquer de détails et d'explications approfondies.
{: .prompt-info }

Ce projet a été réalisé dans le cadre de mes études en électronique et système embarqué à l'INSA Strasbourg. L'objectif était de concevoir un aéroglisseur télécommandé afin de participer à une course contre la montre. Pour cela nous devions réaliser un aéroglisseur, c’est-à-dire un véhicule sur coussin d’air qui sera commandé par un téléphone via bluetooth.

# La structure mécanique


# Les parties électroniques

## Motorisation

## Alimentation

## Commande


# Les logicielles embarqués

## dsPIC

## PIC

## La télécommande

# Une petite démo




## Mécanique

Commençons par la partie mécanique de l’aéroglisseur. Cette partie n’est pas le centre du projet, mais elle reste importante, car elle va être le support de toute la partie électroniques. Ainsi les contraintes mécaniques sont les suivantes :
• Les dimensions de la savonnette ne doivent pas excéder 30 cm en hauteur, 25 cm en largeur et 35 cm en longueur
• Un espace pour un système de coupure générale facilement accessible doit être prévu
• La structure doit pouvoir accueillir toutes l’électronique et une caméra type GoPro
• L’hélice permettant la propulsion doit être protégée et ne présenter aucun danger pour l’environnement extérieur
• La structure sera réalisée intégralement en carton plume pour être le plus léger possible

## Electronique

Les contraintes electroniques se divisent en trois parties : les contraintes de la motorisation, les contraintes de l’alimentation et les contraintes de la commande.

### Motorisation

Le déplacement de l’aéroglisseur se fait grâce à une hélice entraînée par un moteur qui permet la propulsion et la création du coussin d’air. Les contraintes liées à cette motorisation sont les suivantes :
• La motorisation sera assurée par un moteur brushless triphasé (Propdrive 35-30)
• Le moteur brushless sera alimenté par un onduleur triphasé en pont constitué de six transistors MOSFET canal N BSC0902NS et des drivers de demi-pont MIC4104
• L’onduleur triphasé sera commandé par un circuit programmable DsPIC
• L’onduleur triphasé devra avoir une fréquence de fonctionnement d’au moins 20 Khz afin de ne pas être audible
• Le moteur brushless ne disposant pas de capteurs de positions, une commande par retour de force électromotrice devra être implémentée

### Alimentation

L’alimentation du système doit être conçus de manière à alimenter correctement tous les composants électroniques. Les contraintes liées à l’alimentation sont les suivantes :
• La source d’alimentation sera une unique batterie LiPo 3S 2200 mAh
• Les autres tensions nécessaires à l’alimentation des composants seront générées par des convertisseurs DC/DC indépendants
∗ La conversion 11.1V vers 5V sera assuré par le régulateur à découpage LM22672 (convertisseur Buck)
∗ La conversion 5V vers 3V sera assuré par le régulateur linéaire MCP1826
• Un interrupteur général marche arrêt doit pouvoir couper l’alimentation du moteur et de l’électronique
• Un système de coupure général permettant de couper l’alimentation en cas d’urgence devra également être câblé (bouton d’urgence, long fil facilement déconnectable, ...)

### Commande

Afin de piloter l’aéroglisseur un circuit de commande et une télécommande doivent être réalisés. Les contraintes liées à la commande sont les suivantes :
- L’aéroglisseur sera télécommandé avec un smartphone via Bluetooth
- L’application sur smartphone est libre et peut inclure des joysticks, jauges, boutons ou une gestion de l’inclinaison
- La communication entre le circuit programmable de commande et la télécommande sera faite en Bluetooth à l’aide d’un module MLT-BT05
- Les protocoles de communication entre le circuit programmable de commande (récepteur) et le circuit programmable de l’onduleur devront respecter les normes de modélisme
- Les commandes devront être proportionnelles avec au minimum 5 niveaux de direction et propulsion
- Un système de sécurité sera mis en œuvre pour arrêter le moteur si l’aéroglisseur ne reçoit aucune information de la télécommande pendant plus de 200 ms


## Analyse fonctionnelle


{% include embed/youtube.html id='vdvZXc05rZM' %}

