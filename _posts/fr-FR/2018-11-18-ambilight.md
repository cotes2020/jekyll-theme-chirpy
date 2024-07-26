---
title: "Écran ambilight : Ajouter un rétro-éclairage à un écran"
description: "Une solution bon marché pour ajouter une fonction ambilight à mon écran."
date: 2018-11-18
categories: [Embedded]
media_subpath: /assets/img/posts/ambilight
tags: [arduino, C] 
lang: fr-FR
---

> Le projet a été réalisé des années avant la rédaction de cet article, il peut donc manquer de détails et d'explications approfondies.
{: .prompt-info }

Mon premier projet personnel d'électronique embarquée était un écran LED ambilight. Ambilight est un éclairage ambiant qui crée de la lumière derrière l'écran en fonction de ce qui y est affiché. Le but était d'utiliser mon écran classique et d'y ajouter cette technologie. 

Pour ce faire, j'ai utilisé une bande de LED coupée en 4 parties et soudée ensemble pour être collée tout autour de l'arrière de mon écran. Ensuite, la lumière LED a été contrôlée par un Arduino Nano directement connecté par USB à mon ordinateur pour capturer la couleur actuellement affichée par le PC.
Pour alimenter toutes les LEDs, j'ai réutilisé un vieux chargeur de téléphone 5V dont le connecteur était cassé et que j'ai retiré pour ne garder que le fil d'alimentation. 

Bien qu'un léger retard soit perceptible, j'étais assez satisfait du résultat. Vous pouvez voir une petite démo ci-dessous.

{% include embed/youtube.html id='vdvZXc05rZM' %}

