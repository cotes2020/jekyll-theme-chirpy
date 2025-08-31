---
title: 'Écran ambilight : Rétro-éclairage DIY'
description: Une solution bon marché pour ajouter une fonction ambilight à un écran.
date: 2018-11-18T00:00:00.000Z
categories:
  - Embedded
media_subpath: /assets/img/posts/ambilight
tags:
  - arduino
  - c
lang: fr-FR
image: ambilight.jpg
---

> Le projet a été réalisé des années avant la rédaction de cet article, il peut donc manquer de détails et d'explications approfondies.
{: .prompt-info }

Ma première expérience personnelle dans le domaine de l'électronique embarquée était un écran LED ambilight. L'Ambilight est un système d'éclairage ambiant qui génère de la lumière diffusée derrière l'écran en fonction de l'affichage. L'objectif était d'utiliser mon écran traditionnel et d'y intégrer cette technologie. 

Pour cela, j’ai découpé une bande de LED en quatre segments, que j’ai soudés puis collés à l’arrière de l’écran. Ensuite, un Arduino Nano, connecté à mon ordinateur via USB, pilotait ces LEDs selon les couleurs affichées sur l’écran. 

Pour alimenter toutes les LEDs, j'ai réutilisé un vieux chargeur de téléphone 5V dont le connecteur était endommagé et que j'ai retiré afin de ne conserver que le fil d'alimentation. 

Le résultat m’a satisfait, malgré un léger décalage et un peu de latence dans les couleurs.Vous trouverez ci-dessous une démonstration rapide :

{% include embed/youtube.html id='vdvZXc05rZM' %}
