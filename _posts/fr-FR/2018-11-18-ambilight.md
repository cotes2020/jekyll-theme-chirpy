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

Ma première expérience personnelle dans le domaine de l'électronique embarquée était un écran LED ambilight. L'Ambilight est un système d'éclairage ambiant qui génère de la lumière derrière l'écran en fonction de l'affichage. L'objectif était d'utiliser mon écran traditionnel et d'y intégrer cette technologie. 

Afin d'accomplir cela, j'ai employé une bande de LED divisée en 4 parties et soudée ensemble afin de la fixer autour de l'arrière de mon écran. Par la suite, j'ai utilisé un Arduino Nano connecté directement par USB à mon ordinateur pour contrôler la lumière LED et capturer la couleur affichée actuellement par le PC. 
Pour alimenter toutes les LEDs, j'ai réutilisé un vieux chargeur de téléphone 5V dont le connecteur était endommagé et que j'ai retiré afin de ne conserver que le fil d'alimentation. 

Malgré un léger retard visible, j'étais plutôt content du résultat. Une brève démonstration est disponible ci-dessous.

{% include embed/youtube.html id='vdvZXc05rZM' %}
