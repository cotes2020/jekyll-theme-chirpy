---
title: "Culturion : Quiz sur les régions françaises"
description: "Une application mobile pour apprendre tout en jouant avec la culture française."
date: 2021-08-15
categories: [Mobile]
media_subpath: /assets/img/posts/culturion
tags: [mobile, game, android, java]
lang: fr-FR
image: culturion_banner.png
---

> Le projet a été réalisé des années avant la rédaction de cet article, des détails et explications approfondies peuvent donc manquer.
{: .prompt-info }

Je vais vous présenter l'un de mes plus grands projets : **Culturion**. 

Vous pouvez découvrir l’application Android via une courte vidéo ci-dessous, utilisée comme bande-annonce sur YouTube et les réseaux sociaux :

{% include embed/youtube.html id='7KrYA6vUp8Q' %}

> Essayez l'application vous même depuis le Play Store : [https://play.google.com/store/apps/details?id=com.culturion.culturion](https://play.google.com/store/apps/details?id=com.culturion.culturion).
{: .prompt-info }

Le premier objectif de ce projet était de réutiliser les connaissances acquises lors du développement de ma première application Android : **Moovic**. 

# Moovic : un blind-test cinéma

Mon voyage a commencé avec **Moovic**, ma toute première application Android native développée avec Android Studio en 2018.

![Moovic](moovic.png){: w="700"}
_Moovic_


Le concept est un blind-test cinéma en trois manches distinctes :
  - Deviner un film à partir d’un **synopsis** 
  - Deviner un film à partir d’une **image**  
  - Deviner un film à partir de sa **bande-son originale**   

Chaque manche est chronométrée et la rapidité influence le score final. Vous pouviez également jouer entre amis dans des parties personnalisées conçues pour que les joueurs puissent se défier sur les mêmes manches de films.

Ce jeu a été publié sur le Play Store début 2019 mais n'est aujourd'hui plus disponible. Moovic était surtout une expérience préliminaire avant Culturion qui m'a permit d'avoir une meilleure maitrise des langages, outils et process de développement sur Android et l'IDE Android Studio. 

# Culturion : une application plus ambitieuse

Avec Culturion, je souhaitais mettre la barrière un peu plus haute, car j'avais une idée précise du prochain jeu que je souhaitais développer et je savais qu'une vraie demande existait.

Toujours intéressé par les jeux de quiz et de culture générale, en 2020, je commence donc à travailler sur Culturion : un quiz sur différents aspects du patrimoine français.

![Culturion](culturion.jpg){: w="700"}
_Culturion_

Le développement de ce jeu subira des hauts et des bas lors de pertes de motivation au cours de l'année, mais se concrétisera vraiment à la fin de l'été 2020 grâce à une nouvelle idée pour l'interface utilisateur. Le concept était d'afficher une carte de France vierge, et de la remplir au fur et à mesure que le joueur donne de bonnes réponses. Par exemple, en devinant dans quelle ville une photo a été prise, son emplacement est révélé sur la carte.

J'ai ensuite étendu ce concept à d'autres éléments liés à la France : numéro de département, spécialités gastronomiques et ville natale de célébrités. J'ai travaillé sur ce projet pendant plusieurs mois, passant du temps à développer l'application, mais aussi à rassembler toutes les données nécessaires (photos de villes, spécialités culinaires, ...).

Toutes ces données (+ de 500 questions différentes) ont ensuite dû être ajoutées à une base de données MySQL pour pouvoir être exploitées facilement par l'application.

J'ai également dû réfléchir à un nom et **Culturion** est venu simplement de la contraction de **Culture** et **Région** car le jeu repose sur les connaissances culturelles divisées  en régions françaises.

Une fois le jeu dans un état jouable, ma famille et mes amis m'ont beaucoup aidé à corriger les erreurs, les bugs et à améliorer certaines fonctionnalités. Devant les retours très positifs, j'ai publié l'application sur le Play Store en janvier 2021.

# Monétisation et marketing

Culturion a aussi été mon premier pas dans le monde de la publicité et de la promotion marketing.

Le modèle économique de mon application repose sur un jeu gratuit mais comportant de la publicité sous la forme de bannière, interstitiel et vidéo à récompenses pour obtenir des indices supplémentaires. Les publicités peuvent être supprimées par l'achat d'un **Pack Explorateur** (0.99€) ou **Pack VIP** (1.79€) donnant aussi accès à des légers avantages. 

Le vrai défi consistait à équilibrer la quantité de publicité et le prix de ces achats intégrés pour pouvoir financer la campagne publicitaire lancé en parallèle pour promouvoir le jeu. Cette campagne a été gérée avec Google Ads et a obtenu un taux de conversion de 7.10% pour environ 12 900 conversions.

Le résultat a dépassé mes attentes, car je suis fier que l'application ait été téléchargée par plus de **15 000 personnes** et notée **4.5/5** sur plus de 200 avis ! 

![Culturion](culturion_stats.png){: w="300"}
_Google Play Store statistiques_

**MISE À JOUR (2025-05-24)** : J'ai récemment mis à jour le SDK Android afin de pouvoir compiler Culturion pour tous les nouveaux smartphones Android et le republier sur le Play Store. Ce fut un travail fastidieux à cause de nombreuses dépendances obsolètes et des milliers d'avertissements après 6 ans laissé à l'abandon.

<br>
<p align="center">
  <a href="https://play.google.com/store/apps/details?id=com.culturion.culturion"><img src="playstore.png" alt="Culturion"/></a>
</p>