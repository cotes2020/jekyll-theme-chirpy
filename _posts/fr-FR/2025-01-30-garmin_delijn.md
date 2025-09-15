---
title: "Garmin Bus Tracker"
description: "Un widget compatible avec les montres Garmin pour suivre tous les bus DeLijn en Flandre."
date: 2025-01-30
categories: [Embedded]
tags : [c, network, garmin, watch]
media_subpath : /assets/img/posts/garminDeLijn
lang : fr-FR
image :
    path : garmin_screen.png
---

Un an après mon dernier projet **Bus Tracker** pour les bus à Toulouse, j'ai déménagé en Belgique et j'ai décidé de créer un outil similaire, mais pour les montres **Garmin**.

> Si vous voulez en savoir plus sur mon projet **Bus Tracker**, clicker [ici]({% post_url fr/2023-02-10-bustracker %}).
{: .prompt-tip }

Je porte personnellement une Forunner 245 et j'ai toujours voulu pouvoir savoir exactement quand le bus allait venir me chercher.
J'ai donc créé cette application pour suivre tous les bus **DeLijn** en Flandre et afficher l'heure d'arrivée du prochain bus.

## API de données de bus DeLijn

La première étape a consisté à obtenir l'accès à l'**API DeLijn** pour recevoir toutes les informations en temps réel sur les bus.
Heureusement, son utilisation est gratuite et il suffit de créer un compte sur le [portail De Lijn Open Data](https://data.delijn.be/).

Une fois mes clés en main, DeLijn m'a fourni trois API différentes :
- **GTFS Static** : API standard pour obtenir des informations statiques sur les transports publics
- **GTFS Realtime** : API standard pour obtenir des informations en temps réel sur les transports publics
- **Open Data Services** : API personnalisée pour obtenir des informations en temps réel et statiques sur les transports publics

D'après cette brève description, vous pouvez déduire que pour obtenir des informations en temps réel, il est possible d'utiliser soit l'API **GTFS Realtime**, soit l'API **Open Data Services**.
Mais avant de vous dire laquelle j'ai choisi d'utiliser, examinons la différence entre les deux API.

### GTFS

**General Transit Feed Specification (GTFS)** est un format standard pour les horaires des transports publics créé par Google en 2005. Initialement créé pour intégrer les données de transport dans Google Maps, il a depuis été adopté par la plupart des autres services de navigation (Apple Maps, Moovit, ...) et services de transport public.
Il permet aux agences de transport de publier leurs horaires, leurs itinéraires et leurs données d'arrêts dans un format que les services de navigation peuvent facilement intégrer.

Un flux **GTFS** est un ensemble de fichiers qui décrivent le réseau de transports publics, y compris les itinéraires, les arrêts et les horaires. Cependant, un flux GTFS ne suffit pas pour obtenir des informations en temps réel, car il ne contient que les horaires statiques/planifiés et non les retards en temps réel ou les mises à jour des trajets.

Pour obtenir ces informations supplémentaires en temps réel, vous devez récupérer un autre flux, **GTFS-RT** (**GTFS Realtime**), qui est une extension de GTFS. Ce flux ne contient que les retards relatifs pour chaque trajet et non l'heure d'arrivée absolue. Vous devez donc combiner les résultats des deux API GTFS pour calculer l'heure d'arrivée/de départ absolue des véhicules de transport public.

> Pour plus d'informations, consultez le [site web officiel de GTFS](https://gtfs.org).
{: .prompt-info }

### Services de données ouvertes
L'API **Open Data Services** est une API personnalisée de DeLijn qui fournit des informations en temps réel et statiques sur les transports publics. Elle n'est pas standard, mais offre les mêmes informations que les API **GTFS** au format **JSON** et ne nécessite pas de requêtes supplémentaires pour récupérer les informations en temps réel.

> Pour plus d'informations, consultez le [site web DeLijn Data](https://data.delijn.be/product#product=5978abf6e8b4390cc83196ad).
{: .prompt-info }

Dans un environnement intégré avec une mémoire et une puissance de traitement limitées, l'API **Open Data Services** m'a semblé être le meilleur choix. La réponse étant un objet **JSON**, elle peut être facilement analysée et ne nécessite pas de naviguer entre différents fichiers et d'exécuter différentes requêtes HTTP comme les API **GTFS**.

## SDK Garmin
Pour développer une application pour un produit **Garmin**, vous devez utiliser le [**Connect IQ SDK**](https://developer.garmin.com/connect-iq/overview/) de Garmin.

Le SDK permet de créer des applications natives, des widgets et des champs de données pour toutes les montres connectées **Garmin**.

### Monkey C

**Garmin** a développé son propre langage de programmation appelé **Monkey C** pour utiliser le SDK. La syntaxe est dérivée de plusieurs langages (C, C#, Java, Javascript, ...) et est assez facile à comprendre.

![Monkey C](monkeyc.png){: w="300"}
_Monkey C_

> Pour plus d'informations, consultez [**Garmin Monkey C**](https://developer.garmin.com/connect-iq/monkey-c/).
{: .prompt-info }

L'une des premières choses que j'ai faites a été d'installer l'extension officielle [Monkey C language support extension for VS Code](https://marketplace.visualstudio.com/items?itemName=garmin.monkey-c).

En plus d'offrir la coloration syntaxique et la complétion de code, elle facilite également l'interaction avec le **Connect IQ SDK** en fournissant quelques commandes utiles (créer un nouveau projet, compiler, ouvrir des exemples, ouvrir le gestionnaire SDK, ...).

### Émulateur

Le gestionnaire SDK est également fourni avec un émulateur de montre **Garmin** permettant d'exécuter directement l'application depuis l'ordinateur.
Il est très pratique pour exécuter rapidement l'application sans avoir à l'installer sur la montre ou pour pouvoir la tester sur différentes montres.

![Émulateur Garmin](emulator.png){: w="300"}
_Émulateur Garmin_

## Application

### Architecture

L'application est assez simple, elle récupère l'heure d'arrivée du prochain bus à partir de l'**API DeLijn Open Data Services** et affiche un compte à rebours sur la montre.
Le compte à rebours est mis à jour toutes les secondes et une nouvelle requête API est effectuée toutes les minutes ou, si l'on appuie sur le bouton d'actualisation, pour corriger le compte à rebours si le bus est retardé ou avancé. De cette manière, les informations affichées sont toujours à jour.

![Diagramme d'architecture](schema.png){: w="300"}
_Diagramme d'architecture_

### Paramètres

Dans les paramètres, accessibles à partir de l'application **Connect IQ**, vous pouvez sélectionner les arrêts et les lignes de bus que vous souhaitez suivre. Vous devez également définir votre propre clé **API DeLijn**, afin de ne pas avoir à vous soucier des limites de requetes journalières.

![Paramètres](settings.png){: w="200"}
_Paramètres_

L'intervalle des requêtes API peut également être modifié.

### Interface
Le compte à rebours (en minutes) jusqu'au prochain bus est affiché. L'interface peut afficher jusqu'à deux informations sur les arrêts de bus, un sur chaque ligne.

La couleur du compte à rebours indique si l'heure d'arrivée estimée du bus (basée sur la position GPS) est à l'heure, en retard ou en avance par rapport aux informations statiques du planning :
- <span style="color:purple;font-weight:bold">Mauve</span> : Le bus devrait arriver plus tôt que prévu.
- <span style="color:green;font-weight:bold">Vert</span> : Le bus est à l'heure.
- <span style="color:red;font-weight:bold">Red</span> : Le bus devrait arriver plus tard que prévu.

![Interface](interface.png){: w="200"}
_Interface_

> Le code est disponible sur Github ici : [https://github.com/nicopaulb/Garmin-DeLijn-Bus-Tracker](https://github.com/nicopaulb/Garmin-DeLijn-Bus-Tracker).
{: .prompt-tip }

## Boutique Garmin Connect IQ

La publication de l'application a été assez simple si vous la comparez à beaucoup d'autres plateformes. Il suffit de créer un compte, de télécharger l'application, de mettre à jour les détails de la boutique et le tour est joué.

> L'application **Garmin** est disponible sur le **Connect IQ** Store ici : [https://apps.garmin.com/fr-FR/apps/1d2b5826-ae2e-4bb9-a6e7-76e3e6b1ef5a](https://apps.garmin.com/fr-FR/apps/1d2b5826-ae2e-4bb9-a6e7-76e3e6b1ef5a).
{: .prompt-tip }
