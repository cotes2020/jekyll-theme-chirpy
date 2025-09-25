---
title: "Bus Tracker : Afficheur temps réel pour les transports publics"
description: "Un ESP32 et un écran TFT pour ne plus jamais être en retard."
date: 2023-02-10
categories: [Embedded]
media_subpath: /assets/img/posts/bustracker
tags: [arduino, esp, network, c]
lang: fr-FR
image:
  path: bustracker_pcb.png
---

> Le projet a été réalisé des années avant la rédaction de cet article, des détails et explications approfondies peuvent donc manquer.
{: .prompt-info }

L'idée de ce projet est née de mon utilisation régulière des **transports publics** pour me rendre au travail. Habitant près de mon travail, je n'avais besoin que de 5 minutes de trajet pour aller travailler tous les jours. Cependant, le bus avait cette fâcheuse habitude d'être soit en retard, soit en avance, mais rarement à l'heure.

Pour ne jamais arriver en retard ou attendre trop longtemps le bus, je n'avais pas d'autre choix que d'utiliser l'application Tisseo (la société de transport en commun de Toulouse) pour avoir une mise à jour en **temps réel** du prochain passage du bus.

Avoir toujours à regarder mon téléphone le matin pour savoir quand je dois partir pour arriver au travail à l'heure n'était pas pratique et c'était sans compter les bugs qui rendaient parfois l'application inutilisable.

J’ai donc décidé de créer un affichage, à l’image de ceux que l’on voit aux arrêts de bus, mais en version compacte pour mon appartement afin d’avoir l’heure d’arrivée du prochain bus sans sortir mon téléphone.

![Affichage d'informations en temps réel sur les bus](bustracker_example.jpg){: w="400" h="150"}
_Affichage de l'information en temps réel sur les bus_

Forcément, il sera plus petit pour s'adapter à mon appartement et à mon budget, mais rien ne m'empêche aussi de pouvoir ajouter quelques fonctionnalités supplémentaires !

## Obtenir les données des bus en temps réel

Tout d'abord, je devais m'assurer qu'il était possible d'accéder à toutes les **informations en temps réel** sur les transports publics à Toulouse. J'ai donc cherché sur internet et j'ai trouvé une **API officielle** fournie par **Tisseo**.

Pour l'utiliser, une **clé privée** était nécessaire, je me suis donc directement adressé à Tisseo et je leur ai demandé s'il était possible d'en obtenir une gratuitement. Heureusement, ils ont accepté et j'ai pu commencer à tester l'API quelques jours plus tard.

Malgré une documentation limitée, l'utilisation de l'API était relativement simple. En envoyant simplement une **requête HTTP GET** avec la clé, la ligne de bus, l'identifiant de l'arrêt de bus et quelques paramètres supplémentaires pour filtrer les informations inutiles, j'ai pu obtenir tout ce que je voulais.

Par exemple, en envoyant cette requête GET (j'ai censuré ma clé, inutile d'essayer chez vous) :

`https://api.tisseo.fr/v2/stops_schedules.json?timetableByArea=1&lineId=line:170&stopPointId=stop_point:SP_1423&number=2&key=fffffff-ffff-ffff-ffff-ffffffffffff`

On obtient la réponse JSON ci-dessous :

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
L'étape suivante consistait à utiliser cette API à partir d'un microcontrôleur et à afficher uniquement la valeur `waiting_time` de la réponse sur un écran.

## Choix du matériel

### Microcontrôleur

Mon choix d'un microcontrôleur pour ce projet a été basé sur deux critères :
- Pouvoir se **connecter à Internet** sans avoir besoin d'un module supplémentaire.
- Avoir un **framework facile** à utiliser pour mettre en place un prototype rapidement.

**L'ESP32/ESP8266** répond à tous ces critères grâce à sa fonctionnalité WIFI intégrée et la possibilité d'utiliser le **framework Arduino**.
De plus, j'avais déjà des cartes **WEMOS D1 Mini Pro** (ESP8266) achetées en vrac dans le passé, il était donc temps de les utiliser.

![ESP32](bustracker_WEMOSD1.jpg){: w="150" h="50"}
_ESP32_

### Écran

Ensuite, pour afficher le prochain arrêt de bus, un **écran** était également important. Il devait être ni trop petit pour pouvoir lire rapidement les informations qu'il contenait, mais aussi ni trop grand, car je voulais qu'il soit alimenté par le régulateur de tension interne de l'ESP32. Le microcontrôleur devait également être capable de le piloter via **I2C** ou **SPI**.

J'ai trouvé deux candidats intéressants pour mon petit budget : un mini **écran OLED 128x64** et un **écran TFT 128x160**.

![écran OLED 128x64](bustracker_OLED.jpg){: w="150" h="50"}
_Écran OLED 128x64_
![Écran TFT 128x160](bustracker_TFT.jpg){: w="150" h="50"}
_Écran TFT 128x160_

Bien que l'écran OLED soit plus petit, j'ai pensé que les meilleurs contrastes pourrait aider à voir l'écran d'un peu plus loin.
Comme je n'arrivais pas à prendre une décision, j'ai acheté les deux pour les essayer, et j'ai finalement opté pour l'écran TFT.

### Bouton

Un bouton était nécessaire pour interagir avec l'écran et pour l'éteindre lorsqu'il n'est pas utilisé. Comme je voulais avoir le moins de composants possible, je n'ai utilisé qu'un **encodeur rotatif** avec un bouton poussoir. De cette manière, il est possible de naviguer dans le menu en faisant tourner l'encodeur et valider son choix en appuyant dessus.

![Bouton encodeur rotatif](bustracker_BUTTON.jpg){: w="150" h="50"}
_Bouton encodeur rotatif_

### Bonus

J'ai pensé qu'une autre fonctionnalité intéressante serait de jouer une petite sonnerie lorsqu'il est l'heure d'aller au travail. J'ai donc ajouté un **haut-parleur piézo** à ma liste d'achats.

## Écrire le code

Comme je l'ai dit précédemment, le but de ce projet était d'avoir rapidement un prototype fonctionnel. Comme de nombreuses bibliothèques Arduino étaient déjà disponibles pour l'écran et l'encodeur rotatif, j'ai choisi de développer le logiciel de l'ESP32 avec le framework Arduino.

J'ai séparé chaque fonctionnalité dans une classe pour améliorer la lisibilité du code :
- **Controller** : Décoder le signal de l'encodeur rotatif et créer une callback pour les événements utilisateur (bouton pressé, rotation, ...).
- **Display** : Communique avec l'écran TFT et met à jour l'interface graphique.
- **TimeTable** : Se connecte à un réseau WIFI et utilise l'API avec une requête HTTP pour obtenir les informations en temps réel du bus.
- **Alarm** : Joue des effets sonores pour l'interface graphique et pour l'alarme grâce au haut-parleur piézo.

Chaque classe expose une fonction d'initialisation appelée lors de la configuration et une fonction tick (mise à jour de l'état interne) appelée depuis la boucle principale.

> Le code est disponible sur Github ici : [https://github.com/nicopaulb/BusTracker/tree/main](https://github.com/nicopaulb/BusTracker/tree/main).
{: .prompt-tip }

## Construction d'un prototype

La connexion des périphériques à l'ESP a été simple :
- Les broches de communication de **l'écran TFT** vers les ports SPI (SCK, MOSI, MISO, CS) et les broches RST et DC vers certaines sorties numériques. J'ai également ajouté un transistor entre l'alimentation du rétro-éclairage de l'écran (broche LED) et le VCC de l'ESP pour pouvoir éteindre l'écran si nécessaire via l'un des ports de sortie.
- Les pins de **l'encodeur rotatif** (CLK/DT pour l'encodeur et SW pour le bouton) vers un port d'entrée.
- Le **haut-parleur piézo** vers un port PWM pour jouer des tonalités différentes en variant le duty-cycle.

![Bus Tracker schematic](bustracker_schematic.png){: w="500" h="500"}
_Schéma du Bus Tracker_

Pour les besoins du développement, j'ai simplement utilisé une **breadboard** et quelques fils de connexion pour relier les composants.

![Bus Tracker protoype](bustracker_photo.jpg){: w="500" h="500"}
_Prototype de Bus Tracker_

J'étais assez satisfait du résultat et je l'ai utilisé dans cet état pendant plusieurs semaines.

Ensuite, j'ai voulu utiliser une perfboard pour rendre l'ensemble du montage plus petit et joli. Je n'ai pas trouvé d'outil approprié pour planifier le positionnement des composants et les pistes pour une **perfboard**, j'ai donc utilisé un **logiciel de conception de PCB**. Ce n'était pas très pratique, mais j'ai réussi à dessiner une première version du schéma.

![Bus Tracker perfboard schematic](bustracker_pcb.png){: w="500" h="500"}
_Schéma de la carte à perfusion du Bus Tracker_

Malheureusement, ce projet n'a pas vraiment de conclusion, car j'ai été distrait par un autre projet et je ne l'ai jamais terminé.
Néanmoins, c'était un projet intéressant et je l'ai utilisé tous les jours pendant plusieurs mois sans jamais être en retard au travail !
