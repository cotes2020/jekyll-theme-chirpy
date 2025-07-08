---
title: "Avis Canal+ : Extension Firefox"
description: "Ajoute les évaluations IMDb, Rotten Tomatoes et Allociné sur la plateforme Canal+"
date: 2025-06-24
categories: [Website]
tags: [website, firefox, extension, javascript]
media_subpath: /assets/img/posts/canalratings
lang: fr-FR
image: 
    path: cover.png
---

Trouver quoi regarder sur une plateforme de streaming comme Canal+ peut parfois être déroutant. Avec autant de films et de séries disponibles, il n’est pas toujours facile de savoir ce qui vaut vraiment le temps investi. Or le système de notation intégré à la plateforme Canal+ n’est pas très fiable, je me retrouvais donc souvent à vérifier les notes manuellement sur des sites comme IMDb ou Rotten Tomatoes.  

Pour simplifier ce processus, j’ai décidé de créer une petite extension Firefox qui récupère automatiquement les notes de plusieurs sources et les affiche directement sur Canal+. Cela permet de repérer rapidement les meilleurs films sans quitter la plateforme.

> Ce projet a été écrit en JavaScript vanilla et le code source est disponible ici : [https://github.com/nicopaulb/Canal-Ratings](https://github.com/nicopaulb/Canal-Ratings).  
{: .prompt-tip }

## Sources des notes
La première étape a été d’identifier quelles notes seraient les plus utiles à afficher. D’après mon expérience, j’ai sélectionné trois plateformes bien connues :  

- **IMDb** – Probablement la base de données de films la plus reconnue à l’international. Son système de notation repose sur les votes de millions d’utilisateurs à travers le monde, offrant un score global et communautaire.  
- **Rotten Tomatoes** – Un agrégateur de notes qui distingue critiques et spectateurs. Le Tomatometer reflète les avis des critiques professionnels, tandis que l’Audience Score montre l’opinion des spectateurs. Cette double perspective est très utile.  
- **Allociné** – Le plus grand site français dédié au cinéma, combinant un large catalogue de films, des critiques éditoriales et les notes du public. Étant donné que Canal+ propose un catalogue français conséquent, Allociné était un choix naturel.  

Ces trois sources offrent un bon équilibre entre **opinions internationales**, **critiques professionnelles** et **références locales françaises**.  

### API

Pour récupérer automatiquement les notes d’un film donné, chaque site sélectionné propose une API. Cependant, pour un petit projet personnel comme celui-ci, les API officielles sont souvent trop coûteuses ou restreintes.  
Par exemple, accéder aux données IMDb via leur API officielle est possible via AWS Data Exchange, avec des prix à partir de 150 000 $ par an. Clairement, cela dépasse largement le budget d’un projet personnel.

Face à ces contraintes, se reposer sur les API officielles n’était pas une option viable. J’ai donc implémenté une **solution de scraping** pour récupérer directement les notes depuis les pages web.

### Scraping
La solution la plus évidente serait de scraper les pages web de chaque site individuellement. Cependant, cette approche est chronophage et difficile à maintenir, car chaque source possède sa propre structure et ses mesures anti-scraping, comme les limites de requêtes et les pages qui changent fréquemment.  

J’ai réalisé qu’une alternative plus simple existait : lorsque l’on recherche un film sur un moteur comme Google, l’aperçu inclut souvent les notes de plusieurs sources, toutes réunies.  

![Résultats de recherche affichant les notes des films](search_results.png)
_Résultat Google affichant les notes_

Au lieu d’analyser chaque site séparément, j’ai donc choisi d’analyser les résultats des moteurs de recherche pour récupérer simultanément les notes IMDb, Rotten Tomatoes et Allociné.  

Pour améliorer la fiabilité, cette approche a été appliquée sur plusieurs **moteurs de recherche** : Google, Bing et Yahoo. Cela fournit une sauvegarde si une recherche ou un scraping échoue et augmente les chances de récupérer les notes de toutes les sources, certains moteurs n’affichant qu’une partie des notes.

### Gestion des limites de requêtes  

Bien sûr, les moteurs de recherche ont également des **mesures anti-scraping**, comme la détection de trop nombreuses requêtes provenant d’une même IP. Pour un film, ce n’est pas un problème, mais lorsqu’il faut récupérer les notes de centaines de films en même temps (par exemple, pour afficher les notes sur toutes les vignettes), cela devient compliqué.  

![Google Captcha](captcha.png)
_Google Captcha_

La première solution consiste à mettre en place un **système de cache**. Les notes des films ne changent pas fréquemment, donc une fois récupérées, elles peuvent être stockées et réutilisées sans envoyer de nouvelles requêtes à chaque fois. La durée du cache est configurable (et peut même être désactivée, bien que non recommandée), et est par défaut de 7 jours (configurable dans [Pages d’options](#page-doptions)). Toutes les données mises en cache sont stockées dans le stockage de l’extension Firefox, garantissant leur persistance même après le redémarrage du navigateur ou de l’ordinateur.  

Le cache seul ne suffit pas. Lorsqu’il faut récupérer beaucoup de notes pour de nouveaux films, j’ai ajouté un petit délai entre les requêtes et empêché l’envoi de multiples requêtes en parallèle. Cela évite de déclencher les mécanismes anti-bot des moteurs de recherche. Pour gérer cela, j’ai implémenté un **système de file d’attente** : toutes les requêtes sortantes sont stockées dans une file, et les requêtes vers le même moteur sont espacées selon un intervalle configurable. Cela assure un fonctionnement fluide tout en minimisant le risque de blocage.

J’ai également envisagé l’utilisation de **proxies** pour envoyer les requêtes depuis différentes adresses IP, mais cela s’est avéré difficile à mettre en œuvre dans une extension Firefox, probablement pour des raisons de sécurité. Heureusement, le système actuel fonctionne bien dans la plupart des cas, et avec plusieurs moteurs de recherche, si une requête échoue sur un moteur, les autres servent de secours fiables.

## Personnalisation de la page Canal+ 

Une fois que l’extension peut récupérer automatiquement les notes d’un film, l’étape suivante consistait à intégrer cette fonctionnalité directement dans l’interface Canal+. Deux tâches principales devaient être réalisées :  

1. **Extraire le nom du film ou de la série** depuis le HTML de la page, afin que l’extension sache quel titre rechercher.  
2. **Afficher les notes récupérées** sur la même page, à la fois sur la vue détaillée et sur les vignettes, pour une expérience utilisateur fluide.  

La première étape a été d’implémenter ces fonctionnalités sur la **page de détails du film** (la page visible après avoir cliqué sur un film). C’était plus simple car il n’y avait qu’une seule note à afficher. J’ai choisi de placer les notes juste en dessous des notes officielles Canal+ dans le DOM HTML.

![Page de détails des notes d’un film](ratings_details.png)
_Notes sur la page de détails_

Une fois cela fonctionnel, j’ai étendu la fonctionnalité aux **vignettes des films** sur la page principale et dans différents carrousels. C’était plus complexe, car les films sont chargés dynamiquement. Pour gérer cela, j’ai utilisé un **MutationObserver** pour détecter les nouveaux éléments ajoutés au DOM. Les notes sont affichées dans le **coin inférieur droit de chaque vignette**, dans un encadré légèrement transparent. Cela permet de visualiser rapidement et de comparer les notes de tous les films directement depuis la page principale. Les notes des vignettes peuvent également être désactivées via les options de l’extension pour les utilisateurs préférant une interface plus épurée.

![Notes sur les vignettes des films](ratings_thumbnails.png)
_Notes sur les vignettes_

Avec l’affichage des notes sur les vignettes, le nombre de requêtes a considérablement augmenté, entraînant une file plus longue. Récupérer les notes de tous les films de la page peut prendre jusqu’à 30 secondes, voire plusieurs minutes si l’utilisateur navigue rapidement et ajoute de nouveaux films à la file (en supposant que le cache est vide). C’est un compromis nécessaire pour éviter de déclencher la détection anti-bot des moteurs (voir [Gestion des limites de requêtes](#gestion-des-limites-de-requêtes)).  

Pour améliorer l’expérience utilisateur, j’ai mis en place un **système de priorité** dans la file. Si l’utilisateur clique sur un film dont la note n’a pas encore été récupérée parce qu’il se trouve loin dans la file, ce film est automatiquement déplacé en tête. Cela garantit que la note du film sélectionné s’affiche immédiatement, sans attendre que toutes les autres notes soient traitées.

Autre amélioration importante : gérer les films dont la récupération de note échoue. Cela peut arriver si le titre est trop générique ou si le film est peu connu des moteurs. Dans ce cas, l’échec est enregistré pour éviter des requêtes répétitives inutiles, qui ralentiraient la file. Un nouvel essai est planifié après un intervalle configurable (1 heure par défaut, ajustable dans les [Pages d’options](#page-doptions)).

Enfin, pour informer l’utilisateur, un **spinner de chargement** est affiché à la place de la note tant qu’elle n’est pas récupérée. Cela montre visuellement que l’extension travaille et que la note apparaîtra bientôt.

![Chargement des notes sur les vignettes](spinner.png)
_Chargement des notes sur une vignette_

## Publier une extension Firefox

J’ai choisi de publier l’extension sur le **Firefox Add-ons Store officiel** pour me familiariser avec le processus. Grâce à la documentation détaillée de [Mozilla](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Your_first_WebExtension), la procédure a été simple.  

L’effort principal a été de créer le fichier `manifest.json`, incluant la description de l’extension, la liste des fichiers sources, les permissions requises et l’ID de l’extension. Une fois enregistrée, l’extension est alors rapidement disponible sur le Firefox Store.

> L'extension est disponible ici pour Firefox : [Movie Ratings for Canal+](https://addons.mozilla.org/fr/firefox/addon/movie-ratings-for-canal/)
{: .prompt-tip }

### Page d’options

Depuis la **page de gestion des extensions** de Firefox, les utilisateurs peuvent également ajuster plusieurs paramètres. Par exemple, il est possible de désactiver certaines sources de notes ou d’ajuster la durée du cache selon les préférences personnelles.  

![Options de l’extension](options.png)

## Améliorations futures  

Bien que de nombreuses améliorations soient possibles, il s’agissait d’un petit projet, j’ai donc décidé de m’arrêter ici pour me concentrer sur de nouveaux projets.  

Améliorations potentielles :  
- **Support d’autres navigateurs** : Migrer l’extension vers **Manifest V3** la rendrait compatible avec Chrome et autres navigateurs basés sur Chromium, avec quelques ajustements mineurs.  
- **Amélioration des recherches de notes** : Intégrer des informations supplémentaires, comme l’année de sortie, pourrait réduire les confusions entre films portant le même titre et améliorer l’exactitude des notes.  
- **Ajout de nouvelles sources de notes** : Aller au-delà d’IMDb, Rotten Tomatoes et Allociné offrirait une perspective encore plus large.  
- **Support d’autres plateformes de streaming** : Le mécanisme backend existant pourrait être adapté pour fonctionner avec Netflix, Disney+ et autres, permettant d’utiliser la même extension pour plus de contenus.
