---
title: API de gestion de colocation
description: >-
  Une API qui permet de fournir des données à une application de gestion de colocation
author: archalos
date: 2024-09-30 07:00:00 +0100
categories: [Project, Personnal]
tags: [api, in_progress]
lang_tags: [NestJS, PostgreSQL]
pin: false
github_repo: https://github.com/Archalos0/coloc-server
image:
  path: /assets/img/colocation.jpg
---

## Pourquoi ce projet ?
***

L’idée de ce projet m’est venue après plusieurs années de colocation. Depuis mon bac, j’ai toujours vécu en colocation, et même si ces expériences ont toujours été positives, je me suis rendu compte qu'une meilleure organisation pourrait simplifier la gestion de la vie commune. Bien que je n'aie jamais eu de conflits majeurs, chaque colocataire ayant son propre rythme et ses propres habitudes, il peut parfois être difficile de s'organiser de manière fluide. Par exemple, la gestion des courses, des tâches ménagères ou même des documents comme les baux pourrait bénéficier d’un peu plus de structure.

C’est ainsi que j’ai eu l’idée de créer une solution qui non seulement facilite la communication et l’organisation entre colocataires, mais qui me permettrait aussi de démontrer mes compétences en développement d’applications. En plus de répondre à un besoin que j’avais identifié, ce projet représentait également une opportunité de le transformer en produit potentiel, voire de me lancer en freelance.

## Changement de situation
***

Au départ, mon idée était de développer une application à la fois web et mobile, capable d'aider les colocataires à s'organiser. L'application aurait regroupé plusieurs fonctionnalités comme :

- la répartition des courses,
- la répartition des tâches ménagères,
- des informations sur la chambre louée (prix du loyer, état des lieux, baux, etc.),
- la gestion du frigo,
- l’organisation des événements entre colocataires,
- un chat commun à la colocation.
  
Toutefois, en commençant à travailler sur le projet, je me suis rapidement rendu compte que mes compétences en design n’étaient pas à la hauteur. Les interfaces que je créais pour le site et l'application mobile étaient visuellement peu attrayantes. Plutôt que de me disperser sur un domaine où je n'étais pas à l'aise, j'ai choisi de me concentrer sur ce qui me passionne réellement : le développement back-end.

J'ai donc orienté le projet vers la création de l'API, la gestion de la base de données, et la gestion des documents physiques, qui sont pour moi les aspects les plus stimulants et les plus enrichissants.


## Technologie utilisées
***

Pour ce projet, j'ai choisi d'utiliser un ensemble de technologies robustes et adaptées à mes besoins en développement back-end et en gestion de données.

- **Back-end** : J'ai opté pour un framework Node.js pour la gestion du serveur. Après avoir hésité entre Express, que je maîtrisais déjà, et ***NestJS***, mon choix s'est finalement porté sur ce dernier. Cela s'explique par deux raisons principales : d'une part, NestJS m'offrait l'opportunité d'élargir mes compétences en explorant un nouveau framework plus structuré, et d'autre part, un ami développeur m'en avait fait de très bons retours.

- **Base de données** : Pour la gestion des données, j'ai choisi ***PostgreSQL***, une base de données relationnelle reconnue pour sa robustesse et sa capacité à gérer des requêtes complexes. PostgreSQL est non seulement fiable, mais il est également très bien supporté par l’écosystème Node.js, notamment avec l’intégration de ***Prisma***, un ORM moderne qui simplifie la gestion des bases de données. Prisma m'a permis de gérer les relations entre les différentes entités du projet (comme les utilisateurs, les tâches et les locations) de manière efficace, tout en assurant des requêtes SQL optimisées. Le système de gestion des transactions avancé de PostgreSQL s'est avéré particulièrement utile dans ce projet, où l’organisation des données (répartition des tâches, gestion des loyers, etc.) exigeait une structure stricte et sécurisée.

- **Hébergement et infrastructure** : J'ai décidé d'héberger mon API ainsi que la base de données sur un serveur ***OVH***. Le serveur me permet également de gérer un serveur ***SFTP*** pour stocker les fichiers partagés et documents importants (contrats de bail, états des lieux, etc.). OVH offre des solutions d’hébergement performantes et une grande flexibilité, ce qui correspondait bien à mes besoins pour ce projet.

## Architecture du système
***

### Base de données
***

Voici l'architecture de la base de données à laquelle j'ai imaginée il y a environ 7 mois : [Architecture BDD](https://miro.com/app/board/uXjVN8nFNlg=/?share_link_id=592237369844){:target="_blank"}

J'ai réalisé ce schéma assez rapidement, en me concentrant principalement sur les interactions entre les différentes tables. C’est pourquoi vous ne trouverez pas les typages exacts des données dans ce diagramme. Mon objectif initial était d'organiser les relations entre les entités clés (utilisateurs, tâches, loyers, etc.) afin d’assurer une cohérence dans la gestion des informations.

### API
***

Ce que j'apprécie particulièrement dans NestJS, c’est l’organisation des fichiers par **module**. Chaque fonctionnalité de l'application est structurée en un module distinct, représenté par un sous-dossier dans le répertoire `src`. À l'intérieur de chaque dossier de module, nous retrouvons les éléments essentiels comme le **contrôleur**, les **services**, et éventuellement les **DTO** (Data Transfer Objects).

Les **DTO** jouent un rôle important dans NestJS. Ils permettent de valider et d’assurer que toutes les données envoyées pour une opération, qu’il s’agisse d’une modification ou d’un ajout dans la base de données, respectent la **cohérence** et les règles définies (une approche similaire au principe **ACID** des transactions en base de données).

Voici un exemple de la hiérarchie de mon API avec le module `users` :

```
src/
├── app.module.ts
├── main.ts
├── users/
│   ├── users.controller.ts
│   ├── users.module.ts
│   ├── users.service.ts
│   └── dto/
│       └── create-user.dto.ts
│       └── update_user.dto.ts
```

- `users.module.ts`: le module regroupe le controller, le service et d'autres composant nécessaire au fonctionnement du module,
- `users.controller.ts`: le controller gère les requêtes HTTPS du client (Get, Update, Delete, etc),
- `users.service.ts`: le service lui gère la logique métier et les interactions avec la base de données.
- Le dossier dto contient les différents format pour valider les requêtes. `create-user-dto.ts` valide les données lors de la création d'un utilisateur et `update-user-dto.ts` lors de la modification.


Après avoir mis ce projet de côté pour me consacrer à d'autres sujets, comme la création de ce site, l'apprentissage de Rust, et mes premières expériences en développement de jeux vidéo avec Godot, il reste encore plusieurs points à retravailler et à compléter :

- **Infrastructure** : J'ai récemment perdu mon VPS chez OVH, ce qui signifie que je dois configurer un nouveau serveur. Cela inclut la sécurisation du serveur, la configuration du SFTP, ainsi que la remise en place de la base de données.

- **Ajout des fonctionnalités manquantes** : Pour l’instant, seule la base du projet est en place. Il reste encore beaucoup de travail, notamment pour développer l’API et intégrer toutes les fonctionnalités prévues initialement (gestion des tâches, organisation des documents, etc.).

- **Partie design** : Bien que l’aspect visuel ne soit pas ma priorité pour le moment, je prévois de l'améliorer à terme. Pour cela, je ferai probablement appel à un designer, ou je demanderai à un ami spécialisé dans le front-end de m'aider, si cela l'intéresse et qu'il a le temps.
