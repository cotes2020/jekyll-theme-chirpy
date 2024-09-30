---
title: Calculatrice dans une cmd
description: >-
  Création d'une calculatrice en Rust
author: archalos
date: 2024-09-02 16:00:00 +0100
categories: [Project, Personnal]
tags: [dev, in_progress]
lang_tags: [Rust]
pin: false
github_repo: https://github.com/Archalos0/rust-calculatrice
image:
  path: /assets/img/rust-lang-light.svg


mermaid: true
---

Ce projet est un projet d'entraînement au langage Rust. Je voulais apprendre Rust afin de me familiariser avec les langages bas-niveaux.

Pourquoi Rust précisément ? Tout simplement car c'est un langage prometteur dans lequel Google a investi 1 million de dollars et que Linus Torvalds, créateur du noyau Linux, a affirmé que Rust va être de plus en plus utilisé dans le noyau Linux.

## Fonctionnement
***

### Fonctionnement très peu détaillé du programme
***

1. Recherche de groupements entre parenthèses
2. Si on en trouve, on revient au début pour calculer le groupement
3. Si non, on recherche l'opération prioritaire : la multiplication ou la division la plus à gauche
4. Si on trouve, on effectue le calcul avec le membre de droite et de gauche
5. Si non, on effectue l'opération la plus à gauche

### Fonctionnement présenté sous forme de logigramme
***

*Ce diagramme sera prochainement amélioré*
``` mermaid
flowchart TD
  A([Start]) --> B[/Saisie de l'utilisateur/]
  B --> C[Nettoyage de la saisie]
  C --> D['s' = saisie nettoyé]
  D --> E{Si 's' contient un groupement entre parenthèses}
  E -- oui --> F['s' = groupement entre parenthèses]
  F --> E
  E -- non --> G{Si 's' ne contient q'un seul membre} 
  G -- oui --> H([Renvoie le chiffre])
  G -- non --> I[Cherche index operation prioritaire]
  I --> J{Si operation prioritaire trouvée}
  J -- oui --> K[calcul operation prioritaire]
  J -- non --> L[calcul premier opérateur]
  K --> M['s' = 's' où l'opération effectuée est remplacé par le résultat]
  L --> M
  M --> G
```



## Task List

- [x] Gérer les parenthèses consécutives : (\<calcul\>)(\<calcul\>)
- [ ] Gérer les formes de calcul suivantes : (8+2)2
