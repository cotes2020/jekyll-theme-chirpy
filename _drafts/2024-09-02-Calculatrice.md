---
title: Une calculatrice dans une console 
description: >-
  Création d'une calculatrice en Rust
author: archalos
date: 2024-09-02 16:00:00 +0100
categories: [Project, Personnal]
tags: [dev, draft, in_progress]
lang_tags: [Rust]
pin: false
github_repo: https://github.com/Archalos0/rust-calculatrice
image:
  path: /assets/img/rust-lang-light.svg


mermaid: true
---

Ce projet est un projet d'entrainement au langage Rust. Je voulais apprendre Rust afin de me familiariser avec les langages bas-niveaux.

Pourquoi Rust précisément ? Tout simplement car c'est un langage prometteur dans lequel Google à investi 1 million de dollars et que Linus Torvalds, créateur du noyeau Linux, à affirmé que Rust va être de plus en plus utilisé dans le noyau Linux.

## Fonctionnement
``` mermaid
flowchart
  A[Start] --> B[calcul groupement]
  B --> C{Si contient un groupement entre parenthèse}
  C -- oui --> B
  C -- non --> D{Si un seul membre} 
  D -- oui --> E{Renvoie le chiffre}
  D -- non --> F[Cherche index operation prioritaire]
  F --> G{Si operation prioritaire trouvé}
  G -- oui --> H[calcul operation prioritaire]
  G -- non --> I[calcul premier operateur]
```


### Task List

- [ ] Gérer l'association groupements (\<calcul\>)(\<calcul\>)
