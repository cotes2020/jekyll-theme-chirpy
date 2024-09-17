---
title: Final Fantasy Like
description: >-
  Un Final Fantasy au tour par tour
author: archalos
date: 2024-09-02 16:00:00 +0100
categories: [Project, Personnal]
tags: [game-dev, draft, in_progress]
lang_tags: [Godot, GDScript]
pin: false
github_repo: https://github.com/Archalos0/ff-fangame
image:
  path: /assets/img/FF-fond.avif

mermaid: true
---

Un projet personnel qui me tient à cœur car je suis fan des jeux tour par tour et surtout de la licence Final Fantasy.

J'ai décidé d'utiliser Godot car c'est à la "mode" en ce moment et j'étais curieux de voir comment cela fonctionnait.


## Architecture


### Personnage
***

```mermaid
classDiagram
    Battler *-- "1" Character
    Character <|-- PlayerCharacter
    Character <|-- Enemy
    
    class Battler{
        bool is_player
        Sprite2D sprite
        Sprite2D arrow_playing
        Sprite2D arrow_selection
        ProgressBar health_bar
        bool is_selectable

        execute_action()
    }

    class Character{
      <<Abstract>>
      string name
      int level
      Sprite sprite  
      Stats stats
      Action[] actions

      load_actions()*
      load_stats()*
    }
    class Enemy{
      int gils
      int experience

      load_actions()
      load_stats()
    }
    class PlayerCharacter{  
      Job job
      equipments equipments

      load_actions()
      load_stats()
    }
```

Ce qui est compliqué au niveau de la gestion des personnages, c'est le fait que nos personnages ont un "job", autrement dit, une classe, qui détermine les statistiques et les actions. 


Cependant, les ennemis ne sont pas soumis à cette règle. Ils ont leur niveau, leur statistique et leurs actions.

J'ai dû faire un choix sur comment gérer cette différence entre les deux types de personnages et j'ai opté pour une abstraction des actions et des statistiques au niveau de la classe abstraite "Character". Cette technique me permet d'accéder  facilement aux données, même si cela me complique légèrement la tâche au niveau du chargement. 

![](/assets/img/character-system-option-non-gardée.svg){: width="250" .left}

<br>
<br>

La deuxième solution (schéma ci-contre) était d'avoir une séparation des actions et statistiques selon le type de personnage. Pour cela, les actions et les statistiques seraient, pour la classe "Enemy", stockées dans cette dernière et pour les "PlayerCharacter" elles seraient stockées dans la variable job.

Cette option aurait permis de garder une séparation sur le fonctionnement des deux types de personnages dans le cas où elles seraient amenées à évoluer différemment or, pour le moment je ne pense pas.

<br>
<br>
<br>
<br>
<br>

### Equipement
***

Cette partie est en cours d'analyse, le schéma qui suit n'est pas définitif.

```mermaid
classDiagram
    Equipments *-- "5" Equipment
    Equipment <|-- Weapon
    Equipment <|-- Armor
    
    class Equipments {
        Equipment left_hand // where body_part == HAND
        Equipment right_hand // where body_part == HAND
        Armor body // where body_part == BODY
        Armor arm // where body_part == ARM
        Armor head // where body_part == HEAD
    }

    class ARMOR_TYPE {
        <<Enumeration>>
        SHIELD
        HEAD
        BODY
        ARM
    }
    class ELEMENT{
        <<Enumeration>>
        FIRE
        ICE
        LIGHTNING
        EARTH
        WIND
        HOLY
    }
    class BODY_PART{
        <<Enumeration>>
        HAND
        HEAD
        BODY
        ARM
    }
    class WEAPON_TYPE{
        <<Enumeration>>
        DAGGER
        SWORD
        DARK_BLADE
        STAVE
        ROD
        NUNCHAKU
        BOW
        ARROW
        BOOK
        KNUCKLES
        SPEAR
        HAMMER
        AXE
        THROWING_WEAPON
        BELL
        HARP
        UNARMED
    }
    class STATUS{
        <<Enumeration>>
        POISON
        PARALYSIS
        CONFUSION
        PETRIFY
        SLEEP
        TOAD
        MINI
        GRADUAL_PETRIFY
    }

    
    
    
    class Armor {
        ARMOR_TYPE armor_type
        int defense
        int magic_defense
        Stats stats_upgraded
        ELEMENT[] elemental_resistance
        ELEMENT[] elemental_weakness
        STATUS[] statuses_immunity
    }

    
    class Equipment {
        BODY_PART body_part
    }

    class Weapon{
        body_part = HAND
        WEAPON_TYPE weapon_type
        int attack
        Stats stats_upgraded
        ELEMENT element
        // add weapon used as item system
        // add additional effect on physical attack
    }
```

## Task list

### Combat 
- [ ] Navigation dans le menu du combat
  - [ ] Pouvoir annuler l'action sélectionnée (annuler l'action lors de la sélection de la cible)
  - [ ] Pouvoir annuler l'action du joueur précédent (s'il y en a un) 
  - [ ] Menu des actions dynamique selon les compétences des personnages
  - [ ] Menu des ennemis et des alliés dynamique selon leur état
    - [ ] Mort / Vivant
    - [ ] Poison, Brulure, Gel, Sommeil, Mini, Crapaud, etc

***

- [ ] Ajouter des animations pour les attaques (Attaquant et Défenseur)

***

- [ ] Ajouter des compétences
  - [ ] Magie blanche
  - [ ] Magie noire
  - [ ] Compétences

***

- [ ] Utiliser des items

### Hors combat

**Tout** 😅

