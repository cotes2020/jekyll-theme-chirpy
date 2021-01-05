---
layout: post
title:  "(Faux-)problèmes du javascript"
---

Bonjour à tous !  

Dans ce billet, nous allons parler d'un sujet très sensible dans les communautés de développeurs, "Pourquoi faire ou ne pas faire du JavaScript ?".   

# Environnements et versions concernées

Dans cet article, nous allons partir du principe que le développeur utilise mode strict, les dernières versions de JavaScript - ES8/9/10/11/next selon le support - quelques détails sont disponibles sur [Wikipedia](https://fr.wikipedia.org/wiki/ECMAScript) ou bien sûr la [spécification officielle](https://www.ecma-international.org/ecma-262/).  

On partira aussi du principe que le développeur suivra de bonnes pratiques de développement telles qu'un Style Guide - comme celui de [airbnb](https://github.com/airbnb/javascript), de [google](https://google.github.io/styleguide/jsguide.html), etc. - ou tout du moins de bonnes pratiques généralistes.  

Maintenant que le contexte est défini, entrons dans le vif du sujet.  

# Le système de type

## Les comparaisons

On peut trouver deux sortes de problèmes liés au type, le premier est le système de comparaison. En JavaScript, on peut écrire:  

```js
[] == 0; // true
"0" == 0; // true
[] == "0"; // false
```

Il y a de nombreux cas de comparaisons où un à un les cas sont plutôt "logiques" mais la transitivité n'est pas satisfaite. Mais heureusement, il est facile de contourner ce problème.

```js
[] === 0; // false
"0" === 0; // false
[] === "0"; // false
```

La différence entre == et === est que le premier vérifie une égalité entre les valeurs (quitte à les changer de type pour les comparer) alors que le second vérifie aussi le type de variable qui est comparée.  

## typeof et instanceof

Ces deux instructions ont des comportements assez inattendus, en effet, on peut voir sur le type **string** que:  
```js
typeof "chaine de caractere"; // "string"
"chaine de caractere" instanceof String; // false
```

Ce n'est pas réellement un bug, mais plutôt un comportement mal défini, car on a aussi:  
```js
(new String("chaine de caractere")) instanceof String; // true
```

On peut donc remarquer que la chaîne de caractère déclarée en tant que "variable primitive" n'est pas considérée comme équivalent à son type objet.  

Un autre exemple de problème avec typeof, souvent utilisé pour tourner en ridicule le langage est:
```js
typeof NaN; // "number"
```

Vous avez bien lu, NaN - Not a Number - est considéré comme un nombre via l'opérateur typeof.  

Il faut aussi noter que null est considéré comme de type object - comme NaN est considéré comme de type number - mais que ceux-ci ne sont pas considérés comme instance de leur type respectif via l'opérateur instanceof.  

# Les faux problèmes

Durant mes différentes recherches pour écrire cette article, j'ai remarqué qu'il y'avait des problèmes répertoriés qui n'en sont pas - ou bien par erreur, ou bien par mauvaise foi - nous allons donc les passer en revue.  

## Problème d'ordre des nombres

```js
Math.min() < Math.max() // false
```

Quand j'ai vu cet exemple pour la première fois, j'ai commencé à me poser desacrées questions sur le langage, puis j'ai fait quelques recherches, ces deux fonctions min() et max(), ne donnent pas les bornes du type Number comme on pourrait le croire mais le minimum ou le maximum entre plusieurs nombres. Et les valeurs par défaut expliquent le résultat précédent:

```js
Math.min() // Infinity
Math.max() // -Infinity
```

Le vrai problème de ces fonctions est qu'il devrait y avoir une erreur de levée lorsque aucun argument n'est fourni.

## L'immutabilité et la portée des variables

Un problème reproché pendant longtemps était que le hoisting (la portée d’existence d'une variable) et l'immutabilité étaient mal conçus en JavaScript. C'est dans le but de résoudre ce soucis que ES6 a ajouté les mots-clés **const** et **let**. On peut aussi de **Object.freeze()** qui va pousser encore plus loin le comportement des constantes que **const**.

## Problèmes "Hors jeu"

Il y a quelques problèmes, notamment la représentation des nombres flottants,qui sont inhérents à beaucoup de langages différents dont nous ne parleront pas dans cet article. Exemple:  
```js
0.1+0.2==0.3; // false, car egal à 0.30000000000000004 par la representation des nombres flottants
```

# Solution générale aux problèmes

Après avoir vu cette liste de points positifs et négatif, on peut se rendre compte qu'il y a pas mal de problèmes faciles a faire (et a resoudre), heureusement, il existe comme pour beaucoup de langage des "vérificateurs de code" qu'on appelle Linter, tels que eslint, jshint ou jslint que vous pouvez intégrer (ou qui le sont déjà) à vos éditeurs et IDE afin de prévenir la plupart de ces problèmes et bien d'autres encore.  

Je pourrais aussi parler (et je le ferais probablement dans un article futur), d'aller plus loin dans la qualité de code - peu importe le langage - en utilisant un logiciel tel que Sonarqube ou Codacy.

# JSFuck

Petit aparté final sur ce billet pour parler de JSFuck ([http://www.jsfuck.com](http://www.jsfuck.com)). C'est un style de programmation en Javascript qui n'utilise que 6 caractères différents afin d'encoder n'importe quel script javascript.  

Il est ici question de pousser le langage dans ses retranchement, utiliser certains de ces comportements (bon ou mauvais) afin de réussir a construire n'importe quoi avec 6 caractères.  

Si vous êtes intéressés, vous pouvez allez faire un tour sur le site officiel: [http://www.jsfuck.com](http://www.jsfuck.com)

# Conclusion

N’hésitez pas à partager lien de cette page lors de vos débats afin de pouvoir bien peser le pour et le contre du JavaScript. J'espère que vous avez apprecié ce petit billet, en attendant, je vous dit a la prochaine dans un prochain article !  

---

**Sources utilisées dans cet article:**  
- [https://wiki.theory.org/index.php/YourLanguageSucks#JavaScript_sucks_because](https://wiki.theory.org/index.php/YourLanguageSucks#JavaScript_sucks_because)
- [https://charlieharvey.org.uk/page/javascript_the_weird_parts](https://charlieharvey.org.uk/page/javascript_the_weird_parts)
- [https://developer.mozilla.org/fr/docs/Web/JavaScript/Reference/Opérateurs/instanceof](https://developer.mozilla.org/fr/docs/Web/JavaScript/Reference/Opérateurs/instanceof)

**Ainsi que tout les liens référencés dans l'article:**  
- [https://fr.wikipedia.org/wiki/ECMAScript](https://fr.wikipedia.org/wiki/ECMAScript)
- [https://www.ecma-international.org/ecma-262/](https://www.ecma-international.org/ecma-262/)
- [https://github.com/airbnb/javascripthttps://google.github.io/styleguide/jsguide.html](https://github.com/airbnb/javascripthttps://google.github.io/styleguide/jsguide.html)