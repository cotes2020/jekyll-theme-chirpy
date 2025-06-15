---
title : 'Home Server : Services auto-hébergés'
description : Découvrez comment j'héberge plusieurs services sur mon propre serveur
date : 2022-02-03
categories : []
media_subpath : /assets/img/posts/homeserver
tags : [network]
lang : fr-FR
image:
  path: homepage.png
---

Dans un monde où nous sommes de plus en plus dépendants du cloud et des grandes entreprises pour stocker nos données, je me suis rapidement intéressé à l'auto-hébergement. Depuis 3 ans, j'héberge plusieurs services sur mon serveur pour mes proches et moi.

## Le début de mon voyage : Raspberry PI et PI-Hole 

Mon addiction à l'auto-hébergement a commencé avec un Raspberry PI et l'application **PI-Hole** pour filtrer toutes les publicités et le tracking dans mon réseau au niveau DNS. Je l'ai utilisé comme ça pendant presque un an, mais ensuite j'ai voulu en avoir plus. 

![PI-Hole](pihole.png){: w="150" h="150"}
_PI Hole_

Le premier service que je voulais ajouter était un serveur média. Le plus populaire à l'époque était Plex. 
J'ai donc connecté un disque dur de 1 TO pour stocker les médias et j'ai essayé d'installer et de configurer Plex Server. Il était relativement facile à installer, mais après l'avoir utilisé pendant un certain temps, j'ai rencontré certaines limitations. 

Tout d'abord, des instabilités se produisaient régulièrement lorsque Plex ou d'autres logiciels étaient mis à jour. 
Deuxièmement, le Raspberry PI modèle 3B+ avec 1 Go de RAM et un processeur quadricœur cadencé à 1,4 GHz, avait du mal à décoder les médias haute résolution. 

Pour résoudre ce problème, j'ai pris deux initiatives : utiliser Docker et acheter du nouveau matériel.

## Installation facile avec Docker

**Docker** est une plateforme conçue pour exécuter des logiciels à l'intérieur d'un conteneur. Les conteneurs sont isolés les uns des autres et regroupent leurs propres logiciels, bibliothèques et fichiers de configuration. Et comme tous les conteneurs partagent un même noyau de système d'exploitation, ils utilisent moins de ressources que les machines virtuelles. 

![Docker](docker.png){: w="300" h="150"}
_Docker_

Pour ces raisons, l'utilisation de Docker est un choix idéal, et il devient trivial de gérer plusieurs services tournant sur le même serveur sans se soucier des dépendances, des incompatibilités, etc...

De plus, je définis toutes mes configurations de conteneurs dans des fichiers YAML en utilisant Docker Compose. De cette façon, toutes les configurations du serveur peuvent être facilement sauvegardées et mises à jour.

## Un matériel plus puissant

Avec de plus en plus de services et l'incapacité de mon RPI à décoder certains médias volumineux, j'ai choisi d'acheter du matériel plus puissant. J'ai trouvé une **Lenovo Thinkstation** sur le marché de l'occasion (souvent inondé de vieux matériel d'entreprise). C'est beaucoup moins cher qu'un vrai serveur et c'est largement suffisant pour mon utilisation.

![Lenovo Thinkstation](lenovo.png){: w="200" h="150"}
_Thinkstation Lenovo_

Pour le système d'exploitation, j'ai installé la dernière version stable de Debian (headless) pour avoir une base solide et rester compatible avec la plupart des logiciels.

Il a été installé sur un nouveau SSD de 512Go pour améliorer les performances. Le disque dur 1 TO ne stocke que les données des services (média, stockage statique, ...).

## Les services de mon serveur

Maintenant que j'ai un serveur fonctionnel, j'ai commencé à ajouter de plus en plus de services en fonction de mes besoins. J'ai réalisé l'infographie suivante avec les services que j'utilise actuellement :

![Homeserver Architecture](beniserv.png){: w="600" h="650"}
_Mes services_

### Accès et sécurité

Le service le plus important est probablement **Caddy**. Il s'agit d'un serveur web avec des fonctions HTTPS et reverse proxy automatiques. 

En d'autres termes, grâce à Caddy, je peux accéder directement à mes services en utilisant un sous-domaine et tout le trafic sera crypté et transféré vers le port 443. Par exemple, je peux accéder à un de mes services directement sur « myservice.beniserv.fr » et un autre sur « myotherservice.beniserv.fr », etc...

Avec cette configuration, je n'ai à exposer qu'un seul port (443) à l'internet, et tout est toujours crypté.

J'ai également ajouté un plugin DNS dynamique à Caddy, afin de synchroniser l'adresse de mon serveur IP avec celle de mon fournisseur de noms de domaine. Ainsi, même si mon fournisseur d'accès à internet m'attribue une IP dynamique, je peux toujours avoir accès à mon serveur domestique via mon nom de domaine.

Pour améliorer la sécurité, j'utilise également un service appelé **Crowdse**c pour détecter les pairs ayant des comportements malveillants et les empêcher d'accéder à mon serveur. Crowdsec analyse les journaux Caddy et si son moteur de comportement détecte l'un des scénarios configurés, il bloque l'IP au niveau du pare-feu. L'avantage de Crowdsec par rapport aux autres services similaires disponibles (Fail2ban par exemple), est qu'il offre une solution collaborative. En effet, lorsqu'un utilisateur de Crowdsec bloque une IP agressive, cette information est également partagée par tous les utilisateurs afin d'améliorer encore la sécurité de chacun.

### Mise à jour et sauvegarde

J'ai aussi **Watchtower** pour garder tous mes services à jour et bénéficier des derniers correctifs de sécurité et des dernières fonctionnalités. Chaque nuit, il télécharge les dernières versions et les remplace si une mise à jour est disponible.

Et comme il arrive qu'une mise à jour ou un disque de stockage tombe en panne, j'utilise un service pour effectuer une sauvegarde périodique. Chaque nuit, il enregistre toutes les configurations du serveur et certaines données dans un fichier d'archive crypté. Cette archive est ensuite sauvegardée sur mon disque dur de 1 TO et sur un serveur de stockage AWS. Une politique de rétention de 7 jours permet de ne conserver seulement les sauvegardes les plus récentes.

> Find the docker-volume-backup service here : [https://github.com/offen/docker-volume-backup](https://github.com/offen/docker-volume-backup).
{: .prompt-tip }

### Supervision

Avec plusieurs services en cours d'exécution, j'ai besoin d'un moyen de les surveiller et de recevoir des alertes lorsqu'un problème survient. C'est pourquoi j'utilise **Portainer** et **Homepage**.

**Portainer** fournit une interface graphique pour gérer les conteneurs Docker. Même si je préfère utiliser Docker Compose pour déployer de nouveaux services, j'utilise toujours Portainer pour surveiller la santé de mes services existants. Si nécessaire, je peux facilement les redémarrer à partir de l'interface utilisateur de Portainer.

**Homepage** est un tableau de bord hautement personnalisable. Il fournit des widgets pour un large éventail de services et peut afficher diverses mesures du système (temps de fonctionnement, espace disque, température du serveur, etc...). Je l'utilise pour accéder rapidement à mes différents services.

![Exemple de page d'accueil](homepage.png){: w="600" h="350"}
_Exemple de page d'accueil_

> Trouvez le service de page d'accueil ici : [https://github.com/gethomepage/homepage](https://github.com/gethomepage/homepage).
{: .prompt-tip }

Pour être informés des erreurs, mes services envoient également des messages à un serveur Discord privé.

### Autres services

Comme vous pouvez le voir dans l'image précédente, les services suivants sont également en cours d'exécution :
- **Jellyfin** : Serveur multimédia (remplace Plex)
- **Paperless-ngx** : Système de gestion et d'archivage de documents
- **Nextcloud** : Service de stockage cloud
- **QBitorrent** : Client torrent
- Et bien d'autres choses encore...