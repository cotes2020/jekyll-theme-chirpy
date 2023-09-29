---
layout: post
title: La sécurité des protocoles d’authentification NTLM et Kerberos en environnement
  Active Directory
image:
  path: "/assets/img/Article%20Kerberos%20NTLM/article.png"
categories:
- Microsoft
- Active Directory
tags:
- ad
- protocol
- kerberos
- ntlm
description: Cet article explique le fonctionnement de NTLM dans ces différentes versions
  et de Kerberos v5 au sein de l’environnement Active Directory. Les différences entre
  ces protocoles seront étudiées, ainsi que les attaques liées à ces protocoles.
date: 2023-09-27 19:00 +0100
---
<style>
.tab {
    width: 100%; 
    border-collapse: collapse; 
    text-align: center;
}

.tab tbody, 
.tab td, 
.tab tfoot, 
.tab th, 
.tab thead, 
.tab tr {
    border: 1px solid var(--tb-border-color);
}

.star {
    color: var(--highlighter-rouge-color);
}
</style>
## Introduction
**NTLM** et **Kerberos** sont des protocoles d'authentification utilisés au sein de l’environnement Microsoft **Active Directory**. Kerberos a été développé par le MIT, tandis que NTLM a été développé par Microsoft pour ses systèmes d'exploitation Windows.

Depuis leurs créations en 1993, les protocoles Kerberos v5 et NTLM gèrent les besoins d’authentification au cœur du SI des entreprises disposant d'un environnement Active Directory. Bien que différents, ces deux protocoles disposent chacun de faiblesses exploitées par les attaquants et les chercheurs en cybersécurité.

Cet article explique le fonctionnement de NTLM dans ses différentes versions et de Kerberos v5 au sein de l’environnement Active Directory. Les différences entre ces protocoles seront étudiées, ainsi que les attaques liées à ces protocoles.

## NTLM
Le protocole d'authentification NTLM ou "New Technology Lan Manager" a été créé en 1993 pour remplacer le protocole Lan Manager devenu trop vulnérable.
NTLM est utilisé pour vérifier l'identité d'un utilisateur ou d'une machine sur un réseau en se basant sur un système de challenge-response.

### Fonctionnement de NTLM
Au sein de l’Active Directory les utilisateurs et services utilisant NTLM fonctionne sur le schéma suivant[^1] :
![NTLM](/assets/img/Article%20Kerberos%20NTLM/ntlm.png){: .shadow }

1. Un utilisateur s’authentifie avec ses identifiants sur sa machine (nom de domaine, d'utilisateur et mot de passe…).
2. La machine client envoie un message **“Negociate”** contenant, entre autres, le nom du domaine, les fonctions supportées par le client et le nom de la machine. [^2]
3. Le serveur cible génère alors un message **“Challenge”** composé notamment du “défi”, qui est un nombre aléatoire et l'envoie à l'ordinateur client. [^3]
4. La machine client répond par un message **“Authenticate”** et envoie, entre autres, la "réponse", qui est le challenge chiffré par le hash NT<span class="star">*</span> du mot de passe du client. [^4]
5. Le serveur envoie au Contrôleur de Domaine (DC)<span class="star">*</span> le nom d'utilisateur, le défi et la réponse via un message **“NETLOGON_NETWORK_INFO”**. [^5]
6. Le DC récupère le hash du mot de passe de l'utilisateur à partir de sa base de données puis chiffre le “défi” et le compare avec “la réponse”. S’ils sont identiques, l’authentification est approuvée et l’utilisateur peut accéder à la ressource. Le DC envoie alors un message **“NETLOGON_VALIDATION_SAM_INFO4”** au serveur. [^6]

### Différences entre Net-NTLMv1 et Net-NTLMv2
Net-NTLMv1 et Net-NTLMv2 sont les deux versions du protocole NTLM. Net-NTLMv2 a été lancé en 1998 avec la sortie de Windows NT 4.0 SP4, cette deuxième version de NTLM a été créée dans le but de remplacer Net-NTLMv1 afin d'améliorer la sécurité cryptographique du protocole.

La principale différence entre ces deux versions est l'algorithme utilisé pour chiffrer le challenge. Pour **Net-NTLMv1**, c’est **DES** qui est utilisé, à noter que cet algorithme n’est pas résistant aux attaques par brute force du fait notamment sa faible complexité et de la petite taille de la clé utilisé (56 bits).  
D'autre part, pour **Net-NTLMv2** c’est **HMAC-MD5** qui est utilisé. Bien que ce dernier soit plus résistant contre une attaque par brute force que DES, dû spécifiquement au fait que la clé est plus grande (128 bits), cela reste tout à fait possible. 

**Net-NTLMv1**
```plaintext
C = 8-byte server challenge, random
K1 | K2 | K3 = NTLM-Hash | 5-bytes-0
response = DES(K1,C) | DES(K2,C) | DES(K3,C)
```
**Net-NTLMv2**
```plaintext
SC = 8-byte server challenge, random
CC = 8-byte client challenge, random
CC* = (X, time, CC2, domain name)
v2-Hash = HMAC-MD5(NT-Hash, user name, domain name)
LMv2 = HMAC-MD5(v2-Hash, SC, CC)
NTv2 = HMAC-MD5(v2-Hash, SC, CC*)
response = LMv2 | CC | NTv2 | CC*
```

Ensuite, une deuxième différence est que **Net-NTLMv2** peut implémenter un mécanisme d’intégrité qui se traduit par un champ **MIC** (Message Integrity Code). 

Ce champ peut être présent dans le message “Authenticate” et est calculé avec la fonction HMAC-MD5 en prenant en entrée la clé de session (si présente) et les trois messages d’authentification (NEGOCIATE, CHALLENGE et AUTHENTICATE). Il permet de garantir qu’un attaquant n’a pas pu modifier le contenu des messages d’authentification.

Enfin, une troisième différence est que pour Net-NTLMv2, un **timestamp** est ajouté au message “Authenticate” pour empêcher les attaques par rejeu.

# Attaques sur le protocole NTLM

### Relai NTLM
L’attaque par relai NTLM est une attaque de type **Man In The Middle** (MitM). Afin de réaliser cette attaque, l’attaquant se positionne entre l’utilisateur et le serveur cible et relaie les requêtes au serveur. De cette façon, un attaquant peut se faire passer pour un utilisateur authentifié et accéder aux ressources du serveur.

Avant d’envoyer la première requête “Negotiate”, la victime va chercher l’adresse IP du serveur. Lors de cette étape, l’attaquant va, par exemple, abuser des protocoles **LLMNR/NBT-NS** ou pratiquer du **DNS Spoofing** afin de se faire passer pour le serveur cible aux yeux de la victime.

Une fois que la machine de la victime traite la machine de l’attaquant en tant que serveur cible, elle va effectuer les trois étapes de l'authentification NTLM auprès de l'attaquant.

De son côté, l'attaquant va simplement transmettre les requêtes du client au serveur et vice versa. À la fin de l'authentification, l'attaquant aura donc les droits de l'utilisateur victime sur les ressources du service.
![Man In The Middle](/assets/img/Article%20Kerberos%20NTLM/MitM.png){: .shadow }

On peut identifier divers outils permettant de réaliser cette attaque comme `Responder` pour la partie MitM et `ntlmrelayx` pour le relai.

Cependant, il existe différents mécanismes de protection contre ces attaques, mais la plupart d'entre eux ne sont pas activés par défaut : 

- **MIC** : mécanisme d’intégrité permettant d’empêcher la modification des messages NTLM
- **Signing** (SMB, LDAP, ...) : mécanisme garantissant l’identité de l'émetteur des requêtes après la phase d’authentification
- **Channel binding** : mécanisme interdisant la modification du protocole demandé initialement lors l’authentification

### Pass-The-Hash
Puisque le protocole NTLM utilise le hash NT du mot de passe de l’utilisateur afin de répondre au challenge du serveur, il n'est donc pas nécessaire de connaître le mot de passe de l'utilisateur, mais **seulement son hash NT** pour s'authentifier. 

C'est pourquoi la plupart du temps, les attaquants se contentent de récupérer les hash des mots de passe, car ils permettent également de s'authentifier.

![meme pth](/assets/img/Article%20Kerberos%20NTLM/meme-pth.jpg){: .shadow }

Cette “attaque” par pass-the-hash est faisable puisque lors de la conception du protocole NTLM, Microsoft a estimé que l'utilisation du hash NT au lieu du mot de passe en clair était une mesure de sécurité suffisante. Néanmoins, la conception même du processus d'authentification engendre une vulnérabilité qui permet l'exploitation de cette technique.

---------
## Kerberos
Kerberos est un protocole d’authentification réseau permettant à des utilisateurs et services d’accéder à des ressources.  Il a été créé au début des années 1980 au sein du projet Athena du MIT. 

Le but initial de ce protocole était de fournir une authentification sécurisée pour les services informatiques du MIT. Kerberos a été conçu pour remplacer les méthodes d'authentification moins sécurisées telles que l'authentification en clair ou le protocole NTLM.  La version 5 du protocole a elle été publiée en 1993 et est régulièrement mise à jour.

Kerberos repose sur un système de tickets et de clés secrètes permettant de gérer l'authentification, sans transmettre les informations d'authentification des utilisateurs/services en clair sur le réseau.

Kerberos est activé par défaut dans les réseaux internes des entreprises utilisant Active Directory.

### Fonctionnement de Kerberos
Lorsqu'un utilisateur souhaite s'authentifier auprès d'un service qui utilise Kerberos, ce dernier renseigne ses identifiants (en fonction du service demandé) lançant ainsi la phase d’authentification. Le reste du processus d'authentification est transparent pour l'utilisateur.

L’authentification Kerberos s’effectue en **six étapes**. Certaines de ces étapes vont produire des **tickets** qui pourront être réutilisés afin d’effectuer des demandes d’accès à un service ou d’accéder à un service. L’utilisation des tickets permet d’assurer la fonctionnalité d’authentification unique, ou **Single Sign On** en anglais (SSO).

Dans le but d'expliquer son fonctionnement de manière concrète, prenons l'exemple suivant : l'utilisateur **"jdoe"** du domaine **"asgard.com"** souhaite accéder au service **HTTP** de la machine **SERV01**, qui se trouve dans le même domaine.

Voici un schéma résumant l’authentification de l’utilisateur : 
![Legende](/assets/img/Article%20Kerberos%20NTLM/Kerberos-Legende.svg){: .shadow }
![Kerberos](/assets/img/Article%20Kerberos%20NTLM/Kerberos-Page-1.svg){: .shadow }

#### Les tickets
Le protocole d’authentification Kerberos implémente deux types de tickets distincts :  les Tickets Granting Tickets <span class="star">*</span>(TGT) et les Service Tickets <span class="star">*</span>(ST).

Les TGT permettent d'obtenir des ST, tandis que les ST permettent de demander l'accès à un service.

D’un point de vue technique, ces tickets sont structurés de la même façon [^7] : 
- **tkt-vno** : numéro de la version de Kerberos utilisée pour le ticket
- **realm** : royaume Kerberos où le ticket a été généré
- **sname** : SPN<span class="star">*</span> du service (nom du service et nom du serveur), pour un TGT ce sera toujours le service krbtgt<span class="star">*</span> et le nom de domaine
- **enc-part** : la partie chiffrée du ticket (chiffrée avec le hash du compte krbtgt pour les TGT et chiffrée avec le hash du compte de service pour le ST)

Dans la partie chiffrée “enc-part”, on retrouve différents champs : 

- **flags** : option décrivant des comportements du ticket (forwardable, renewable, initial…)       
- **key** : soit K<sub>C-TGS</sub> pour un TGT ou K<sub>C-SERV01</sub> pour un ST
- **cname** & **crealm** : nom du client et de son royaume Kerberos<span class="star">*</span>
- **transited** : liste des royaumes qui ont été utilisés pour authentifier l’utilisateur
- **authtime, endtime, starttime** (optionnel), **renew-till** (optionnel) : dates relatives au ticket
- **caddr** (optionnel) : adresse du client
- **authorization-data** (optionnel) : données d'autorisations/restrictions supplémentaires (PAC<span class="star">*</span>)

> K<sub>C-TGS</sub>  représente la clé de session utilisée pour les communications entre le client et le TGS et K<sub>C-SERV01</sub> représente la clé de session utilisée entre le client et le serveur SERV01.
<br>
Dans cet article, le terme “hash du compte” sera utilisé pour désigner par raccourci le hash NT du mot de passe du compte.
{: .prompt-info }


#### AS_REQ
Après que l'utilisateur a entré ses identifiants dans le service de son choix, le client Kerberos de l’utilisateur va effectuer la première étape d'authentification. 

La première étape du protocole Kerberos se nomme KRB_AS_REQ. Cette étape consiste à l’envoi par le client de données d’authentification à l’AS<span class="star">*</span> afin de récupérer un TGT.

Cette étape est souvent appelée “pré-authentification”.
![KRB_AS_REQ](/assets/img/Article%20Kerberos%20NTLM/Kerberos-KRB_AS_REQ.svg){: .shadow }

La requête KRB_AS_REQ est composée de différents éléments nécessaires à la création du TGT tels que le nom du client, du service, du domaine ou encore la date d’expiration du ticket etc. [^8]

Afin que l’AS puisse authentifier le client, la requête KRB_AS_REQ dispose d’un champ contenant un timestamp chiffré par le hash du mot de passe de l’utilisateur et d’un champ “etype” contenant le nom de l'algorithme utilisé pour le chiffrement. 

Ce champ “etype” est présent systématiquement lorsqu’un élément d’une requête est chiffré. À ce jour, il existe 6 types de chiffrement pour les tickets : 

<table class="tab">
        <tr style="background-color: #f8485e;">
            <th style="color: white;">Type</th>
            <th style="color: white;">Nom du type</th>
            <th style="color: white;">Description</th>
        </tr>
        <tr style="background-color: #1b1b1e;">
            <td>0x1</td>
            <td>DES-CBC-CRC</td>
            <td style="white-space: normal;">Désactivé par défaut à partir de Windows 7 et Windows Server 2008 R2.</td>
        </tr>
        <tr style="background-color: #1b1b1e;">
            <td>0x3</td>
            <td>DES-CBC-MD5</td>
            <td style="white-space: normal;">Désactivé par défaut à partir de Windows 7 et Windows Server 2008 R2.</td>
        </tr>
        <tr style="background-color: #d9ead3;">
            <td style="color: #4a4a4a;">0x11</td>
            <td style="color: #4a4a4a;">AES128-CTS-HMAC-SHA1-96</td>
            <td style="color: #4a4a4a; white-space: normal;">Pris en charge à partir de Windows Server 2008 et Windows Vista.</td>
        </tr>
        <tr style="background-color: #d9ead3;">
            <td style="color: #4a4a4a;">0x12</td>
            <td style="color: #4a4a4a;">AES256-CTS-HMAC-SHA1-96</td>
            <td style="color: #4a4a4a; white-space: normal;">Pris en charge à partir de Windows Server 2008 et Windows Vista.</td>
        </tr>
        <tr style="background-color: #1b1b1e;">
            <td>0x17</td>
            <td>RC4-HMAC</td>
            <td style="white-space: normal;">Suite par défaut pour les systèmes d’exploitation antérieurs à Windows Server 2008 et Windows Vista.</td>
        </tr>
        <tr style="background-color: #1b1b1e;">
            <td>0x18</td>
            <td>RC4-HMAC-EXP</td>
            <td style="white-space: normal;">Suite par défaut pour les systèmes d’exploitation antérieurs à Windows Server 2008 et Windows Vista.</td>
        </tr>
</table>

À la réception de la requête, l’AS va essayer de déchiffrer le timestamp en se servant du hash de l’utilisateur présent dans sa base de données. En cas de succès, la pré-authentification est validée et le TGT est envoyé lors de la prochaine étape. Dans le cas contraire, la pré-authentification est alors refusée et le TGT n’est pas envoyé. [^9]

> Pour certains systèmes “Legacy” ou mal configurés, cette étape de pré-authentification n’est pas utilisée.
{: .prompt-info }

#### AS_REP
Une fois la pré-authentification réussie, l’AS envoie une réponse KRB_AS_REP  au client, contenant le nom de l’utilisateur (cname), le nom du royaume Kerberos du client (crealm), une partie chiffrée avec le hash de l’utilisateur contenant une clé K<sub>C-TGS</sub> (key) et le TGT chiffré avec le hash du compte krbtgt. [^10]

La clé K<sub>C-TGS</sub> (key) est également comprise dans le TGT.

![KRB_AS_REQ](/assets/img/Article%20Kerberos%20NTLM/Kerberos-KRB_AS_REP.svg){: .shadow }

Une fois la réponse reçue par le client, ce dernier est alors capable de déchiffrer à l’aide du hash de l’utilisateur la partie chiffrée et ainsi obtenir la clé K<sub>C-TGS</sub> (key). [^11]

#### TGS_REQ
Une fois en possession d’un TGT et d’une clé K<sub>C-TGS</sub>, le client va être en mesure de faire une requête afin d’obtenir un ST pour pouvoir accéder au service.

Dans cette requête KRB_TGS_REQ, le client va envoyer au TGS<span class="star">*</span> le TGT, le SPN du service demandé ainsi qu’une partie chiffrée par la clé K<sub>C-TGS</sub> contenant des éléments d’authentification (timestamp, nom de l’utilisateur, …). [^8]

![KRB_TGS_REQ](/assets/img/Article%20Kerberos%20NTLM/Kerberos-KRB_TGS_REQ.svg){: .shadow }

De cette façon, le TGS déchiffre le TGT avec le hash du compte krbtgt pour récupérer la clé K<sub>C-TGS</sub>. Avec la clé K<sub>C-TGS</sub> le TGS déchiffre les éléments d’authentification et valide ou non la requête. [^12]

#### TGS_REP
Si la requête de ST est validée, le TGS va envoyer au client une réponse contenant un ST chiffré avec le hash du compte de service et une nouvelle clé K<sub>C-SERV01</sub> chiffrée avec la première clé K<sub>C-TGS</sub> (en jaune sur le schéma). [^10]

Cette nouvelle clé K<sub>C-SERV01</sub> est également présente dans le ST.

![KRB_TGS_REP](/assets/img/Article%20Kerberos%20NTLM/Kerberos-KRB_TGS_REP.svg){: .shadow }

Lorsque le client reçoit la réponse, ce dernier récupère le ST et obtient la nouvelle clé K<sub>C-SERV01</sub> en la déchiffrant à l’aide de la première clé K<sub>C-TGS</sub>. [^13]

#### AP_REQ
Maintenant que le client a obtenu un ST, la requête de demande d’accès au service KRB_AP_REQ va pouvoir être initiée.

Cette requête contient le ST et une partie chiffrée avec la clé K<sub>C-SERV01</sub> contenant des éléments d’authentification appelés “authenticator”. [^14]

![KRB_AP_REQ](/assets/img/Article%20Kerberos%20NTLM/Kerberos-KRB_AP_REQ.svg){: .shadow }

Lorsque le serveur hébergeant le service reçoit cette requête, il déchiffre le ST grâce à son hash et récupère la clé K<sub>C-SERV01</sub>. Le serveur peut ensuite déchiffrer la partie “authenticator” grâce à cette nouvelle clé et vérifier si l’utilisateur dispose des autorisations lui permettant d’accéder au service. [^15]

> Cette requête dispose d’un champ “AP-Options” prenant en charge une liste de drapeaux (flags) permettant d’activer des fonctionnalités protocolaires optionnelles. Le drapeau “mutal-required”, lorsqu’il est activé, va obliger le serveur à répondre par une requête KRB_AP_REP. Ce mécanisme permet l’authentification mutuelle du client et du serveur. 
{: .prompt-info }

#### AP_REP
Dans le cas où le serveur détermine que l’utilisateur a bien le droit d’accéder à son service et que le flag “mutual-required” est activé dans la requête **KRB_AP_REQ**, le serveur renvoie une réponse **KRB_AP_REP** contenant une partie chiffrée avec la clé K<sub>C-SERV01</sub>.

![KRB_AP_REP](/assets/img/Article%20Kerberos%20NTLM/Kerberos-KRB_AP_REP.svg){: .shadow }

Cette partie chiffrée contient des timestamps liés au client (ctime et cusec) que le client vérifiera pour authentifier le serveur, et une sous-clé (subkey) qui pourra être utilisée par le client et le serveur pour communiquer de manière sécurisée. [^16] [^17]

# Attaques sur le protocole Kerberos
### AS_REP roasting
L’attaque par **AS_REP** roasting s’effectue comme son nom l’indique au niveau de l’étape **“KRB_AS_REP”** c'est-à-dire après la demande de pré-authentification. Pour rappel, lors de cette étape, le client souhaitant s’authentifier récupère un TGT ainsi qu’une **clé de session chiffrée avec le hash du compte utilisateur**.

Dans certains cas, pour des raisons de compatibilité vis à vis de systèmes “Legacy” ou de mauvaise configuration, certains comptes n’envoient pas d’éléments permettant la pré-authentification de l’utilisateur (éléments chiffrés avec le hash de son compte) lors de la première étape KRB_AS_REQ.

Dans cette configuration, un attaquant a alors la possibilité de forger une requête KRB_AS_REQ pour un compte qui ne requiert pas de pré-authentification. Comme la réponse KRB_AS_REP contient des éléments chiffrés avec le hash de ce compte, l’attaquant sera en mesure d’effectuer une attaque par **brute-force** sur la **partie chiffrée** afin de retrouver le hash du compte et par extension, son mot de passe.  

Des outils comme `GetNPUsers.py` (suite Impacket) ou encore `Rubeus` peuvent être utilisés pour effectuer cette attaque.

### Kerberoasting
L’attaque par Kerberoasting s’effectue au niveau de la réponse **KRB_TGS_REP**, c'est-à-dire à l’envoi de ticket de service par le TGS.

Le principe de l’attaque est de faire des demandes de ticket de service pour tous les SPN du domaine. Une fois le ticket de service reçu, une attaque par brute-force peut être réalisée afin de récupérer le mot de passe du compte auquel le SPN est associé. En effet, le ST est chiffré par le hash du compte de service.

Cependant, pour la plupart des SPN, l’attaque ne sera pas possible car leurs services seront liés à des comptes machines. Ces derniers ont des mots de passes très longs, générés aléatoirement, ce qui rend ces comptes invulnérables à ce type d'attaque.

Toutefois, certains comptes de service sont générés par des utilisateurs humains, ce sont donc ces derniers qui vont être ciblés puisque possédant potentiellement un mot de passe faible, facilement vulnérable à une attaque par brute-force. 

Plusieurs outils peuvent être utilisés pour réaliser cette attaque tels que `GetUserSPNs.py` (suite Impacket) et `Rubeus`.

### Tickets attacks (silver, golden, diamond, sapphire)
Les attaques par tickets sont un type d’attaque permettant de forger des tickets valides permettant d’usurper l'identité d'autres utilisateurs ou d’octroyer des privilèges d’accès légitimes (PAC) afin d’accéder aux ressources désirées.

Ces attaques représentent un bon moyen de persistance pour un attaquant sur un domaine. 

Différents outils peuvent être utilisés pour effectuer ces attaques tels que `Mimikatz`, `Rubeus` ou encore `ticketer.py`.

### Silver Ticket
Le Silver Ticket est un Ticket de Service (ST) forgé par l’attaquant où la PAC (Privileged Attribute Certificate) inscrite dans le champ “authorization-data” a été modifiée pour y inscrire les informations d’un utilisateur privilégié (administrateur du domaine par exemple). 

Afin de réaliser cette attaque, l’attaquant doit être en possession du hash du compte de service ciblé. Ce hash est nécessaire pour pouvoir chiffrer le ST afin de le rendre valide.

Ce ticket fonctionnera bien car c’est la PAC qui va être vérifiée par le service pour décider si oui ou non l’utilisateur envoyant le ST (KRB_AP_REQ) est autorisé à accéder aux ressources du service.

### Golden Ticket
Le Golden Ticket est similaire dans son principe au Silver Ticket. La principale différence est qu’au lieu de forger un ST, l’attaquant va forger un TGT avec la PAC de l’utilisateur dont l’on souhaite usurper les droits.

![Golden Ticket](/assets/img/Article%20Kerberos%20NTLM/willywonka.png){: .shadow }

Pour pouvoir forger ce ticket, il doit être en possession du hash du compte krbtgt, préalablement extrait de la base de données de l’Active Directory grâce à des outils tels que `ntdsutil.exe` ou `Mimikatz` avec la technique DCSync.

Ce TGT forgé avec la PAC d’un utilisateur privilégié (Administrateur de domaine par exemple) permettra ainsi de faire des demandes de ST pour n’importe quel service de l’AD et ainsi avoir tous les droits sur le domaine.

### Diamond Ticket & Sapphire Ticket
Les Diamond et Sapphire tickets sont des versions améliorées du Golden Ticket. Le principe de fonctionnement est le même à une subtilité près : la mise en place de l’attaque.

Pour rappel, le Golden Ticket consiste en la création non légitime d’un TGT. Les méthodes de génération de ces tickets par le biais d’outils offensifs est cependant bien connue et l’absence de personnalisation dans le paramétrage des Golden Tickets rend leur utilisation facilement détectable. 

Les attaques par Diamond et Sapphire Ticket ont pour objectif de rendre l’utilisation de ces tickets plus difficiles à détecter. Dans le cas des Diamond Tickets, l’attaquant va effectuer une demande de TGT pour un utilisateur lambda, une fois le TGT reçu, ce dernier est déchiffré et la PAC modifiée. Le ticket est ensuite de nouveau chiffré . En utilisant la méthode du Diamond Ticket, l’attaquant dispose d’un TGT prenant en compte les configurations Kerberos de l’environnement ciblé, le rendant plus difficilement détectable.

Pour le Sapphire Ticket, le principe reste le même, mais au lieu de modifier manuellement la PAC, l’attaquant va demander à l’AS de fournir directement la PAC d’un utilisateur privilégié et de l’injecter dans le ticket. Cette méthode utilise deux mécanismes de Kerberos : **U2U** et **S4U2Self**. [^18]

---------
## Comparaison des protocoles

<table class="tab">
        <tr style="background-color: #f8485e;">
            <th style="color: white;">NTLM</th>
            <th style="color: white;">Kerberos</th>
        </tr>
        <tr style="color: #000000;">
            <td style="white-space: normal; background-color: #f4cccc;">Limité en matière d’interopérabilité avec les systèmes UNIX</td>
            <td style="white-space: normal; background-color: #d9ead3;">Interopérabilité avec les systèmes UNIX</td>
        </tr>
        <tr style="color: #000000;">
            <td style="background-color: #f4cccc;">Ne supporte pas la MFA</td>
            <td style="white-space: normal; background-color: #d9ead3;">Supporte la MFA</td>
        </tr>
        <tr style="color: #000000;">
            <td style="background-color: #f4cccc;">Ne permets pas l’authentification mutuelle</td>
            <td style="white-space: normal; background-color: #d9ead3;">Supporte l’authentification mutuelle</td>
        </tr>
        <tr style="color: #000000;">
            <td style="background-color: #f4cccc;">Ne supporte pas la délégation</td>
            <td style="white-space: normal; background-color: #d9ead3;">Supporte la délégation</td>
        </tr>
        <tr style="color: #000000;">
            <td style="background-color: #f4cccc;">Chiffrement faible (DES, HMAC-MD5)</td>
            <td style="white-space: normal; background-color: #d9ead3;">Peut-utiliser un chiffrement fort (AES-128, AES-256)</td>
        </tr>
        <tr style="color: #000000;">
            <td style="background-color: #d9ead3;">Fonctionnement plus simple que Kerberos</td>
            <td style="white-space: normal; background-color: #f4cccc;">Les machines doivent être synchronisées au niveau horaire car Kerberos utilise entre autres des timestamps pour valider l’authentification</td>
        </tr>
        <tr style="color: #000000;">
            <td style="background-color: #d9ead3;">Prends mieux en charge les systèmes legacy</td>
            <td style="white-space: normal; background-color: #f4cccc;">Chaque service doit être “kerbérisé”, donc adapté au protocole Kerberos</td>
        </tr>
</table>

Concernant les attaques liées à ces protocoles, il est intéressant de noter que les attaques sur NTLM sont inhérentes à la façon dont il a été créé alors que celles sur Kerberos sont plus souvent liées à des mauvaises configurations.

---------
## Conclusion : NTLM vs Kerberos
Depuis 2010, Microsoft recommande de ne plus utiliser NTLM au sein de l’environnement Active Directory lorsque cela est possible [^19]. L’utilisation de Kerberos pour l’authentification est préférable. 

En effet, Kerberos fournit une authentification mutuelle, un chiffrement fort des données et la possibilité d'utiliser la MFA. NTLM, de son côté, n'offre aucune de ces fonctionnalités.

Cependant, dans le cas où NTLM doit être conservé dû à des incompatibilités système ou applicative, plusieurs protections doivent être mises en place comme : 
- imposer l’utilisation de NTLMv2 (et refuser l’authentification via LM et NTLMv1)
- implémenter le signing (SMB, LDAP…) lorsque cela est possible
- utiliser le channel binding quand cela est possible 

De nos jours, les attaquants ont à leur disposition une multitude d’outils pour faciliter l’exploitation des failles de sécurité ou de mauvaises configurations, on pourra citer `Mimikatz`, `Rubeus`, `Impacket` ou encore les différents logiciels de Command & Control souvent remplis de fonctionnalités.

Alors que les protocoles d'authentification basé sur les mots de passes tels que NTLM ont été largement utilisés pour assurer la sécurité des réseaux d'entreprise, de nouvelles méthodes d'authentification ont émergé pour répondre aux besoins de sécurité en constante évolution. 

Parmi ces nouvelles méthodes, on trouve l'authentification **“passwordless”**, qui utilise des technologies comme la reconnaissance biométrique ou encore les clés de sécurité pour identifier les utilisateurs et leur accorder un accès sécurisé aux ressources du réseau. 

Cette méthode d'authentification est considérée comme plus sûre que les méthodes traditionnelles, car elle élimine la vulnérabilité des mots de passe qui peuvent être piratés ou oubliés.

---------
## Glossaire
<ins>Contrôleur de domaine / Domain Controller (DC) :</ins>
<br>
Au sein de l’Active Directory, le DC est le serveur central qui prend en charge l’ensemble des requêtes d’authentification et qui permet la gestion des utilisateurs, des machines, des politiques de sécurité ou encore la gestion logique des sites et des plages réseau au sein de l’annuaire Active Directory.

<ins>Centre de distribution de clés / Key Distribution Center (KDC) :</ins>
<br>
Le Key Distribution Center est un service exécuté au sein de chacun des DC et est impliqué dans le fonctionnement du protocole d’authentification Kerberos. Le KDC peut être décomposé en deux sous-services : l’Authentication Service (AS) et le Ticket Granting Service (TGS).

<ins>Service d’authentification / Authentication Service (AS) :</ins>
<br>
Le Service d’Authentification (AS) est un sous service du KDC impliqué dans l’étape de pré-authentification à un service. L’AS transmet un Ticket Granting Ticket (TGT) qui est nécessaire pour la suite de l’authentification du client. Le TGT fourni par ce service n’est valide que pour le ou les royaumes Kerberos autorisés. 

<ins>Ticket Granting Service (TGS) :</ins>
<br>
Le Ticket Granting Service est un sous service du KDC impliqué dans les demandes d’accès aux services. Le TGS fournit les Service Tickets (ST) demandés par le client et permettent de contacter ces derniers pour la suite de l’authentification..

<ins>Ticket Granting Ticket (TGT) :</ins>
<br>
Ticket généré par l’AS du domaine lors de l’étape KRB_AS_REP et donné au client. Ce ticket est par la suite utilisé par le client pour faire une requête de ST auprès du TGS (étape KRB_TGS_REQ). 

<ins>Ticket de Service / Service Ticket (ST) :</ins>
<br>
Ticket généré par le TGS du domaine lors de l’étape KRB_TGS_REP permettant au client d’accéder à un service du domaine (étape KRB_AP_REQ).

<ins>Royaume Kerberos (realm) :</ins>
<br>
Le royaume Kerberos est un domaine administratif et non physique. Il définit les limites de contrôle du KDC. Un utilisateur, un ordinateur ou un service appartient à un royaume Kerberos. Cependant, il est possible d’établir des relations entre royaume pour permettre l’authentification entre utilisateurs et services de royaumes différents.

<ins>krbtgt :</ins>
<br>
Le compte krbtgt est, dans un domaine, le compte lié au KDC, ce dernier l’utilise notamment pour générer des TGT.

<ins>Service Principal Name (SPN) :</ins>
<br>
Un SPN est utilisé pour identifier les services au sein de l’Active Directory. Il est notamment utilisé dans les tickets et requêtes Kerberos.

<ins>Privilege Attribute Certificate (PAC) :</ins>
<br>
La PAC est un élément souvent inclus (mais facultatif) dans les tickets Kerberos. La PAC contient des éléments d’autorisation supplémentaires. Cette partie est chiffrée dans le ticket.

<ins>Hash NT ou hash NTLM :</ins>
<br>
Le hash NT d’un mot de passe est son empreinte MD4. 
Le hash NTLM est composé d’une partie LM “vide” et de sa partie NT.

---------
> La première version de cet article a été rédigée lors de mon stage chez Devoteam. Merci à eux !
[https://france.devoteam.com/paroles-dexperts/la-securite-des-protocoles-dauthentification-ntlm-et-kerberos-en-environnement-active-directory/](https://france.devoteam.com/paroles-dexperts/la-securite-des-protocoles-dauthentification-ntlm-et-kerberos-en-environnement-active-directory/)
{: .prompt-info }

---------
## Références
[^1]: “[MS-APDS]: NTLM Pass-Through Authentication,” learn.microsoft.com, Jul. 04, 2021. [https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-apds/5bfd942e-7da5-494d-a640-f269a0e3cc5d](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-apds/5bfd942e-7da5-494d-a640-f269a0e3cc5d) (accessed Mar. 08, 2023).

[^2]: “[MS-NLMP]: NEGOTIATE_MESSAGE,” learn.microsoft.com, Aug. 01, 2022. [https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/b34032e5-3aae-4bc6-84c3-c6d80eadf7f2](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/b34032e5-3aae-4bc6-84c3-c6d80eadf7f2) (accessed Mar. 09, 2023).

[^3]: “[MS-NLMP]: CHALLENGE_MESSAGE,” learn.microsoft.com, Aug. 01, 2022. [https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/801a4681-8809-4be9-ab0d-61dcfe762786](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/801a4681-8809-4be9-ab0d-61dcfe762786) (accessed Mar. 09, 2023).

[^4]: “[MS-NLMP]: AUTHENTICATE_MESSAGE,” learn.microsoft.com, Aug. 01, 2022. [https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/033d32cc-88f9-4483-9bf2-b273055038ce](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/033d32cc-88f9-4483-9bf2-b273055038ce) (accessed Mar. 09, 2023).

[^5]: “[MS-NRPC]: NETLOGON_NETWORK_INFO,” learn.microsoft.com, Apr. 07, 2021. [https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nrpc/e17b03b8-c1d2-43a1-98db-cf8d05b9c6a8](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nrpc/e17b03b8-c1d2-43a1-98db-cf8d05b9c6a8) (accessed Mar. 09, 2023).

[^6]: “[MS-NRPC]: NETLOGON_VALIDATION_SAM_INFO4,” learn.microsoft.com, Apr. 27, 2022. [https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nrpc/bccfdba9-0c38-485e-b751-d4de1935781d](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nrpc/bccfdba9-0c38-485e-b751-d4de1935781d) (accessed Mar. 09, 2023).

[^7]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-5.3](https://www.rfc-editor.org/rfc/rfc4120#section-5.3) (accessed Mar. 09, 2023).

[^8]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-5.4.1](https://www.rfc-editor.org/rfc/rfc4120#section-5.4.1) (accessed Mar. 09, 2023).

[^9]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-3.1.2](https://www.rfc-editor.org/rfc/rfc4120#section-3.1.2) (accessed Mar. 09, 2023).

[^10]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-5.4.2](https://www.rfc-editor.org/rfc/rfc4120#section-5.4.2) (accessed Mar. 09, 2023).

[^11]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-3.1.5](https://www.rfc-editor.org/rfc/rfc4120#section-3.1.5) (accessed Mar. 09, 2023).

[^12]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-3.3.2](https://www.rfc-editor.org/rfc/rfc4120#section-3.3.2) (accessed Mar. 09, 2023).

[^13]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-3.3.4](https://www.rfc-editor.org/rfc/rfc4120#section-3.3.4) (accessed Mar. 09, 2023).

[^14]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-5.5.1](https://www.rfc-editor.org/rfc/rfc4120#section-5.5.1) (accessed Mar. 09, 2023).

[^15]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-3.2.3](https://www.rfc-editor.org/rfc/rfc4120#section-3.2.3) (accessed Mar. 09, 2023).

[^16]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-5.5.2](https://www.rfc-editor.org/rfc/rfc4120#section-5.5.2) (accessed Mar. 09, 2023).

[^17]: C. Neuman, T. Yu, S. Hartman, and K. Raeburn, “The Kerberos Network Authentication Service (V5),” Jul. 2005. [https://www.rfc-editor.org/rfc/rfc4120#section-3.2.5](https://www.rfc-editor.org/rfc/rfc4120#section-3.2.5) (accessed Mar. 09, 2023).

[^18]: O. S. Roitman Shachar, “Precious Gemstones: The New Generation of Kerberos Attacks,” Unit 42, Dec. 12, 2022. [https://unit42.paloaltonetworks.com/next-gen-kerberos-attacks/#post-126011-_k56ak9qw5q5d](https://unit42.paloaltonetworks.com/next-gen-kerberos-attacks/#post-126011-_k56ak9qw5q5d) (accessed Mar. 09, 2023).

[^19]: “[MS-NLMP]: Security Considerations for Implementers,” learn.microsoft.com, Oct. 01, 2020. [https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/1e846608-4c5f-41f4-8454-1b91af8a755b](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/1e846608-4c5f-41f4-8454-1b91af8a755b) (accessed Mar. 09, 2023).