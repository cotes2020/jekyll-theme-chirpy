---
title: TryHackMe - Agent Sudo
date: 2023-01-01 19:05:20 +800
categories: [TryHackMe, -Eazy]
tags: [sudo]
mermaid: true
---
#Prueba
<h2 data-toc-skip>H2 - heading</h2>

## Information Gathering
---
## Enumeration
### Escaneo
Luego de un escaneo para descubrir los puertos abiertos, procedemos a escanear las versiones y scripts por default.

## Analisis Web
Ingresando a la Web con la IP, observamos un mensaje.
    alt: imagen
Podemos intuir que en la cabecera `User-agent` ira el `codename`, para descubrir el caracter correcto usare Burpsuite
### Burpsuite
-Prueba de respuesta

-Intruder

---
## Explotation
Ya sabemos el nombre del usuario, lo que sigue es realizar un ataque de fuerza bruta con `Hydra`
### Fuerza Bruta

Luego de tener las credenciales, accedemos por `ftp`.
Encontrando 2 imagenes, los copio a mi terminal para analizarlo.

### Stenography y crackeo
Con la tool `Binwalk` observamos si estas imagenes contienen datos adicionales.

Verificamos que contiene un ZIP.Extraemos con `binwalk`
```
binwalk -e cutie.png
```
-Crackeo
Pero al intentar descomprimir nos indica que debemos usar una clave, para esto usaremos `zip2john`, de esta manera tendremos un hash al cual crackear con John the Ripper.

Luego del crackeo usaremos esta key para descomprimir


Esto nos extrae un documento .txt, el cual nos brindara una supuesta clave, la cual esta codificada en base64.


-stehide
Con la password anterior podemos tratar de extraer informacion de la otra imagen 

Brindando credenciales.De esta manera accedemos por ssh al sistema.

---
## Privilage Escalation
Enumeramos el sistema y econtramos un comando que se ejecuta como root.

En la web `GTFOBins` nos indica como escalar privilegios segun la version del sudo.

Logrando asi obtener el acceso
