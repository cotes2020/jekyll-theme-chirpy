---
title: Como programar posts con Jekyll en GitHub Pages.
author: Marcos Ramírez
date: 2021-02-26 21:00:00 +0100
categories: [Informática, Jekyll]
tags: [Configurar, Jekyll, posts, programados, diferidos, deferred, posts]
pin: false
toc: true
excerpt_separator: <!--more-->
excerpt: Excerpt
permalink: /:title/ # title is filename NOT title in YAML
---

El primer problema "serio", que me he encontrado con Jekyll, es que no podía programar los posts, crear un post con la fecha en futuro, no es suficiente para que se publique, también hay que hacer que se ejecute el build, ¿como?, simplemente modificando el archivo:

| .github/workflows/pages-deploy.yml

Y añadiendo esto:

```yaml
on:
  schedule:
    - cron: '*/30 * * * *' # Runs every 30 mins

```

Con esto, forzaremos un build cada media hora, que hará que ahora ya sí se publiquen los posts programados.


Otra opción (menos elegante), es forzar el rebuild, ejecutando un push vacío desde nuestro local, o donde sea:

```bash
git commit -m 'Force Rebuild' --allow-empty
git push origin <branch-name>
```


***Y ahora ya solo tendrás que escribir y hacer push de tus posts con fecha futura, y se publicarán automáticamente***.



Espero que os haya sido útil
