---
title: Docker - create dockerfiles images
date: 2020-07-16 11:11:11 -0400
categories: [30System, Docker]
tags: [Docker]
math: true
image:
---

# create dockerfiles images

- [create dockerfiles images](#create-dockerfiles-images)
  - [basic](#basic)
  - [build 1](#build-1)
    - [Dockerfile](#dockerfile)
  - [Image Management](#image-management)
  - [build 2](#build-2)
    - [Dockerfile](#dockerfile-1)

---

## basic

hello world

```bash
FROM scratch
COPY hello /
CMD ["/hello"]
```

## build 1

```bash

~ $ wget --content-disposition 'https://github.com/linuxacademy/content-Introduction-to-Containers-and-Docker/raw/master/lessonfiles/demo-app.tar'

~ $ tar -xf demo-app.tar
~ $ ls
app  code   default.conf  demo-app.tar
Desktop    Downloads   Music	 Public  Templates  Videos
bin  code2  default.conf   demos		Documents  index.html  Pictures  sample  venvs

~ $ cd app

app $ ls
index.js  node_modules	nodesource_setup.sh
package.json  package-lock.json  public  views


app $ vim Dockerfile

docker build . -t appimage

docker run -dt --name app01 appimage

docker inspect app01 | grep IPAddress

curl 172.17.0.2:8080
```


### Dockerfile

```bash

# define parent image with FROM; all valid Dockerfiles must begin with FROM.
FROM node:10-alpine

# create a directory and set its owner.
# use the RUN command, execute any commands against the shell. Our image is based on Alpine, which uses the ash shell
RUN mkdir -p /home/server/app/node_modules && chown -R server:server /home/server/app

# move into the /home/node/app directory. We do this by setting a working directory with WORKDIR. Any RUN, CMD, COPY, ADD, and ENTRYPOINT instructions that follow will be performed in this directory:
WORKDIR /home/server/app

# to add our files.
# two options for this: COPY and ADD.
# ADD can pull files from outside URLs, and thus utilizes additional functionality.
# COPY: package* files are all local
COPY package*.json ./
# With the first argument being our source file(s) and the second our destination.

# ensure the prerequisite packages for our application are installed from NPM. These are just more RUN commands.
RUN npm config set registry http://registry.npmjs.org/
RUN npm install


# We can now copy over the rest of our application files using COPY. We'll also want to set the owner of our working directory to the node user and group. Since we're on Linux, this can also be achieved using some special COPY functionality.
COPY --chown=server:server . .

# to switch users. This works similarly to the previous WORKDIR command, but now we're switching users not directories for any following RUN, CMD, and ENTRYPOINT commands.
USER server

# Our application is hosted on port 8080, so we also want to make sure that port is available on our container. For this, we'll use the EXPOSE keyword.
EXPOSE 8080

# And now, finally, we want to provide the command (CMD) run as soon as the container is launched. Unlike RUN, CMD's preferred format doesn't take this as a shell command. Instead, the executable name should be supplied in an array, followed by any parameters: CMD ["executable","param1","param2"]. In our case, this will be:
CMD [ "node", "index.js" ]
```


## Image Management

```bash
docker pull image_name
docker image ls
docker image rm image_name

docker image prune -a   # remove useless image

docker image inspect
```



## build 2

```bash
cloud_user@dockerhost:~$ ls
containerhub

cloud_user@dockerhost:~$ cd containerhub/

cloud_user@dockerhost:~/containerhub$ ls
files
cloud_user@dockerhost:~/containerhub$ ls files/
default.conf  html
cloud_user@dockerhost:~/containerhub$ cat files/default.conf
server {
        listen 80 default_server;
        listen [::]:80 default_server;

        root /var/www/html/;
}


cloud_user@dockerhost:~/containerhub$ vim Dockerfile

cloud_user@dockerhost:~/containerhub$ docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS      PORTS               NAMES
cloud_user@dockerhost:~/containerhub$ docker images ls
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE

cloud_user@dockerhost:~/containerhub$ docker build . -t web

cloud_user@dockerhost:~/containerhub$ docker run -dt -p 80:80 --name web01 web
f0cd9a64eee810886db2229f233191eb2ef11de1d537f1410b0f9e6306bcaf36


cloud_user@dockerhost:~/containerhub$ curl localhost
<html>
<head><title>403 Forbidden</title></head>
<body>
<center><h1>403 Forbidden</h1></center>
<hr><center>nginx</center>
</body>
</html>

cloud_user@dockerhost:~/containerhub$ docker container ls
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS           PORTS                NAMES
f0cd9a64eee8        web                 "nginx -g 'pid /tmp/…"   2 minutes ago       Up 2 minutes        0.0.0.0:80->80/tcp   web01
```

### Dockerfile

```
FROM alpine:latest
RUN apk upgrade
RUN apk add nginx
COPY files/default.conf /etc/nginx/conf.d/default.conf
RUN mkdir -p /var/www/html
WORKDIR /var/www/html
COPY --chown=nginx:nginx /files/html/ ,
EXPOSE 80
CMD ["nginx", "-g", "pid /tmp/nginx.pid; daemon off;"]
```





.
