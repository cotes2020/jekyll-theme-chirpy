---
title: Docker - basic
date: 2020-07-16 11:11:11 -0400
categories: [30System, Docker]
tags: [Docker]
math: true
image:
---


# basic

- [basic](#basic)
  - [command](#command)
  - [Accessing the Container](#accessing-the-container)
  - [container management](#container-management)


---

## command

```bash

docker ps
docker container ls
docker container ls -a

docker images

docker run -d -p 3000:80 -p 8080:80 contanerimage
docker stop contanerid

docker start container_name
docker stop container_name

docker run -i alpine
docker run -it --name a-container alpine
docker run -it --name a-test -rm alpine   # remove once stopped

docker run -dt --restart always --name bg-container alpine   # persist
docker run -dt --restart unless-stopped/on-failure/no


docker image ls

# clean
docker rmi $(docker images --filter "dangling=true" -q --no-trunc)
docker image rm image:tag

# get image digest:
docker inspect image:tag \
    | jq -r '.[0].RepoDigests[0]' \
    | cut -d'@' -f2


```

## Accessing the Container

docker exec `<container> <command>`

docker exec -it `<container> <command>`

```py

docker exec a-container apk add nginx

docker exec a-container ls

docker exec -it a-container ash

docker cp a-container:/etc/nginx/conf.d/default.conf .

docker cp default.conf a-container:/etc/nginx/conf.d/default.conf

```


## container management


```py
docker image ls
# clean
docker rmi $(docker images --filter "dangling=true" -q --no-trunc)
docker image rm image:tag

# stop container
docker stop container_name
docker rm container_name
docker restart container_name

docker container prune # stop all stopped container


# change name
docker rename a-container web01


# know info
docker stats <container_name>
```









.
