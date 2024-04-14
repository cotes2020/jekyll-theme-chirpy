DOCKER_USERNAME ?= opslogic
APPLICATION_NAME ?= tr-blog
TAG ?= u2204
 
build:
	docker build --tag docker.io/${DOCKER_USERNAME}/${APPLICATION_NAME}:${TAG} .
push:
	docker push ${DOCKER_USERNAME}/${APPLICATION_NAME}:${TAG}
