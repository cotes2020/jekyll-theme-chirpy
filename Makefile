ROOT = .
DOCKER_COMPOSE_FILE = $(ROOT)/docker-compose.yml
DOCKER_COMPOSE = docker-compose -f $(DOCKER_COMPOSE_FILE)

DEFAULT_GOAL := help
.PHONY: help
help:
	@printf "[Makefile] Usage\n\tmake \033[36m<target>\033[0m\n"
	@count=`awk '/^.*-->@.*@$$/ {count++} END {print count}' $(MAKEFILE_LIST)`; index=0; while [ $$index -lt $$count ] ; do \
  		index=`expr $$index + 1`; \
  		regex="/^.*-->@$$index.*@$$/"; awk "$$regex "'{ printf "\n%s\n", substr($$0, 9, length($$0) - 9) }' $(MAKEFILE_LIST); \
  		fs='BEGIN {FS = ":.*==>@'$$index'"}'; regex="/^[a-zA-Z0-9_-]+:.*?==>@$$index/"; awk "$$fs $$regex "'{ printf "\t\033[36m%-27s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST); \
	done

.env:
	rsync --ignore-existing .env.example .env

include .env
##-->@1 [Docker] Automatized Management of Project@
.PHONY: up
up: .env ##==>@1 Start all docker containers. To only start one container, use CONTAINER=<service>
	make clear
	$(DOCKER_COMPOSE) up --build -d $(CONTAINER)

.PHONY: down
down: ##==>@1 Stop all docker containers. To only stop one container, use CONTAINER=<service>
	$(DOCKER_COMPOSE) down $(CONTAINER)
	make clear

.PHONY: run
run: ##==>@1 Run command (force new container) on specific tool container. To use CONTAINER=<service> COMMAND=<command>
	$(DOCKER_COMPOSE) run --rm $(CONTAINER) $(COMMAND)

.PHONY: exec
exec: ##==>@1 Exec command on specific container. To use CONTAINER=<service> COMMAND=<command>
	$(DOCKER_COMPOSE) exec -T $(CONTAINER) $(COMMAND)


##-->@2 [Jekyll] Automatized Build of Framework@
.PHONY: clear
clear: ##==>@2 Clear all temporary and build files
	sudo rm -Rf .jekyll-cache _data/updates.yml _site/* categories tags .jekyll-metadata Gemfile.lock

.PHONY: build
build: ##==>@2 Publish site to "_site" (default) folder with production URL
	make run CONTAINER="app" COMMAND="env JEKYLL_ENV=production bundle exec jekyll build --incremental"
	sudo chown -R $$USER:$$USER _site
