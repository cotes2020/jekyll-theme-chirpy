---
title: Docker+Django
author: Кирьянов Артем
date: 2021-12-27 19:00:00 +0300
categories: [IT, DevOps,Django]
tags: [Docker,Django]
---

## Django, PostgreSQL и Docker

В этом статье, создадим новый проект Django, используя Docker, PostgreSQL, Gunicorn. Я вообще стараюсь для локальной разработки повторять продуктив и использовать близкие настройки. 

Зачем использовать Docker ? Да потому что тупо лень настраивать систему под проект, с Docker проще работать. Меньше слов, больше дел. 
Кому лень, вот репозиторий с моим тестовым шаблоном [Скачать](https://github.com/hacker342/docker-django)


# Ставим Docker

Тут все просто. Ставим Docker для вашей ОС с сайта [docker](https://www.docker.com/). Помним, что Docker для корпоративного сегмента, Desktop версия стала платная, поэтому не используем его без особых разрешений не ставим на корпоративную технику. Я же поставлю его без особых раздумий на свой макбук:)

##  Проект Django

Создайте новый каталог проекта вместе с новым проектом Django. На момент написания статьи, версия Django = 4.0:
```
    $ mkdir django && cd django
    $ mkdir django-env && cd django-env
    $ python3.10 -m venv env
    $ source env/bin/activate
    (env)$ pip install django==4.0
    (env)$ django-admin.py startproject hello_django .
    (env)$ python manage.py runserver
 ```
Перейдем по адресу http://localhost:8000/ для просмотра экрана приветствия Django. Остановим сервер и выйдем из виртуальной среды. Теперь у нас есть простой проект Django для работы. Мы создали костяк для дальнейшей работы Django with Docker. 

Дальше можно конечно создать файл _requirements.txt_, прописать туда версию Django и т.п, потом снова туда дописывать - так долго и не интересно. Проще создать сразу структуру Dockerfile, docker-compose.yml и _requirements.txt_. Давайте это и сделаем

## Структура

Общая структура выглядит вот так: 
![enter image description here](https://lh3.googleusercontent.com/pw/AM-JKLUdOkAloz3q2OLAWMu9WCz0YDwnEO8hgxHtroaldpvH3afpvDGotvx8GXz2WUOMTh5ospwSorf64uYYR64pw8wAYc23HP0QmVvnxqKa1Or2JuLiTIs-IlUoK77cVeOMfwx9C57RPqkO7epcoxoaHpf8PA=w696-h570-no?authuser=0)


## Dockerfile
Я собираюсь создать Dockerfile в корне проекта. Dockerfile - это текстовый документ, содержащий все команды, которые пользователь может вызвать в командной строке для сборки образа.
```
# pull official base image
FROM python:3.8.3-alpine

# set work directory
WORKDIR /code

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apk update \
    && apk add postgresql-dev gcc python3-dev musl-dev


# install dependencies
RUN pip install --upgrade pip
COPY requirements.txt /code/
RUN pip install -r requirements.txt

COPY ./entrypoint.sh .

# copy project
COPY . /code/
RUN chmod +x /code/entrypoint.sh
ENTRYPOINT ["/code/entrypoint.sh"]
```

## docker-compose
Теперь  создадим новый файл под названием _docker-compose.yml_. Согласно Docker: Compose - это инструмент для определения и запуска многоконтейнерных приложений Docker. В Compose вы используете файл YAML для настройки служб вашего приложения. Затем с помощью одной команды вы создаете и запускаете все службы из своей конфигурации.

Файл набора описывает службы, из которых состоит ваше приложение. Как вы можете видеть на изображении ниже, существует 3 сервиса: nginx, web, db.

Файл набора также описывает, какие образы Docker используют эти службы, как они связываются друг с другом, а также любые тома, которые, возможно, потребуется смонтировать внутри контейнеров. Наконец, файл набора также описывает, какие порты открывают эти службы.
```
version: '3.9'
services:
  nginx:
    container_name: dev_web
    restart: on-failure
    image: nginx:1.19.8
    volumes:
      - ./nginx/dev/nginx.conf:/etc/nginx/conf.d/default.conf
      - static_volume:/code/static
    ports:
      - 80:80
    depends_on:
      - web
  web:
    container_name: dev_backend
    build: .
    restart: always
    env_file: dev.env
    command: sh -c "gunicorn --bind 0.0.0.0:5000 djangotgprojects.wsgi"
    volumes:
     - .:/code
     - static_volume:/code/static
    depends_on:
     - db
  db:
    container_name: dev_db
    image: postgres:12.0-alpine
    env_file: dev.env
    volumes:
      - postgres_data:/var/lib/postgresql/data/

volumes:
  static_volume:
  postgres_data:
  ```
  

## NGINX

Как показано в приведенной ниже конфигурации, nginx отправляет прокси-запрос на вышестоящий сервер, работающий на порту 5000, который указывает на наше внутреннее приложение.
```
upstream demo_project {
    server web:5000;
}
server {

    listen 80;

    location / {
        proxy_pass http://demo_project;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_redirect off;
    }

    location /static/ {
     alias /app/static/;
    }
}
```

## env файлы
Я вообще стараюсь работать и хранить какую-то информацию в переменных окружения. Создадим в корне dev.env и prod.env, названия говорят сами за себя. 
Ниже пример dev.env , если будете пушить это дело в Github в публичную репу, не забываем это дело добавить в .gitignore или убрать все это дело в приватный репозиторий. Поэтому в моем Github данных файлов нет. Короче создайте по аналогии env файлы в корне проекта.
```
DEBUG=1
SECRET_KEY=django-insecure-_g)1a3yjwm4jqij*iae_n=f*ldvv16ddsq2#ky0+l!qb51q=6c
DJANGO_ALLOWED_HOSTS=localhost 127.0.0.1 [::1]

POSTGRES_USER="hello_django"
POSTGRES_PASSWORD="hello_django"
POSTGRES_DB="hello_django_dev"
DB_NAME="hello_django_dev"
DB_USERNAME="hello_django"
DB_PASSWORD="hello_django"
DB_HOST="db"
DB_PORT="5432"
DJANGO_SETTINGS_MODULE="djangotgprojects.settings"
```


## settings.py

Необходимо еще поменять пару строчек в settings.py. 
В начале добавим import
```
import os
```
Затем меняем подключение к БД на
```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME'),
        'USER': os.getenv('DB_USERNAME'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST'),
        'PORT': os.getenv('DB_PORT')
    }
}
```
Обновим переменные **SECRET_KEY**, **DEBUG** и **ALLOWED_HOSTS** в settings.py:
```
SECRET_KEY = os.environ.get("SECRET_KEY")
DEBUG = int(os.environ.get("DEBUG", default=0))
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS").split(" ")
```

Вроде бы все, но если что-то упустил, вы можете скачать шаблон с моего Github, вот отсюда [Скачать рыбу](https://github.com/hacker342/docker-django). Не забываем добавить _dev.env_

Собираем образ командой:
```
docker-compose build
```
Как только образ будет собран, запускаем контейнер:
```
docker-compose up -d
```
Далее нужно перейти по адресу http://localhost:8000/, чтобы снова увидеть экран приветствия и убедиться что все работает.

Проверьте наличие ошибок в журналах, если это не работает, через команду:
```
docker-compose logs -f
```
Докер настроен!

Так как мы указали все необходимые доступы для PostgreSQL, произведем миграцию: 
```
docker-compose exec web python manage.py migrate --noinput
```
Если словите ошибку миграции, то остановите контейнер командой **docker-compose down -v**, чтобы удалить тома вместе с контейнерами. Затем заново создайте образы, запустите контейнеры и примените миграции.

Убедимся, что все таблицы Django по умолчанию были созданы:
```
docker-compose exec db psql --username=hello_django --dbname=hello_django_dev
```
Получим вот такой вывод:
![enter image description here](https://lh3.googleusercontent.com/pw/AM-JKLXx-wz6CqBIcvo1camd_kf6H72Gz3GE6FR_I7nyhz7T_xWKRjDC8GCoRv_aSHFvD1ATucB_lYyjUhSSFIGM8OxJX6MwKp_FfS0vkw9cd1Oi5fYwctW9VyT4eL3xb1Yw-G3st_M_Xcrba2qNNkq8pazASg=w1604-h1016-no?authuser=0)

_\c_ - подключение к БД по имени
_\dt_ - перечисляет все таблицы в базе данных

Вы также можете проверить, что том (volume) был создан, запустив команду:
```
docker volume inspect djangoprojecttelegram_postgres_data
```
Получим вот такой вывод:
```
[
    {
        "CreatedAt": "2021-12-24T08:49:39Z",
        "Driver": "local",
        "Labels": {
            "com.docker.compose.project": "djangoprojecttelegram",
            "com.docker.compose.version": "2.2.1",
            "com.docker.compose.volume": "postgres_data"
        },
        "Mountpoint": "/var/lib/docker/volumes/djangoprojecttelegram_postgres_data/_data",
        "Name": "djangoprojecttelegram_postgres_data",
        "Options": null,
        "Scope": "local"
    }
]
```
Ну все, теперь можно создавать приложения и писать логику, ну это уже в следующей статье.

## entrypoint.sh
Давайте добавим файл **_entrypoint.sh_** в каталог нашего проекта, чтобы проверить работоспособность Postgres перед применением миграций и запуском сервера разработки Django:
```
#!/bin/sh

if [ "$DATABASE" = "postgres" ]
then
    echo "Waiting for postgres..."

    while ! nc -z $DB_HOST $DB_PORT; do
      sleep 0.1
    done

    echo "PostgreSQL started"
fi

python manage.py flush --no-input
python manage.py migrate

exec "$@"
```
Обновим локальные права доступа к файлу:
```
chmod +x entrypoint.sh
```
Затем обновим Dockerfile, чтобы скопировать файл  **entrypoint.sh**  и запустите его как команду точки входа Docker:
```
# pull official base image
FROM python:3.8.3-alpine

# set work directory
WORKDIR /code

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apk update \
    && apk add postgresql-dev gcc python3-dev musl-dev


# install dependencies
RUN pip install --upgrade pip
COPY requirements.txt /code/
RUN pip install -r requirements.txt

COPY ./entrypoint.sh .

# copy project
COPY . /code/
RUN chmod +x /code/entrypoint.sh
ENTRYPOINT ["/code/entrypoint.sh"]
```
Проверим все снова:

1.  Пересоберем заново образы
2.  Запустим контейнеры
3.  Перейдем на страницу [http://localhost:8000/](http://localhost:8000/)


## Примечание
Во-первых, несмотря на добавление Postgres, мы все равно можем создать независимый образ Docker для Django, если для переменной среды DATABASE не задано значение **postgres**. Чтобы проверить, создайте новый образ и затем запустите новый контейнер:
```
docker build -f ./app/Dockerfile -t hello_django:latest ./app
docker run -d \
    -p 8006:8000 \
    -e "SECRET_KEY=please_change_me" -e "DEBUG=1" -e "DJANGO_ALLOWED_HOSTS=*" \
    hello_django python /usr/src/app/manage.py runserver 0.0.0.0:8000
```
Вы должны увидеть страницу приветствия по адресу http://localhost:8006.

Во-вторых, вы можете закомментировать команды очистки (flush) и миграции (migrate) базы данных в сценарии entrypoint.sh, чтобы они не запускались при каждом запуске или перезапуске контейнера:
```
#!/bin/sh

if [ "$DATABASE" = "postgres" ]
then
    echo "Waiting for postgres..."

    while ! nc -z $DB_HOST $DB_PORT; do
      sleep 0.1
    done

    echo "PostgreSQL started"
fi

#python manage.py flush --no-input
#python manage.py migrate

exec "$@"
```
Вместо этого вы можете запустить их вручную, после того, как контейнеры запускаться, вот так:
```
docker-compose exec web python manage.py flush --no-input
docker-compose exec web python manage.py migrate
```

## Небольшой список команд Docker

Когда вы закончите, не можете погасить контейнер Docker.
```
docker-compose down
```
Просто приостановить контейнер
```
docker stop CONTAINER ID
```
Запустить ранее остановленный контейнер
```
docker start CONTAINER ID
```
Перегрузить контейнер
```
docker restart CONTAINER ID
```
Что бы посмотреть работающие контейнеры
```
docker ps
```
Что бы посмотреть вообще все контейнеры
```
docker ps -a
```
Посмотреть список всех образов
```
docker images
```
Удалить образ
```
docker rmi CONTAINER ID
или
docker rmi -f CONTAINER ID
```
Иногда может понадобиться зайти в работающий контейнер. Для этого нужно запустить команду запуска интерактивной оболочкой **bash**
```
docker exec -it CONTAINER ID bash
```
Как вы увидели из моего docker-compose.yml файла, я использую Gunicorn

Что же такое Gunicorn?

**_Gunicorn_** -  переводит запросы, полученные от Nginx, в формат, который может обрабатывать ваше веб-приложение, и обеспечивает выполнение кода при необходимости. Как только Nginx решит, что конкретный запрос должен быть передан в Gunicorn (согласно правилам, по которым он был настроен), в работу вступает Gunicorn.

## ПРОМ
Поскольку мы все еще хотим использовать встроенный сервер Django для разработки, создайте новый файл compose под названием **docker-compose.prod.yml** для производственной среды:

```
version: '3.9'
services:
  nginx:
    container_name: prom_web
    restart: on-failure
    image: nginx:1.19.8
    volumes:
      - ./nginx/prod/nginx.conf:/etc/nginx/conf.d/default.conf
      - static_volume:/app/static
    ports:
      - 80:80
    depends_on:
      - web
  web:
    build:
      context: .
      dockerfile: Dockerfile.prod
    restart: always
    env_file: prom.env
    command: sh -c "gunicorn --bind 0.0.0.0:8000 djangotgprojects.wsgi"
    volumes:
      - .:/app
      - static_volume:/app/static
    depends_on:
      - db
  db:
    container_name: dev_db
    image: postgres:12.0-alpine
    env_file: prom.env
    volumes:
      - postgres_data:/var/lib/postgresql/data/

volumes:
  static_volume:
  postgres_data:
```
Я не стал добавлять сюда какие-то дополнительные параметры, а просто изменил имена контейнеров и env.

>Если у вас несколько сред, вы можете использовать конфигурационный файл **docker-compose.override.yml**. При таком подходе вы добавляете базовую конфигурацию в файл docker-compose.yml, а затем используете файл docker-compose.override.yml для переопределения этих параметров конфигурации в зависимости от среды.

После создания необходимых файлов, запустим сборку
```
docker-compose -f docker-compose.prod.yml up -d --build
```
Вывод Docker
![enter image description here](https://lh3.googleusercontent.com/pw/AM-JKLVywKVjOnNvGIGE0DvvkAkC6McFTtSeDyl8Dk_hwDxeGwJSh0qBibxAZ9NIeiKVxJzEQoYAV1vjydF0LprNFiMDcjdmuskMHtmLa4FcysDbCm1DWo-2eskoeDdcZj3_rmOwEBoEcUxgeWyU7skrlRHNrA=w1948-h154-no?authuser=0)

Отлично! Как видим все у нас применилось! Проверяем:
![enter image description here](https://lh3.googleusercontent.com/pw/AM-JKLVDbHEfLIhvHsAkD-fr08uImrGDV4htJKZnVDzMsynvnVSL9-zjR0VkFNbVCw0XnqjryyEh_HeRuiKTuu5fo-EyOVDSYTFtstxJVmJUew4yjD3uMnGze_QSchIw4Jp7ScXYEuCvRdJ4T4m3R3rT3ZhuiQ=w2346-h1500-no?authuser=0)

## Производственный Dockerfile
Вы заметили, что мы все еще выполняем очистку базы данных (flush)(которая очищает базу данных) и переносим команды при каждом запуске контейнера? Это хорошо в разработке, но давайте создадим новый файл точки входа для промышленной эксплуатации.
Файл _entrypoint.prod.sh_:
```
#!/bin/sh

if [ "$DATABASE" = "postgres" ]
then
    echo "Waiting for postgres..."
    while ! nc -z $DB_HOST $DB_PORT; do
      sleep 0.1
    done
    echo "PostgreSQL started"
fi
exec "$@"
```
Обновим права доступа к файлу:
```
chmod +x entrypoint.prod.sh
```
Вообще эту _chmod_ я добавил в Dockerfile
Чтобы использовать этот файл, создайте новый Dockerfile с именем Dockerfile.prod для использования с ПРОМ сборками:
```
###########
# BUILDER #
###########

# pull official base image
FROM python:3.8.3-alpine as builder

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install psycopg2 dependencies
RUN apk update \
    && apk add postgresql-dev gcc python3-dev musl-dev libffi-dev openssl-dev

# install other dependencies
RUN apk --update add \
    build-base \
    jpeg-dev \
    zlib-dev

# lint
# RUN pip install --upgrade pip
# RUN pip install flake8
# COPY . .
# RUN flake8 --ignore=E501,F401 .

# install dependencies
COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt


#########
# FINAL #
#########

# pull official base image
FROM python:3.8.3-alpine

# create directory for the app user
RUN mkdir -p /home/app

# create the app user
RUN addgroup -S app && adduser -S app -G app

# create the appropriate directories
ENV HOME=/home/app
ENV APP_HOME=/home/app/web
RUN mkdir $APP_HOME
WORKDIR $APP_HOME

# install psycopg2 dependencies
RUN apk update \
    && apk add postgresql-dev gcc python3-dev musl-dev libffi-dev openssl-dev

# install other dependencies
RUN apk update && apk add libpq
RUN apk --update add \
    build-base \
    jpeg-dev \
    zlib-dev

COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --no-cache /wheels/*

# copy entrypoint-prod.sh
COPY ./entrypoint.prod.sh $APP_HOME

# copy project
COPY . $APP_HOME

# chown all the files to the app user
RUN chown -R app:app $APP_HOME

# change to the app user
USER app

# run entrypoint.prod.sh
RUN chmod +x /home/app/web/entrypoint.prod.sh
ENTRYPOINT ["/home/app/web/entrypoint.prod.sh"]
```
Что вообще происходит? 
Здесь мы использовали многоэтапную сборку ([multi-stage build](https://docs.docker.com/develop/develop-images/multistage-build/)) Docker, чтобы уменьшить окончательный размер образа. По сути, builder — это временный образ, которое используется для сборки Python. Затем он копируются в конечный производственный образ, а образ builder отбрасывается.

Вы заметили, что мы создали пользователя без полномочий root? По умолчанию Docker запускает контейнерные процессы как root внутри контейнера. Это плохая практика, поскольку злоумышленники могут получить root-доступ к хосту Docker, если им удастся вырваться из контейнера. Если вы root в контейнере, вы будете root на хосте.

Проверим как все работает:
```
docker-compose -f docker-compose.prod.yml down -v
docker-compose -f docker-compose.prod.yml up -d --build
docker-compose -f docker-compose.prod.yml exec web python manage.py migrate --noinput
```
## Nginx
Далее, давайте добавим Nginx, чтобы он действовал как обратный прокси-сервер для Gunicorn для обработки клиентских запросов, а также для обслуживания статических файлов.

Добавим сервис _nginx_ в _docker-compose.prod.yml_ :
```
nginx:
  build: ./nginx
  ports:
    - 1337:80
  depends_on:
    - web
```
Затем в локальном корне проекта создайте или добавьте следующие файлы и папки:
```
nginx
    ├── Dockerfile
    └── nginx.conf
```
Структура должна быть такая:
![enter image description here](https://lh3.googleusercontent.com/pw/AM-JKLVPzlpmePNXoUbrowWvRxv5wCDqcWpuBhMRRzr83LxKZsfuoC-CImdzlnRHmIXaY3u7iuK8b_EOy5vg-ei3jIzQ0l_OYk88GiTeqW2CeYo-11TZ5svqgzepw08QK_CvUbfO50G-NDtVP1iI2oWle0Z6JQ=w674-h270-no?authuser=0)

Файл _Dockerfile_:
```
FROM nginx:1.19.0-alpine
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d
```
Файл _nginx.conf_:
```
upstream hello_django_prom {
    server web:8000;
}

server {

    listen 80;

    location / {
        proxy_pass http://hello_django;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_redirect off;
    }

}
```
Затем обновим сервис `web` в _docker-compose.prod.yml_, заменить `ports` на `expose`:
Общий вид:
```
version: '3.9'
services:
  nginx:
    container_name: prom_web
    build: ./nginx/prod
    restart: on-failure
    image: nginx:1.19.8
    volumes:
      - ./nginx/prod/nginx.conf:/etc/nginx/conf.d/default.conf
      - static_volume:/app/static
    ports:
      - 1337:80
    depends_on:
      - web
  web:
    build:
      context: .
      dockerfile: Dockerfile.prod
    restart: always
    env_file: prom.env
    command: sh -c "gunicorn --bind 0.0.0.0:8000 djangotgprojects.wsgi"
    expose:
      - 8000
    volumes:
      - .:/app
      - static_volume:/app/static
    depends_on:
      - db
  db:
    container_name: dev_db
    image: postgres:12.0-alpine
    env_file: prom.env
    volumes:
      - postgres_data:/var/lib/postgresql/data/

volumes:
  static_volume:
  postgres_data:
```
Теперь порт 8000 открыт только для других сервисов Docker. И это порт больше не будет опубликован на хост-машине.

Проверяем как это работает
```
docker-compose -f docker-compose.prod.yml down -v
docker-compose -f docker-compose.prod.yml up -d --build
docker-compose -f docker-compose.prod.yml exec web python manage.py migrate --noinput
```
Убедимся, что приложение запущено и работает по адресу http://localhost:1337

Структура вашего проекта теперь должна выглядеть так:
![enter image description here](https://lh3.googleusercontent.com/pw/AM-JKLXHSQ_glYaPzZP2aWNP7E1W_sHU5ocB7lKqF_IVg4JP_Ux_71N-VLPOLvLuhKB-rPlKVbmJXVrV0DbssmUn87HWLXHauQ4v0qprNH4OENceyTS9sUQRhGHZyNOyqMuzhk2YEBF5edtXeT4adQfWdnb_Dg=w568-h690-no?authuser=0)

Теперь снова остановим контейнеры:
```
docker-compose -f docker-compose.prod.yml down -v
```
Поскольку Gunicorn является сервером приложений, он не будет обслуживать статические файлы. Итак, настроим обработку статических и мультимедийных файлов

## Статические файлы
Проверим _settings.py_:
```
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "static")
```
### Development

Теперь любой запрос к http://localhost:8000/static/ * будет обслуживаться из каталога «static».
Чтобы проверить, сначала песоберем образы и запустим новые контейнеры в обычном режиме. Убедимся, что статические файлы по-прежнему правильно обслуживаются по адресу http://localhost:8000/admin.

### Production
Для производственной среды добавьте volume в web и службы nginx в **docker-compose.prod.yml**, чтобы каждый контейнер имел общий каталог с именем «static»:
```
version: '3.9'
services:
  nginx:
    container_name: prom_web
    build: ./nginx/prod
    restart: on-failure
    image: nginx:1.19.8
    volumes:
      - ./nginx/prod/nginx.conf:/etc/nginx/conf.d/default.conf
      - static_volume:/app/static
    ports:
      - 1337:80
    depends_on:
      - web
  web:
    build:
      context: .
      dockerfile: Dockerfile.prod
    restart: always
    env_file: prom.env
    command: sh -c "gunicorn --bind 0.0.0.0:8000 djangotgprojects.wsgi"
    expose:
      - 8000
    volumes:
      - .:/app
      - static_volume:/app/static
    depends_on:
      - db
  db:
    container_name: dev_db
    image: postgres:12.0-alpine
    env_file: prom.env
    volumes:
      - postgres_data:/var/lib/postgresql/data/

volumes:
  static_volume:
  postgres_data:
```
Нам также необходимо создать папку «/home/app/web/static» в Dockerfile.prod:
```
# create the appropriate directories
ENV HOME=/home/app
ENV APP_HOME=/home/app/web
RUN mkdir $APP_HOME
RUN mkdir $APP_HOME/staticfiles
WORKDIR $APP_HOME
```
Почему это необходимо?

Docker Compose обычно монтирует именованные тома как root. И поскольку мы используем пользователя без полномочий root, мы получим ошибку отказа в разрешении при запуске команды collectstatic, если каталог еще не существует

Чтобы обойти это, вы можете:

1.  Создайте папку в Dockerfile
2.  Изменить права доступа к каталогу после его монтирования

Мы использовали первое.

Затем обновите конфигурацию Nginx для маршрутизации запросов статических файлов в папку «static»:
```
upstream hello_django {
    server web:8000;
}

server {

    listen 80;

    location / {
        proxy_pass http://hello_django;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_redirect off;
    }

    location /staticfiles/ {
        alias /home/app/web/static/;
    }

}
```
Перезапустим контейнеры
```
docker-compose down -v
```

```
docker-compose -f docker-compose.prod.yml up -d --build
docker-compose -f docker-compose.prod.yml exec web python manage.py migrate --noinput
docker-compose -f docker-compose.prod.yml exec web python manage.py collectstatic --no-input --clear
```
Теперь все запросы к http://localhost:1337/static/ * будут обслуживаться из каталога «static».

Перейдите по адресу http://localhost:1337/admin и убедитесь, что статические ресурсы загружаются правильно.

Вы также можете проверить в логах командой  **docker-compose -f docker-compose.prod.yml logs -f**  что запросы к статическим файлам успешно обрабатываются через Nginx:

Далее снова остановим контейнеры:
```
docker-compose -f docker-compose.prod.yml down -v
```
## Media файлы
Чтобы проверить обработку мультимедийных файлов, начните с создания нового модуля Django:
```
docker-compose up -d --build
docker-compose exec web python manage.py startapp upload
```
Добавим новый модуль в `INSTALLED_APPS` в _settings.py_:
```
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    "upload",
]
```
Внесем изменения в следующие файлы
_code/upload/views.py_:
```
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage


def image_upload(request):
    if request.method == "POST" and request.FILES["image_file"]:
        image_file = request.FILES["image_file"]
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        image_url = fs.url(filename)
        print(image_url)
        return render(request, "upload.html", {
            "image_url": image_url
        })
    return render(request, "upload.html")
```
Добавим директорию «templates», в каталог «code/upload», и добавим новый шаблон _upload.html_:
```
{% block content %}

  <form action="{% url "upload" %}" method="post" enctype="multipart/form-data">
    {% csrf_token %}
    <input type="file" name="image_file">
    <input type="submit" value="submit" />
  </form>

  {% if image_url %}
    <p>File uploaded at: <a href="{{ image_url }}">{{ image_url }}</a></p>
  {% endif %}

{% endblock %}
```
Файл _code/hello_django/urls.py_:
```
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from upload.views import image_upload

urlpatterns = [
    path("", image_upload, name="upload"),
    path("admin/", admin.site.urls),
]

if bool(settings.DEBUG):
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```
Файл _code/hello_django/settings.py_:
```
MEDIA_URL = "/mediafiles/"
MEDIA_ROOT = os.path.join(BASE_DIR, "mediafiles")
```
## Development
Запустим контейнер:
```
docker-compose up --build
```
Скорее всего вы получите ошибку связанную со временем. 
Для этого надо добавить в файлик _requirements.txt_ модуль _tzdata_
И так же у вас появится ошибка, связанная с медиафайлом - Django не сможет найти шаблон. Необходимо поправить в _settings_ на:
```
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```
Создать папку templates в корне и поместить туда файлик upload.html
И проверить. Должна отобразиться такая страница:

![enter image description here](https://lh3.googleusercontent.com/pw/AM-JKLXXnp9kBw1zeAjm4Hk0ff--IRO6mwbJnAmB9FrenTUkk1i_ranDGqp89pBUakqx_HkMYJ3q8VIBGR5WdwZpedCj1UjiLPW8l6L15MsFmZXxaq0grnh5HLE7Ph_j7eWfqoUlo2gRXOxTILEmXMQGshHr4Q=w848-h78-no?authuser=0)

По-сути у нас уже готовое приложение по хостингу наших картинок :)
Теперь у вас должна быть возможность загзулить файл на [http://localhost:8000/](http://localhost:8000/), и затем увидеть этот файл на [http://localhost:8000/mediafiles/IMAGE_FILE_NAME](http://localhost:8000/mediafiles/IMAGE_FILE_NAME).

## Production
Для производственной среды добавим новый том volume в сервисы  `web` и `nginx`:
```
version: '3.9'
services:
  nginx:
    container_name: prom_web
    build: ./nginx/prod
    restart: on-failure
    image: nginx:1.19.8
    volumes:
      - ./nginx/prod/nginx.conf:/etc/nginx/conf.d/default.conf
      - static_volume:/app/static
      - media_volume:/home/app/web/mediafiles
    ports:
      - 1337:80
    depends_on:
      - web
  web:
    build:
      context: .
      dockerfile: Dockerfile.prod
    restart: always
    env_file: prom.env
    command: sh -c "gunicorn --bind 0.0.0.0:8000 djangotgprojects.wsgi"
    expose:
      - 8000
    volumes:
      - .:/app
      - static_volume:/app/static
      - media_volume:/home/app/web/mediafiles
    depends_on:
      - db
  db:
    container_name: dev_db
    image: postgres:12.0-alpine
    env_file: prom.env
    volumes:
      - postgres_data:/var/lib/postgresql/data/

volumes:
  postgres_data:
  static_volume:
  media_volume:
 ```

Создаим каталог /home/code/web/mediafiles в _Dockerfile.prod_:
```
...
# create the appropriate directories
ENV HOME=/home/code
ENV APP_HOME=/home/code/web
RUN mkdir $APP_HOME
RUN mkdir $APP_HOME/staticfiles
RUN mkdir $APP_HOME/mediafiles
WORKDIR $APP_HOME
...
```
Снова обновим конфиг Nginx:
```
upstream hello_django {
    server web:8000;
}

server {

    listen 80;

    location / {
        proxy_pass http://hello_django;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_redirect off;
    }

    location /staticfiles/ {
        alias /home/code/web/staticfiles/;
    }

    location /mediafiles/ {
        alias /home/code/web/mediafiles/;
    }

}
```
Далее перезапустим контейнеры:
```
docker-compose down -v
docker-compose -f docker-compose.prod.yml up -d --build
docker-compose -f docker-compose.prod.yml exec web python manage.py migrate --noinput
docker-compose -f docker-compose.prod.yml exec web python manage.py collectstatic --no-input --clear
```
Проверим как все работает:
1.  Загрузим файл [http://localhost:1337/](http://localhost:1337/).
2.  Затем убедимся что файл доступен на [http://localhost:1337/mediafiles/IMAGE_FILE_NAME](http://localhost:1337/mediafiles/IMAGE_FILE_NAME).

## Заключение

В этой статье мы рассмотрели, как создать контейнер для веб-приложения Django с Postgres. Мы также создали готовый к работе файл Docker Compose, который добавляет Gunicorn и Nginx в нашу конфигурацию для обработки статических и мультимедийных файлов. Теперь вы можете проверить производственную настройку локально.

С точки зрения фактического развертывания в производственной среде, вы, вероятно, захотите использовать:

1.  Полностью управляемый сервис базы данных— такой как [RDS](https://aws.amazon.com/rds/) или [Cloud SQL](https://cloud.google.com/sql/) — вместо того, чтобы управлять своим собственным экземпляром Postgres в контейнере.
2.  Пользователь без полномочий root для `db` и `nginx` сервисов

Спасибо за чтение

Источники используемые для этой статьи

- https://webdevblog.ru/kak-ispolzovat-django-postgresql-i-docker/
- [William Vincent](https://wsvincent.com/)  —  [How to use Django, PostgreSQL, and Docker](https://wsvincent.com/django-docker-postgresql/)
-  [Michael Herman](https://testdriven.io/authors/herman/)  —  [Dockerizing Django with Postgres, Gunicorn, and Nginx](https://testdriven.io/blog/dockerizing-django-with-postgres-gunicorn-and-nginx/)