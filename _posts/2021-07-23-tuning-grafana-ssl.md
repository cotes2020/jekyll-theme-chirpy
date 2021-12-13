---
title: Тюнинг Grafana. Ставим SSL
author: Кирьянов Артем
date: 2021-07-23 11:00:00 +0300
categories: [IT, DevOps]
tags: [monitoring,grafana,prometheus]
---

## Докручиваем Grafana.

![graphana](https://devopsme.ru/assets/img/nginx-proxy.png)

В предыдущей статье, я написал о том, как установить Grafana и Prometheus в Docker. Но работает это все без SSL и у нас нет установленных Dashboard'ов. Пришло время это исправить.

### Правим конфиг

Зайдем в нашем конфиг prometheus.yml и подправим наш файл: 
```yml
# my global config
global:
  scrape_interval:     15s # Set the scrape interval to every 15 seconds. Default is every 1 minute.
  evaluation_interval: 15s # Evaluate rules every 15 seconds. The default is every 1 minute.
  external_labels:
    monitor: 'Monitoring'

alerting:
  alertmanagers:
     - static_configs:
        - targets:
           # - alertmanager:9093


rule_files:
  # - "first.rules"
  # - "second.rules"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['ВАШ_ВНЕШНИЙ_IP:9090']

  - job_name: 'docker'
    static_configs:
      - targets: ['ВАШ_ВНЕШНИЙ_IP:9323']

  - job_name: 'node'
    static_configs:
      - targets: ['ВАШ_ВНЕШНИЙ_IP:9100']

  - job_name: 'cadvisor'
    scrape_interval: 5s
    static_configs:
      - targets: ['ВАШ_ВНЕШНИЙ_IP:8080']
```
В этом файле мы добавили [cadvisor](https://github.com/google/cadvisor). На Github вы можете почитать о нем подробнее. 

### Настраиваем SSL

Для того чтобы можно было управлять SSL для Docker-контейнеров, мы можем воспользоваться готовым инструментом [nginxproxymanager](https://nginxproxymanager.com/). Создадим папку и в ней создаем docker-compose.yml и впишем туда конфиг из Github.

```yml
version: '3'
services:
  app:
    image: 'jc21/nginx-proxy-manager:latest'
    restart: unless-stopped
    ports:
      - '80:80'
      - '81:81'
      - '443:443'
    environment:
      DB_MYSQL_HOST: "db"
      DB_MYSQL_PORT: 3306
      DB_MYSQL_USER: "npm"
      DB_MYSQL_PASSWORD: "npm"
      DB_MYSQL_NAME: "npm"
    volumes:
      - ./data:/data
      - ./letsencrypt:/etc/letsencrypt
  db:
    image: 'jc21/mariadb-aria:latest'
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: 'npm'
      MYSQL_DATABASE: 'npm'
      MYSQL_USER: 'npm'
      MYSQL_PASSWORD: 'npm'
    volumes:
      - ./data/mysql:/var/lib/mysql
```
И запускаем

```sh
docker-compose up -d
```

После этого останавливаем контейнер с Grafana, удаляем его и в терминале пишем уже такую команду

```sh
docker run -d --name grafana --network nginxproxymanager_default -v grafana_config:/etc/grafana -v grafana_data:/var/lib/grafana -v grafana_logs:/var/log/grafana grafana/grafana
```
Тем самым мы укажем нашему контейнеру работать в сети нашего proxy. Т.е сделаем так, чтоб контейнер proxy видел контейнер с grafana. 

Теперь заходим на сайт панели proxy_manager. Впишем в адресную строчку адрес: ВАШ_ВНЕШНИЙ_IP:81 и увидим панель для входа. 
![nginx_admin](https://devopsme.ru/assets/img/nginx_admin.png)

Вводим логин и пароль из документации, попадаем в саму панель и меняем пароль и почту на свои. Панель управления, выглядит удобно и совсем просто разобраться в настройке. 
![nginx_admin_1](https://devopsme.ru/assets/img/nginx_admin_1.png)

Заходим в Proxy Hosts и добавляем наш домен. 
![nginx_admin_2](https://devopsme.ru/assets/img/nginx_admin_2.png)

Теперь выпустим SSL
![nginx_admin_3](https://devopsme.ru/assets/img/nginx_admin_3.png)

На этом настройка закончена и у нас добавлен SSL для нашего поддомена. 
> Внимание! Сначала вы должны добавить в панель управления домена, ваш поддомен и создать A-записи. 
После настройки отобразиться статус работы и вы можете проверить работу proxy. 

### Настройка Dashboard в Grafana. 

На самом деле тут ничего сложного нет, у Grafana есть много готовых Dashboard, которые мы можете импортировать, но для начала необходимо подключить адрес, с которого мы будем считывать метрики. Заходим в конфигурацию Data Source. 
![grafana_admin_1](https://devopsme.ru/assets/img/grafana_admin_1.png)

Добавляем источник. 
![grafana_admin_2](https://devopsme.ru/assets/img/grafana_admin_2.png)
> В URL пишем ваш внешний IP и порт 9090
Нажимаем Save & Test. Если все настроено верно, появится надпись Data Source is working. 

Теперь можем настраивать панели для отображения метрик. Я использую 3 панели. 
[Docker monitoring with node selection](https://grafana.com/grafana/dashboards/8321)
[Linux Hosts Metrics](https://grafana.com/grafana/dashboards/10180)
[Node Exporter Full](https://grafana.com/grafana/dashboards/1860)

Для добавления, необходимо имортировать ID вот сюда
![grafana_admin_3](https://devopsme.ru/assets/img/grafana_admin_3.png)

На этом настройка закончена. Дальше можете прикручивать уведомления в телеграм, можете считывать метрики с БД и т.п.