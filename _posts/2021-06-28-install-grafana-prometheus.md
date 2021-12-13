---
title: Настройка Prometheus + Grafana в Docker
author: Кирьянов Артем
date: 2021-06-28 11:00:00 +0300
categories: [it, devops,monitoring]
tags: [monitoring,grafana,prometheus]
---

## Grafana и Prometheus в Docker.

![graphana](https://devopsme.ru/assets/img/pg.png)

В этой статье я бы хотел рассказать о том, как установить эти инструменты. Я буду писать про установку на сервер. Требования: установленный Docker ну и настроенное окружение, как доступ по ssh и другое по вашему "вкусу".

**Что такое Grafana и Prometheus?**

Prometheus - сердце мониторинга. Это экосистема, позволяющая собирать метрики с физических серверов, VPS, отдельных приложений, различных баз данных, контейнеров и устройств IoT. Prometheus включает в себя сотни готовых решений для сбора информации и позволяет одновременно мониторить тысячи служб. При необходимости, по мере изучения, опытный администратор может для него написать свои сценарии.

Node exporter - это сервер, который показывает метрики хост-машины, на которой он работает (включая метрики файловой системы, сетевых устройств, использование процессора, памяти и многое другое). В нашем случае, мы запускаем node-exporter в docker на порта 9323. 

Grafana - среда визуализации. Использует метрики, которые собрал Prometheus, и отображает их в виде информативных графиков и диаграмм, организованных в настраиваемые панели.

### Установка Prometheus

Заходим на сервер по ssh и создаем дирректорию, с любым названием. У меня это папочка devops, а в ней уже конфиги для нужных мне программ. Необходимо создать конфиг, создав файл в нашей папке - prometheus.yml. Для чего вообще этот файл нужен? В этом файлике описываются сервера для сбора метрик и отправки алертов.  

Теперь Prometheus будет собирать метрики с сервера, на котором он установлен, а также с сервера Node Exporter, который будет настроен **позже**. Система Prometheus может подключаться внутри своего контейнера при помощи имени localhost, однако ей нужно также собирать метрики с Node Exporter, а для этого необходим внешний IP-адрес (поскольку Node Exporter запускается в отдельном контейнере с отдельным сетевым пространством имён). Но есть нюансы, о которых я напишу ниже. 

#### prometheus.yml
```yml 
global:
  scrape_interval:     15s 
  evaluation_interval: 15s 

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']
```
> Примечание: Замените your_server_ip IP-адресом сервера, где job_name: 'docker'.
> Если вы хотите использовать только метрики node_exporter, достаточно указать один узел: ['localhost:9100']

Сохраняем файлик. Теперь о нюансах. Чтобы настроить демона Docker в качестве целевого объекта Prometheus, нам необходимо указать адрес метрики. Лучший способ сделать это-через файл daemon.json, который по умолчанию находится в одном из следующих местоположений. Если файл не существует, создайте его.. Для этого нам необходимо создать в /etc/docker файлик daemon.json. 

```yml
{
  "metrics-addr" : "0.0.0.0:9323",
  "experimental" : true
}
```
Останавливаем docker как службу
```yml
sudo systemctl stop docker
```
Теперь нам необходимо добавить порт 9323 в зону public для firewall
```yml
sudo firewall-cmd --permanent --zone=public --add-port=9323/tcp
```
Перезапускаем сервис
```yml
sudo firewall-cmd --reload
```
> В Ubuntu  firewall-cmd не установлен
> Устанавливаем через apt-get install

После этого запускам docker
```yml
sudo docker start docker
```
Теперь можно запустить Prometheus, указав ему наш файлик. 
Запускаем такую команду: 
```yml
docker run -d -p 9090:9090 -v /home/devops/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```
Переходим на сайт http://localhost:9090/targets
docker и prometheus должны находится в состоянии UP.

### Установка Grafana

Пишем такую команду: 
```yml
docker run -d --name grafana -p 3000:3000 -v grafana_config:/etc/grafana -v grafana_data:/var/lib/grafana -v grafana_logs:/var/log/grafana grafana/grafana
```
Добавляем порт в firewall
```yml
sudo firewall-cmd --permanent --zone=public --add-port=3000/tcp
```
Перезапускаем сервис
```yml
sudo firewall-cmd --reload
```
После этого можно перейти по адресу http://ВАШ_IP:3000 и сменить пароль для Grafana

### Устанавливаем node-exporter 

В прошлых шагах мы установили метрики для docker. А как же собирать тогда метрики с самой хост машины на Linux? 
А это можно решить таким лайфхаком. Node_exporter предназначен для мониторинга хост-системы. Не рекомендуется развертывать его в качестве контейнера Docker, поскольку для этого требуется доступ к хост-системе.
В ситуациях, когда требуется развертывание Docker, необходимо использовать некоторые дополнительные флаги, чтобы разрешить node_exporter доступ к пространствам имен хоста.
Имейте в виду, что любые некорневые точки монтирования, которые вы хотите отслеживать, необходимо будет смонтировать в контейнере.
Если вы запускаете контейнер для мониторинга хоста, укажите аргумент path.rootfs. Этот аргумент должен соответствовать пути в привязке-монтировании корневого узла. node_exporter будет использовать path.rootfs в качестве префикса для доступа к файловой системе хоста.
```yml
docker run -d \
  --net="host" \
  --pid="host" \
  -v "/:/host:ro,rslave" \
  quay.io/prometheus/node-exporter:latest \
  --path.rootfs=/host
```
Затем необходимо добавить правило для фаервола: 
```yml
sudo firewall-cmd --add-port=9100/tcp --permanent
sudo systemctl restart firewalld
```
Затем добавим в наш prometheus.yml нашу хост-машину, итоговый файлик: 
```yml
# my global config
global:
  scrape_interval:     15s # Set the scrape interval to every 15 seconds. Default is every 1 minute.
  evaluation_interval: 15s # Evaluate rules every 15 seconds. The default is every 1 minute.

  
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
      - targets: ['localhost:9090']

  - job_name: 'docker'
    static_configs:
      - targets: ['БЕЛЫЙ_IP:9323'] #Добавляем ваш белый IP или localhost

  - job_name: 'node'
    static_configs:
      - targets: ['БЕЛЫЙ_IP:9100'] #Добавляем ваш белый IP или localhost
```
Заходим на сайт и проверяем все ли таргеты у нас поднялись. 

Ура! Мы установили Prometheus и Grafana в Docker. В следующих статьях, мы прикрутим оповещение в телеграм, выведем метрики для нашего aspnet приложения. 