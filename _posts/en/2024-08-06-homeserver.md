---
title: 'Home Server : My self-hosted services'
description: Discover how I host multiples services on my own server
date: {}
categories: []
media_subpath: /assets/img/posts/homeserver
tags: []
lang: en
---

In a world where we are increasingly reliant on the cloud and large corporations to store our data, I became very interested in self-hosting. For the past three years, I've hosted several services on my server for my relatives and myself.

## The beginning of my journey: Raspberry Pi and PI-Hole

My self-hosted addiction began with a Raspberry PI and the PI-Hole application, which filters all advertisements and tracking on my network at the DNS level. I used it that way for nearly a year before realizing I needed more.

![PI-Hole](pihole.png){: w="150" h="150"}
_PI Hole_

The first service I intended to create was a media server. The most popular at the time was Plex. 
So I attached a 1TB hard drive to store the media and attempted to install and configure Plex Server. It was simple to set up, however after a while, I ran across problems.

First, instabilities occurred frequently when Plex or other software was updated. 
Second, the Raspberry PI model 3B+, which had 1GB RAM and a quad-core processor clocked at 1.4GHz, struggled to decode high-resolution video. 

To address these concerns, I took two steps: I installed Docker and purchased new hardware.

## Easy setup with Docker

Docker is a platform for running applications within a container. Containers are segregated from one another and include their own software, libraries, and configuration files. Furthermore, because all containers use the same operating system kernel, they consume fewer resources than virtual computers. 

![Docker](docker.png){: w="300" h="150"}
_Docker_

For these reasons, Docker is an excellent alternative, making it simple to manage different services operating on the same server without worrying about dependencies, incompatibilities, and so on.

I also define all of my container configurations in YAML files using Docker Compose. This allows all server configurations to be easily backed up and updated.

## A more powerful hardware

With more and more services and my RPI's inability to decode some huge media, I decided to purchase more capable hardware. I found a Lenovo Thinkstation on the used market (which is frequently swamped with outdated enterprise hardware). It is significantly less expensive than a traditional server and is adequate for my needs.

![Lenovo Thinkstation](lenovo.png){: w="200" h="150"}
_Lenovo Thinkstation_

For the OS, I installed the most recent stable Debian release (headless version) to provide a solid base while remaining compatible with the majority of software.

It was placed on a new 512GB SSD to enhance performance. The 1 TO hard disk store only handles data (media, static storage, etc.).

## My server services

Now that I had a functioning server, I began to add more and more services based on my needs. I created the following infographic using the services I am now utilizing :

![Homeserver Architecture](beniserv.png){: w="600" h="650"}
_My Home Server Services_

### Access and security

The most important service is likely Caddy. It is a web server that includes automated HTTPS and reverse proxy functionality. 

In other words, due to Caddy, I can access my services directly via a subdomain, and all traffic is encrypted and sent to the 443 port. For example, I can access one of my services directly at "myservice.beniserv.fr" and another at "myotherservice.beniserv.fr", and so on.

With this configuration, I only need to expose one port (443) to the internet.

I also implemented a dynamic DNS plugin to Caddy to synchronize my IP server address with my domain name provider. So, even if my internet service provider assigns me a dynamic IP address, I can always reach my home server using my domain name.

To improve security, I also use Crowdsec, a service that detects malicious peers and prevents them from accessing my server. Crowdsec analyzes the Caddy logs, and if its behavior engine detects one of the configured scenarios, it blocks the IP address at the firewall level. Crowdsec's advantage over other similar services (such as Fail2ban) is that it provides a collaborative solution. Indeed, when a Crowdsec user blocks an aggressive IP, the information is shared with all users, thereby improving everyone's security.

### Update and backup

I also have Watchtower to always keep all my services updated and benefits from the latest security fix and features. Every night, it downloads the latest versions and replaces them if an update is available.

And because sometimes an update or a storage drive can fail, I use a service to do a periodic backup. Every night, it will save all the server configurations and some data in an encrypted archive file. This archive is then saved on my 1 TO hard drive and on an AWS cloud storage server. A retention policy of 7 days makes sure to not bloat the backup storage by removing the oldest backup archives.

> Find the docker-volume-backup service here : [https://github.com/offen/docker-volume-backup](https://github.com/offen/docker-volume-backup).
{: .prompt-tip }


### Monitoring

With multiple services running, I need to have a way to monitor them and receive notification when an issue is detected. That's why I use Portainer and Homepage.

Portainer allows you to directly manage Docker containers via a graphical interface. Even though I prefer to always use Docker Compose to deploy new service, I still use Portainer to monitor the health status of my already existing services. When needed, I can also easily restart them from the Portainer UI.

Homepage, is a highly customizable dashboard. It proposes widgets for a wide choice of services, and can display different system metrics (uptime, disk space, server temperature, ...). I use it as a quick way to access my different services.

![Homepage example](homepage.png){: w="600" h="350"}
_Homepage example (not mine)_

> Find the homepage service here : [https://github.com/gethomepage/homepage](https://github.com/gethomepage/homepage).
{: .prompt-tip }

To be alerted in case of errors, my services are also sending messages on a private Discord server.

### Other services

Like you can see on the previous image, the following services are also running :
- **Jellyfin** : Media server I use instead of Plex.
- **Paperless-ngx** : Document management and storage system
- **Nextcloud** : Cloud storage service
- **QBitorrent** : Torrent client
- And more...
