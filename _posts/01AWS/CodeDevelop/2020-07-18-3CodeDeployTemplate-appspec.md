---
title: AWS - CodeDevelop - CodeDeploy - appspec.yml Template
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---


# appspec.yml Template


```yml
# example from AWS WhitePaper, no real credential inside

version:[1]version-number                      # version: 0.0
os:[1]operating-system-name                    # os: linux OR windows
files:                                         # files:
[2]-[1]source:[1]source-files-location         #  - source: /
[4]destination:[1]destination-files-location   #    destination: /var/www/html/WordPress
permissions:                                   # permissions:
[2]-[1]object:[1]object-specification
[4]pattern:[1]pattern-specification
[4]except:[1]exception-specification
[4]owner:[1]owner-account-name
[4]group:[1]group-name
[4]mode:[1]mode-specification
[4]acls:                                       # [4]acls:
[6]-[1]acls-specification
[4]context:
[6]user:[1]user-specification
[6]type:[1]type-specification
[6]range:[1]range-specification
[4]type:
[6]-[1]object-type
hooks:                                         # hooks:
[2]deployment-lifecycle-event-name:            #   BeforeInstall:
[4]-[1]location:[1]script-location             #     - location: scripts/install_dependencies.sh
[6]timeout:[1]timeout-in-seconds               #       timeout: 300
[6]runas:[1]user-name                          #       runas: root
                                               #   AfterInstall:
                                               #     - location: scripts/change_permissions.sh
                                               #       timeout: 300
                                               #       runas: root
                                               #   ApplicationStart:
                                               #     - location: scripts/start_server.sh
                                               #     - location: scripts/create_test_db.sh
                                               #       timeout: 300
                                               #       runas: root
                                               #   ApplicationStop:
                                               #     - location: scripts/stop_server.sh
                                               #       timeout: 300
                                               #       runas: root


# example of a correctly spaced AppSpec file:
version: 0.0
os: linux
files:
  - source: /
    destination: /var/www/html/WordPress
hooks:
  BeforeInstall:
    - location: scripts/install_dependencies.sh
      timeout: 300
      runas: root
  AfterInstall:
    - location: scripts/change_permissions.sh
      timeout: 300
      runas: root
  ApplicationStart:
    - location: scripts/start_server.sh
    - location: scripts/create_test_db.sh
      timeout: 300
      runas: root
  ApplicationStop:
    - location: scripts/stop_server.sh
      timeout: 300
      runas: root


# example of a correctly spaced AppSpec file:
version: 0.0
os: linux
files:
  - source: Config/config.txt
    destination: /webapps/Config
  - source: Source
    destination: /webapps/Config
hooks:
  BeforeInstall:
    - location: scripts/install_dependencies.sh
    - location: scripts/UnzipResourceBundle.sh
      location: scripts/UnzipDataBundle.sh
      timeout: 300
      runas: root
  AfterInstall:
    - location: scripts/change_permissions.sh
    - location: scripts/RunResourceTests.sh
      timeout: 300
      runas: root
  ApplicationStart:
    - location: scripts/start_server.sh
    - location: scripts/create_test_db.sh
    - location: scripts/RunFunctionTests.sh
      timeout: 300
      runas: root
  ValidataService:
    - location: scripts/MonitorService.sh
      timeout: 3600
      runas: CodeDeployuser
ApplicationStop:
    - location: scripts/stop_server.sh
      timeout: 300
      runas: root
```
