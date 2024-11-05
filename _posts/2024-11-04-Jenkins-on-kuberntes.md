---
layout: post
title: "Running Jenkins on Kubernetes"
date: 2024-11-04 13:55:00 +0800
categories: jenkins
tags: jenkins
image:
  path: /assets/img/headers/jenkins.png
---

*Jenkins is a self-contained, open source automation server which can be used to automate all sorts of tasks related to building, testing, and delivering or deploying software.*

### Prerequisites:

- Kubernetes cluster
- Helm 3.x

### Installing Jenkins with Helm:
*Helm v3.0+ must be installed on your workstation.*

#### Add the jenkins Helm repository:
```sh
helm repo add jenkins https://charts.jenkins.io
```

#### Fetch the latest charts from the repository:
```sh
helm repo update
```

#### Retrieve the package from jenkins repository, and download it locally:

```sh
helm fetch jenkins/jenkins --untar
```

#### Lets update `values.yaml` file as follow, 

```sh
# Change admin password
controller:
  admin:
    username: "admin"
    password: 'SecureP@ssw0rd!'

# Install Additional Plugins
controller:
  additionalPlugins: ['pipeline-graph-view', 'job-dsl', 'junit', 'coverage', 'dark-theme']

# Configure Dark Theme
controller:
  JCasC:
    configScripts:
      dark-theme: |
        appearance:
          themeManager:
            disableUserThemes: true
            theme: "dark"

# Configure welcome message
controller:
  JCasC:
    configScripts:
      welcome-message: |
        jenkins:
          systemMessage: 🚀🚀🚀 Jenkins Prod instance 🚀🚀🚀

```

#### Install jenkins in the jenkins namespace:

```sh
helm install jenkins jenkins/jenkins --values /tmp/jenkins/values.yaml -n jenkins --create-namespace
```

#### To confirm that the deployment succeeded, run:
```sh
kubectl -n jenkins get pods
```

#### Get the `Jenkins URL` to visit by running these commands in the same shell:
```sh
export NODE_PORT=$(kubectl get --namespace jenkins -o jsonpath="{.spec.ports[0].nodePort}" services jenkins)

export NODE_IP=$(kubectl get nodes --namespace jenkins -o jsonpath="{.items[0].status.addresses[0].address}")

echo http://$NODE_IP:$NODE_PORT
```
`Login with the username: admin and password: SecureP@ssw0rd!`


Reference Links:

- [jenkins](https://artifacthub.io/packages/helm/jenkinsjenkins/jenkins)
