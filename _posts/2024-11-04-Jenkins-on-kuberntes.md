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
          systemMessage: 🚀 Welcome To Jenkins Prod instance 🚀

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

kubectl patch svc jenkins --namespace jenkins -p '{"spec": {"type": "NodePort"}}'

export NODE_PORT=$(kubectl get --namespace jenkins -o jsonpath="{.spec.ports[0].nodePort}" services jenkins)

export NODE_IP=$(kubectl get nodes --namespace jenkins -o jsonpath="{.items[0].status.addresses[0].address}")

echo http://$NODE_IP:$NODE_PORT
```


#### jenkinsUrl must be set correctly else stylesheet won't load and the dark theme. 

```sh
controller:
  jenkinsUrl: http://$NODE_IP:$NODE_PORT
  jenkinsAdminEmail: mkbn@mkbn.com
  # don't set these via JCasC, set as helm chart values
```

#### Run helm upgrade

```sh
helm upgrade jenkins jenkins/jenkins --values /tmp/jenkins/values.yaml -n jenkins
```

`Login with the username: admin and password: SecureP@ssw0rd!`

#### Expose jenkins ui using ingress

#### Install nginx ingress controller
```sh
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

helm install ingress-nginx ingress-nginx/ingress-nginx \
--namespace ingress-nginx --create-namespace
```

#### Check the status of the Ingress Controller:
```sh
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx
```
`Note: Add the EXTERNAL-IP address of your ingress controller to your domain’s DNS records as an A record.`

#### Installing Cert-Manager
```sh
helm repo add jetstack https://charts.jetstack.io --force-update

helm install cert-manager jetstack/cert-manager \
--namespace cert-manager --create-namespace \
--version v1.16.1 \
--set installCRDs=true
```

#### Verify Cert-Manager is running
```sh
kubectl get pods -n cert-manager
```

#### Create the ClusterIssuer resource
```sh
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: mkbn@mkbn.in
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

#### Create the ingress resource
```sh
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: jenkins-ingress
  namespace: jenkins
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - jenkins.mkbn.in
    secretName: jenkins-tls
  rules:
  - host: jenkins.mkbn.in
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: jenkins
            port:
              number: 8080
EOF
```

`Now access the jenkins ui with https://jenkins.mkbn.in`

Reference Links:

- [jenkins](https://artifacthub.io/packages/helm/jenkinsjenkins/jenkins)

- [nginx ingress controller](https://kubernetes.github.io/ingress-nginx/deploy/)

- [Cert-Manager](https://cert-manager.io/docs/installation/helm/)

