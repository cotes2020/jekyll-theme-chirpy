---
layout: post
title: "Install Rancher with Lets Encrypt on Kubernetes"
date: 2024-06-29 10:43:00 +0800
categories: rancher
tags: rancher
image:
  path: /assets/img/headers/rancher.png
---

*Rancher is a Kubernetes management tool to deploy and run clusters anywhere and on any provider.*

### Prerequisites:

- Kubernetes cluster
- Helm 3.x
- Domain name and ability to perform DNS changes
- Port 80 & 443 must be accessible for Let's Encrypt to verify and issue certificates

#### Pick a subdomain and create a DNS entry pointing to the IP Address that will be assigned to the Rancher Server.


#### Run the following command to find the IP Address.
```sh
curl ifconfig.me
```

#### Create an A record with the IP Address in your DNS Provider.
```sh
nslookup subdomain_name
```

### Install cert-manager with Helm

#### Add the Helm repository:

```sh
helm repo add jetstack https://charts.jetstack.io --force-update
```

#### Update the helm chart repository:
```sh
helm repo update
```

#### Install cert-manager:

```sh
helm install \
  cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.15.3 \
  --set crds.enabled=true
```

### Install Rancher:

#### Create `cattle-system` namesapce
```sh
kubectl create ns cattle-system
```

#### Add the `Helm repository`

```sh
helm repo add rancher-latest https://releases.rancher.com/server-charts/latest
```

#### `Update` the helm chart repository:
```sh
helm repo update
```

#### Deploy `Rancher`:

```sh
helm install rancher rancher-latest/rancher --namespace cattle-system \
   --set hostname=your_hostname \
   --set bootstrapPassword=Password \
   --set ingress.tls.source=letsEncrypt \
   --set letsEncrypt.email=email@address \
   --set letsEncrypt.ingress.class=nginx
```

```sh
kubectl -n cattle-system rollout status deploy/rancher

kubectl get pods -n cattle-system -w
```

### Access Rancher User Interface
```sh
https://rancher.url
```
### Reference Links:

- [Installing Helm](https://helm.sh/docs/intro/install/)

- [Cert-manager](https://cert-manager.io/docs/)

- [Rancher](https://ranchermanager.docs.rancher.com/)
