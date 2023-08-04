---
layout: post
title: "How to install and configure Cert manager on kubernetes"
date: 2023-07-14 11:25:00 +0800
categories: kubernetes
tags: cert-manager
image:
  path: /assets/img/headers/cert-manager.jpg
---

### Install cert-manager with Helm

Add the Helm repository
```sh
helm repo add jetstack https://charts.jetstack.io
```

Update the helm chart repository
```sh
helm repo update
```

Install cert-manager
```sh
helm install \
  cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.12.0 \
  --set installCRDs=true
```

### Install cert-manager using kubectl apply

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml
```

### Create a Certificate Authority
 
Create a CA private key
```bash
openssl genrsa -out ca.key 2048
```
Create a CA certificate
```bash
openssl req -new -x509 -sha256 -days 365 -key ca.key -out ca.crt
```
Import the CA certificate in the `trusted Root Ca store` of your clients   

### Create cluster issuer object

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: ca-issuer
spec:
  ca:
    secretName: dev-ca-key-pair
```

### Create a secret that will be used for signing in *cert-manager* name space

Convert the content of the key and crt to base64 

```bash
cat ca.crt | base64 -w 0
cat ca.key | base64 -w 0
```

```yml
apiVersion: v1
kind: Secret
metadata:
  name: dev-ca-key-pair
  namespace: cert-manager
data:
  tls.crt: 
    $(base64-encoded cert data from tls-ingress.crt)
  tls.key: 
    $(base64-encoded cert data from tls-ingress.key)
```

### Create certificate resource to generate certs for your applications 

A `Certificate` resource specifies fields that are used to generate certificate signing requests which are then fulfilled by the issuer type you have referenced.

This should be created in the same namespace where your *application* is installed

```yml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: nginx-cert
  namespace: nginx
spec:
  secretName: nginx-tls-secret   
  issuerRef:
    name: ca-issuer
    kind: ClusterIssuer
  dnsNames:
    - nginx.mkbn.tech
```

This will create the certificate object in *nginx* ns , along with secret call *nginx-tls-secret* which can be used in our nginx-ingress config

```yml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nginx-ingress
  namespace: nginx
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: ca-issuer
spec:
  ingressClassName: nginx
  rules:
  - host: "nginx.mkbn.tech"
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nginx
            port:
              number: 80
  tls:
  - hosts:
    - nginx.mkbn-tech
    secretName: nginx-tls-secret
```

### Reference Links:

✅ [Installing Helm](https://helm.sh/docs/intro/install/)

✅ [Cert-manager](https://cert-manager.io/docs/)
