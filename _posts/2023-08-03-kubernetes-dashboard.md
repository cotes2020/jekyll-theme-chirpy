---
layout: post
title: "Deploy kubernetes dashboard with user account + Ingress"
date: 2023-08-03 12:26:00 +0800
categories: kubernetes
tags: dashboard
image:
  path: /assets/img/headers/k8s-dashboard.png
---

Kubernetes dashboard is a web-based user interface. You can use Dashboard to deploy containerized applications to a Kubernetes cluster, troubleshoot your containerized application, and manage the cluster resources.

### ⛩️ Deploying the Dashboard UI

```sh
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml
```

### Create a service account

```sh
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: dashboard-admin
  namespace: kubernetes-dashboard
EOF
```

### Create a ClusterRoleBinding

```sh
cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: admin-user
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: dashboard-admin
  namespace: kubernetes-dashboard
EOF
```

### Creating Bearer Token for ServiceAccount

```sh
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: dashboard-token
  namespace: kubernetes-dashboard
  annotations:
    kubernetes.io/service-account.name: "dashboard-admin"   
type: kubernetes.io/service-account-token 
EOF
```

### Create certificate resource to generate certs for kubernetes-dashboard

```sh
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: dashboard-cert
  namespace: kubernetes-dashboard
spec:
  secretName: k8s-dashboard-secret   
  issuerRef:
    name: ca-issuer
    kind: ClusterIssuer
  dnsNames:
    - k8s.dashboard.com
EOF
```

### Create a ingress resource
```sh
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dashboard-ingress
  namespace: kubernetes-dashboard
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: ca-issuer
    nginx.ingress.kubernetes.io/backend-protocol: "HTTPS"
spec:
  ingressClassName: nginx
  rules:
  - host: "k8s.dashboard.com"
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: kubernetes-dashboard
            port:
              number: 443
  tls:
  - hosts:
    - k8s.dashboard.com
    secretName: k8s-dashboard-secret
EOF
```
### Accessing the Dashboard UI

Go to "https://k8s.dashboard.com/" to access the dashboard.

*Execute the following command to get the token which saved in the Secret:*

```sh
kubectl get secret dashboard-token -n kubernetes-dashboard -o jsonpath={".data.token"} | base64 -d
```

Reference Links:

- [Kubernetes Dashboard](https://kubernetes.io/docs/tasks/access-application-cluster/web-ui-dashboard/)
