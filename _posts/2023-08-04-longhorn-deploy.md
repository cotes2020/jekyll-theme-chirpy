---
layout: post
title: "Meet Longhorn a cloud native distributed block storage for Kubernetes"
date: 2023-08-04 15:26:00 +0800
categories: kubernetes
tags: longhorn
image:
  path: /assets/img/headers/longhorn.jpg
---

### Prerequisites:

**Minimum Hardware requirements:**

- 3 nodes
- 4 vCPUs per node
- 4 GiB per node
- SSD/NVMe or similar performance block device on the node for storage


### Installation Requirements:

- A container runtime compatible with Kubernetes (Docker v1.13+, containerd v1.3.7+, etc.)
- Kubernetes >= v1.21
- `open-iscsi` is installed, and the `iscsid` daemon is running on all the nodes.
- RWX support requires that each node has a NFSv4 client installed.
- The host filesystem supports the `file extents` feature to store the data. Currently longhorn support:
    - ext4
    - XFS
- `bash, curl, findmnt, grep, awk, blkid, lsblk` must be installed.
- `Mount propagation` must be enabled.

### Install dependencies:

Install `nfs-common, open-iscsi` & ensure `daemon` is running on all the nodes.

```sh
{
sudo apt update
sudo apt install -y nfs-common open-iscsi
sudo systemctl enable open-iscsi --now
systemctl status iscsid
}
```

**Run the Environment Check Script:**

*Note: `jq`[sudo apt install -y jq] maybe required to be installed locally prior to running env check script.*
```sh
curl -sSfL https://raw.githubusercontent.com/longhorn/longhorn/v1.5.1/scripts/environment_check.sh | bash
```

### Installing Longhorn with Helm:
*Helm v3.0+ must be installed on your workstation.*

Add the Longhorn Helm repository:
```sh
helm repo add longhorn https://charts.longhorn.io
```

Fetch the latest charts from the repository:
```sh
helm repo update
```

Retrieve the package from longhorn repository, and download it locally:

```sh
helm fetch longhorn/longhorn --untar
```

Install Longhorn in the longhorn namespace:

```sh
helm install longhorn longhorn/longhorn --values /tmp/longhorn/values.yaml -n longhorn --create-namespace --version 1.5.1
```

To confirm that the deployment succeeded, run:
```sh
kubectl -n longhorn get pod
```

### Accessing the Longhorn UI:

Get the Longhorn’s external service IP:
```sh
kubectl -n longhorn get svc
```
Use `CLUSTER-IP` of the `longhorn-frontend` to access the Longhorn UI using port forward:

```sh
kubectl port-forward svc/longhorn-frontend 8080:80 -n longhorn
```

### Enabling basic authentication with ingress for longhorn UI
*Authentication is not enabled by default for kubectl and Helm installations.*

Create a basic authentication file `auth`. It’s important the file generated is named auth (actually - that the secret has a key data.auth), otherwise the Ingress returns a 503.

```sh
USER=<USERNAME_HERE>; PASSWORD=<PASSWORD_HERE>; echo "${USER}:$(openssl passwd -stdin -apr1 <<< ${PASSWORD})" >> auth
```
Create a secret:

```sh
kubectl -n longhorn create secret generic basic-auth --from-file=auth
```

Create the ingress resource:

```sh
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: longhorn-ingress
  namespace: longhorn
  annotations:
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/ssl-redirect: 'false'
    nginx.ingress.kubernetes.io/auth-secret: basic-auth
    nginx.ingress.kubernetes.io/auth-realm: 'Authentication Required '
    nginx.ingress.kubernetes.io/proxy-body-size: 10000m
spec:
  rules:
  - http:
      paths:
      - pathType: Prefix
        path: "/"
        backend:
          service:
            name: longhorn-frontend
            port:
              number: 80
EOF
```


Reference Links:

- [longhorn](https://longhorn.io/)

- [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)