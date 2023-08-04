---
layout: post
title: "How to install and configure MetalLB"
date: 2023-07-14 01:31:00 +0800
categories: kubernetes
tags: loadbalancer
image:
  path: /assets/img/headers/metallb-logo-4.png
---

*MetalLB is a load-balancer implementation for bare metal Kubernetes clusters, using standard routing protocols*

### Instal MetalLB using kubernetes manifest: 

```bash
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.13.10/config/manifests/metallb-native.yaml
```

### Configure MetatlLB IP range:

*In order to assign an IP to the services, MetalLB must be instructed to do so via the `IPAddressPool` CR.*

```yml
cat <<EOF | kubectl apply -f -
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: first-pool
  namespace: metallb-system
spec:
  addresses:
  - 192.168.229.80-192.168.229.90
EOF
```

### Announce The Service IPs:

*Layer 2 mode is the simplest to configure: in many cases, you don’t need any protocol-specific configuration, only IP addresses.*

```yaml
cat <<EOF | kubectl apply -f -
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: example
  namespace: metallb-system
spec:
  ipAddressPools:
  - first-pool
EOF
```

### Check if all the resources are deployed in `metallb-system` namespace:

```bash
kubectl get all -n metallb-system
```

### Test if LB is assiging the ips:

```bash
kubectl run webapp --image=nginx
kubectl expose pod webapp --port=80 --type=LoadBalancer
```
### Reference Links:

✅ [MetalLb Official Website](https://metallb.org/installation/)

✅ [Configure MetalLB](https://metallb.org/configuration/)
