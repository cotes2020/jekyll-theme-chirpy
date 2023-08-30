---
layout: post
title: "PureLB & OpenELB: LB implementations for bare-metal K8s clusters"
date: 2023-08-30 22:00:00 +0800
categories: kubernetes
tags: loadbalancer
image:
  path: /assets/img/headers/pureopenlb.jpg
---
### PureLB
PureLB is a Service Load Balancer Controller for Kubernetes.

### Instal PureLB using helm charts:

```bash
helm repo add purelb https://gitlab.com/api/v4/projects/20400619/packages/helm/stable

helm repo update

helm fetch purelb/purelb --untar

helm install --create-namespace --namespace=purelb purelb purelb/purelb -f values.yaml
```

### Create IPv4 Service Group:

```yml
cat <<EOF | kubectl apply -f -
apiVersion: purelb.io/v1
kind: ServiceGroup
metadata:
  name: ipv4-routed
  namespace: purelb
spec:
  local:
    aggregation: /32
    pool: 192.168.121.85-192.168.121.95
    subnet: 192.168.121.1/24
EOF
```

### Calico IPv4 Pool:

*Layer 2 mode is the simplest to configure: in many cases, you don’t need any protocol-specific configuration, only IP addresses.*

```yaml
cat <<EOF | kubectl create -f -
apiVersion: crd.projectcalico.org/v1
kind: IPPool
metadata:
  name: purelb-ipv4
spec:
  cidr: 192.168.121.1/24
  disabled: true
EOF
```

### Check if all the resources are deployed in `purelb` namespace:

```bash
kubectl get all -n purelb
```

### PureLB Annotations:
PureLB uses annotations to configure functionality not native in the k8s API.

```sh
apiVersion: v1
items:
- apiVersion: v1
  kind: Service
  metadata:
    annotations:
      purelb.io/service-group: ipv4-routed #service group name
```

### OpenELB
OpenELB is an open-source load balancer implementation designed for bare-metal Kubernetes clusters.

### Instal OpenELB using helm charts:

```sh
helm repo add kubesphere-stable https://charts.kubesphere.io/stable

helm repo update 

helm fetch kubesphere-stable/openelb --untar

helm install openelb kubesphere-stable/openelb -n openelb-system --create-namespace -f values.yaml
```
### Configure IP Address Pools Using Eip:

```sh
cat <<EOF | kubectl create -f -
apiVersion: network.kubesphere.io/v1alpha2
kind: Eip
metadata:
    name: eip-dev-pool
    annotations:
      eip.openelb.kubesphere.io/is-default-eip: "true"
spec:
    address: 192.168.121.85-192.168.121.95
    protocol: layer2
    interface: eth0
    disable: false
status:
    occupied: false
    usage: 1
    poolSize: 10
    used: 
      "192.168.121.91": "default/test-svc"
    firstIP: 192.168.121.85
    lastIP: 192.168.121.95
    ready: true
    v4: true
EOF
```

### Test if OpenELB is assiging the ips:

```bash
kubectl run webapp --image=nginx
kubectl expose pod webapp --port=80 --type=LoadBalancer
```
### Reference Links:

- [PureLB Official Website](https://purelb.gitlab.io/docs/)

- [Gitlab page: PureLB](https://gitlab.com/purelb/purelb/)

- [OpenELB](https://openelb.io)

