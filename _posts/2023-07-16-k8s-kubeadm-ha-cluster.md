---
layout: post
title: "Let's build a HA kubernetes cluster on baremetal servers"
date: 2023-07-17 22:15:00 +0800
categories: kubernetes
tags: kubernetes
image:
  path: /assets/img/headers/helm.jpg
---
# HA Kubernetes cluster with containerd

### Prerequisites:
- 2 Ubuntu20.04 LoadBalancer node's
- 3 Ubuntu20.04 Kubernetes master node's
- 2 Ubuntu20.04 Kubernetes worker node's

### Generate TLS certificates:

*SSH into one of the master-node and run:*

```sh
{
mkdir certs
cd certs
wget https://pkg.cfssl.org/R1.2/cfssl_linux-amd64
wget https://pkg.cfssl.org/R1.2/cfssljson_linux-amd64
chmod +x cfssl*
sudo mv cfssl_linux-amd64 /usr/local/bin/cfssl
sudo mv cfssljson_linux-amd64 /usr/local/bin/cfssljson
}
```

### Create a Certificate Authority (CA):

```sh
{

cat > ca-config.json <<EOF
{
    "signing": {
        "default": {
            "expiry": "8766h"
        },
        "profiles": {
            "kubernetes": {
                "expiry": "8766h",
                "usages": ["signing","key encipherment","server auth","client auth"]
            }
        }
    }
}
EOF

cat > ca-csr.json <<EOF
{
  "CN": "Kubernetes",
  "key": {
    "algo": "rsa",
    "size": 2048
  },
  "names": [
    {
      "C": "IE",
      "L": "Cork",
      "O": "Kubernetes",
      "OU": "CA",
      "ST": "Cork Co."
    }
  ]
}
EOF

cat > kubernetes-csr.json <<EOF
{
  "CN": "Kubernetes",
  "key": {
    "algo": "rsa",
    "size": 2048
  },
  "names": [
    {
      "C": "IE",
      "L": "Cork",
      "O": "Kubernetes",
      "OU": "CA",
      "ST": "Cork Co."
    }
  ]
}
EOF

cfssl gencert -initca ca-csr.json | cfssljson -bare ca
}
```
### Create TLS certificates:
*Set `<load-balancer-ips>`,`<master-node-ips>` as per your configuration:*

```sh
{
cfssl gencert \
-ca=ca.pem \
-ca-key=ca-key.pem \
-config=ca-config.json \
-hostname=<load-balancer-vip>,<master01-ip>,<master02-ip>,<master03-ip>,<load01-balancer-ip>,<load02-balancer-ip>,127.0.0.1,kubernetes.default \
-profile=kubernetes kubernetes-csr.json | \
cfssljson -bare kubernetes
}
```

### Setup kubernetes master & worker nodes:

**Run on all master & worker nodes:**

*Load Kerenel Modules*
```sh
{
cat << EOF | sudo tee /etc/modules-load.d/containerd.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter
}
```
*Add Kernel Settings*
```sh
{
cat <<EOF | sudo tee /etc/sysctl.d/99-kubernetes-cri.conf
net.bridge.bridge-nf-call-iptables = 1
net.ipv4.ip_forward = 1
net.bridge.bridge-nf-call-ip6tables = 1
EOF

sysctl --system
}
```

*Install runtime: containerd*

```sh
{
wget https://github.com/containerd/containerd/releases/download/v1.7.2/containerd-1.7.2-linux-amd64.tar.gz
tar Cxzvf /usr/local containerd-1.7.2-linux-amd64.tar.gz
rm containerd-1.7.2-linux-amd64.tar.gz
mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml
}
```
```sh
{
mkdir -pv /usr/local/lib/systemd/system/
wget https://raw.githubusercontent.com/containerd/containerd/main/containerd.service -O /usr/local/lib/systemd/system/containerd.service
}
```
```sh
{
systemctl daemon-reload
systemctl enable --now containerd
systemctl start containerd
systemctl status containerd
}
```

```sh
#Set the data path for worker nodes only if your using a different data disk
vim /etc/containerd/config.toml
  root = "/data/containerd"
:wq!
```
*Disable swap*
```sh
swapoff -a; sed -i '/swap/d' /etc/fstab
```
### Install runc
```sh
{
wget https://github.com/opencontainers/runc/releases/download/v1.1.8/runc.amd64
install -m 755 runc.amd64 /usr/local/sbin/runc
}
```

### Installing CNI plugins

```sh
{
wget https://github.com/containernetworking/plugins/releases/download/v1.3.0/cni-plugins-linux-amd64-v1.3.0.tgz
mkdir -p /opt/cni/bin
tar Cxzvf /opt/cni/bin cni-plugins-linux-amd64-v1.3.0.tgz
}
```
### Install cricctl 
```sh
{
VERSION="v1.27.1" #change this based on your k8s version
curl -L https://github.com/kubernetes-sigs/cri-tools/releases/download/$VERSION/crictl-${VERSION}-linux-amd64.tar.gz --output crictl-${VERSION}-linux-amd64.tar.gz

tar zxvf crictl-$VERSION-linux-amd64.tar.gz -C /usr/bin
rm -f crictl-$VERSION-linux-amd64.tar.gz
}
```

### Set containerd as default runtime for crictl
```sh
{
cat << EOF | sudo tee /etc/crictl.yaml
runtime-endpoint: unix:///run/containerd/containerd.sock
image-endpoint: unix:///run/containerd/containerd.sock
timeout: 2
debug: false
pull-image-on-create: false
EOF
}
```

### Add Apt repository & Install Kubernetes components:
```sh
{
apt-get update && sudo apt-get install -y apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
cat << EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
}
```

```sh
{
apt-get update
apt-get install -y kubelet=1.27.4-00 kubeadm=1.27.4-00 kubectl=1.27.4-00
}

```

### ETCD cluster creation`[only on master nodes]`:

*Note: Copy `ca.pem, kubernetes.pem, kubernetes-key.pem` to other master nodes.*

```sh
{
declare -a NODES=(192.168.56.101 192.168.56.102 192.168.56.103)

for node in ${NODES[@]}; do
  scp ca.pem kubernetes.pem kubernetes-key.pem root@$node: 
done
}
```

```sh
{
  mkdir -pv /etc/etcd /data/etcd
  mv ca.pem kubernetes.pem kubernetes-key.pem /etc/etcd/
  ll /etc/etcd/
}
```

### Download etcd & etcdctl binaries from Github:

```sh
{
  ETCD_VER=v3.5.9
  wget -q --show-progress "https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz"
  tar zxf etcd-${ETCD_VER}-linux-amd64.tar.gz
  mv etcd-${ETCD_VER}-linux-amd64/etcd* /usr/local/bin/
  rm -rf etcd*
}
```
### Create systemd unit file for etcd service:

*Set NODE_IP to the correct IP of the machine where you are running this*
```sh
{

NODE_IP="k8s-master-1"

ETCD1_IP="k8s-master-1"
ETCD2_IP="k8s-master-2"
ETCD3_IP="k8s-master-3"


cat <<EOF >/etc/systemd/system/etcd.service
[Unit]
Description=etcd

[Service]
Type=notify
ExecStart=/usr/local/bin/etcd \\
  --name ${NODE_IP} \\
  --cert-file=/etc/etcd/kubernetes.pem \\
  --key-file=/etc/etcd/kubernetes-key.pem \\
  --peer-cert-file=/etc/etcd/kubernetes.pem \\
  --peer-key-file=/etc/etcd/kubernetes-key.pem \\
  --trusted-ca-file=/etc/etcd/ca.pem \\
  --peer-trusted-ca-file=/etc/etcd/ca.pem \\
  --peer-client-cert-auth \\
  --client-cert-auth \\
  --initial-advertise-peer-urls https://${NODE_IP}:2380 \\
  --listen-peer-urls https://${NODE_IP}:2380 \\
  --advertise-client-urls https://${NODE_IP}:2379 \\
  --listen-client-urls https://${NODE_IP}:2379,https://127.0.0.1:2379 \\
  --initial-cluster-token etcd-cluster-0 \\
  --initial-cluster ${ETCD1_IP}=https://${ETCD1_IP}:2380,${ETCD2_IP}=https://${ETCD2_IP}:2380,${ETCD3_IP}=https://${ETCD3_IP}:2380 \\
  --initial-cluster-state new \\
  --data-dir=/data/etcd
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

}
```

### Enable and Start etcd service:
```sh
{
  systemctl daemon-reload
  systemctl enable --now etcd
  systemctl start etcd
  systemctl status etcd
}
```
### Verify Etcd cluster status:
```sh
{
ETCDCTL_API=3 etcdctl member list \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/etcd/kubernetes.pem \
  --cert=/etc/etcd/kubernetes.pem \
  --key=/etc/etcd/kubernetes-key.pem
}
```
```sh
25ef0cb30a2929f1, started, k8s-master-1, https://k8s-master-1:2380, https://k8s-master-1:2379, false
5818a496a39840ca, started, k8s-master-2, https://k8s-master-2:2380, https://k8s-master-2:2379, false
c669cf505c64b0e8, started, k8s-master-3, https://k8s-master-3:2380, https://k8s-master-3:2379, false
```
*Repeat the above spets on other master nodes by replacing the ip address with respect to node ip address.*

### Initializing Master Node 1:

Create configuration file : 
```sh
{
HAPROXY_IP="haproxy-ip"

MASTER01_IP="k8s-master-1"
MASTER02_IP="k8s-master-2"
MASTER03_IP="k8s-master-3"

cat <<EOF > config.yaml
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
kubernetesVersion: stable
apiServer:
  certSANs:
  - ${HAPROXY_IP}
  ExtraArgs:
    apiserver-count: "3"
controlPlaneEndpoint: "${HAPROXY_IP}:6443"
etcd:
  external:
    endpoints:
    - https://${MASTER01_IP}:2379
    - https://${MASTER02_IP}:2379
    - https://${MASTER03_IP}:2379
    caFile: /etc/etcd/ca.pem
    certFile: /etc/etcd/kubernetes.pem
    keyFile: /etc/etcd/kubernetes-key.pem
networking:
  podSubnet: 10.244.0.0/16
apiServerExtraArgs:
  apiserver-count: "3"
EOF
}
```
### Initializing the node:

```sh
kubeadm init --config=config.yaml
```

*In order to make kubectl work for non-root users execute the following on master01:*

```sh
{    
  mkdir -p $HOME/.kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
  sudo chown $(id -u):$(id -g) $HOME/.kube/config
}
```

### Option 1: Activate the flannel CNI plugin:

```sh
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

*If you use custom podCIDR (not 10.244.0.0/16) you first need to download the above manifest and modify the network to match your one*
```

### Option 2: Activate the calico CNI plugin:
```sh
#Download the custom resources necessary to configure Calico
curl https://raw.githubusercontent.com/projectcalico/calico/v3.25.0/manifests/calico.yaml -O

#Update the CALICO_IPV4POOL_CIDR block in calico.yaml
- name: CALICO_IPV4POOL_CIDR
  value: '10.244.0.0/16'

# Install calico
kubectl create -f calico.yaml

# Confirm that all of the pods are running with the following command.
watch kubectl get pods -n kube-system
```
*Note: create a directory and store the custom-resources.yaml file for future reference*

### Option 3: Activate the Cilium CNI plugin:
```sh
#install Cilium cli
{
CILIUM_CLI_VERSION=$(curl -s https://raw.githubusercontent.com/cilium/cilium-cli/main/stable.txt)
CLI_ARCH=amd64
if [ "$(uname -m)" = "aarch64" ]; then CLI_ARCH=arm64; fi
curl -L --fail --remote-name-all https://github.com/cilium/cilium-cli/releases/download/${CILIUM_CLI_VERSION}/cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}
sha256sum --check cilium-linux-${CLI_ARCH}.tar.gz.sha256sum
sudo tar xzvfC cilium-linux-${CLI_ARCH}.tar.gz /usr/local/bin
rm cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}
}

#Install cilium 
cilium install --version 1.14.0

#validate the installation
cilium status --wait

#connectivity test
cilium connectivity test

```

### Initializing Master Node 2:

- Copy the kubernetets certs `(/etc/kubernetes/pki/*)` from the master node 1
- Copy the `config.yaml` from master node01

```sh
{
mkdir -p /etc/kubernetes/pki
scp /etc/kubernetes/pki/* root@k8smaster2:/etc/kubernetes/pki/
scp /etc/kubernetes/pki/* root@k8smaster3:/etc/kubernetes/pki/
}
```
```sh
kubeadm init --config=config.yaml
```

### Adding Worker nodes: 

Get the join command from master node and run it on worker nodes to join the cluster:

```sh
kubeadm token create --print-join-command
```
When you run kubeadm init you will get the join command as follows , you need to run this on all worker nodes.

```sh
kubeadm join 192.168.X.X:6443 --token 85iw5v.ymo1wqcs9mrqmnnf --discovery-token-ca-cert-hash sha256:4710a65a4f0a2be37c03249c83ca9df2377b4433c6564db4e61e9c07f5a213dd
```

### Use the below command to label the worker nodes:
```sh
kubectl label nodes k8sworker1 kubernetes.io/role=k8sworker1
```

### Reference Links:

- [Containerd](https://github.com/containerd/containerd)
- [etcd](https://github.com/etcd-io/etcd)
- [runc](https://github.com/opencontainers/runc)
- [cni plugins](https://github.com/containernetworking/plugins)
- [cilium](https://docs.cilium.io/en/stable/gettingstarted/k8s-install-default/)