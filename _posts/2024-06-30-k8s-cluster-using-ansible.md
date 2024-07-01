---
layout: post
title: "Deploy a kubernetes cluster using ansible"
date: 2024-06-30 01:06:00 +0800
categories: kuberntes
tags: ansible
image:
  path: /assets/img/headers/k8s-ansible.png
---

Deploying a Kubernetes cluster can be a daunting task, but with the right tools and guidance, it becomes manageable and efficient. In this blog post, I walk you through the process of setting up a Kubernetes cluster with single master and two worker nodes using Ansible role. 

This setup is perfect for development and testing environments, providing a solid foundation to explore Kubernetes' powerful orchestration capabilities. I will cover everything from preparing your environment to executing the Ansible playbook, ensuring you have a running cluster ready for your applications by the end of this guide. 

### Prerequisites:

Before we begin, there are a few prerequisites we need to address. Here’s what you'll need:

- 3 ubuntu 22.04 virtual machines
- 2 vCPUs & 4 GiB memory per node
- A host with Ansible installed

First, you’ll need to fork and clone the [repo](https://github.com/MBN02/Ansible.git).While you’re at it, give it a ⭐ too!

```sh
git clone https://github.com/MBN02/Ansible.git
```

Next, create a `ansible.cfg` file : 

```yaml
#example ansible.cfg file

[defaults]
inventory       = /Users/mohan.kumar.bn/inventory
command_warnings=False
host_key_checking = false
forks = 20
serial = 20
callback_whitelist = timer, profile_tasks
gathering = smart
stdout_callback = yaml
color = true

[ssh_connection]
pipelining = True
```

Creata an `inventory` file with your vm's ip address

```yaml
[control_plane]
master-node ansible_host=192.168.X.A #repalce the ip

[workers]
worker-node1 ansible_host=192.168.X.B #repalce the ip
worker-node2 ansible_host=192.168.X.C #repalce the ip

[all:vars]
ansible_python_interpreter=/usr/bin/python3

[control_plane:vars]
ansible_ssh_private_key_file= /Users/mohan.kumar.bn/.ssh/id_rsa
ansible_user=root

[workers:vars]
ansible_ssh_private_key_file= /Users/mohan.kumar.bn/.ssh/id_rsa
ansible_user=root
```

Next, create a playbook called  `setup_kubernetes.yml`
```yaml
---
- name: Setup Kubernetes Cluster
  hosts: all
  become: true
  roles:
    - setup-kuberntes-cluster
```

The final step is to execute the ansible role to bootstrap the cluster.

```yaml
#ansible-playbook -i inventory playbook.yaml
ansible-playbook setup_kubernetes.yml
```

Once done, login to controlplane node and run the following command to confirm if the cluster is created successfully.

```yaml
kubectl get nodes
```

### Reference Links:

- [Ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html)

- [kubeadm](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/)
