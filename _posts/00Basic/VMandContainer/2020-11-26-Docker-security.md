---
title: Virtulization - Docker security
date: 2020-11-26 11:11:11 -0400
categories: [00Basic, VMandContainer, Containers]
tags: [Linux, VMs, Docker]
math: true
image:
---

[toc]

---


# Docker security


---

## basic

评估 Docker 的安全性时，主要考虑三个方面:
- 由内核的命名空间和控制组机制提供的 容器内在安全
- Docker 程序（特别是服务端）本身的抗攻击性
- 内核安全性的加强机制 对 容器安全性的影响

### namespace
- Docker 容器和 LXC 容器很相似，所提供的安全特性也差不多。
- 当用 docker run 启动一个容器时，在后台 Docker 为容器创建了一个独立的`命名空间 namespace 和控制组 cgroups`集合。
- 命名空间提供了最基础也是最直接的隔离，在容器中运行的进程不会被运行在主机上的进程和其它容器发现和作用。
- 每个容器都有自己独有的网络栈，意味着它们不能访问其他容器的 sockets 或接口。
  - 不过，如果主机系统上做了相应的设置，容器可以像跟主机交互一样的和其他容器交互。
  - 当指定公共端口或使用 links 来连接 2 个容器时，容器就可以相互通信了（可以根据配置来限制通信的策略）。
- 从网络架构的角度来看，所有的容器通过本地主机的网桥接口相互通信，就像物理机器通过物理交换机通信一样。
- 那么，内核中实现命名空间和私有网络的代码是否足够成熟？

### cgroup
- 它提供了很多有用的特性；以及确保各个容器可以公平地分享主机的内存、CPU、磁盘 IO 等资源；当然，更重要的是，控制组确保了当容器内的资源使用产生压力时不会连累主机系统。
- 尽管控制组不负责隔离容器之间相互访问、处理数据和进程，它在防止拒绝服务（DDOS）攻击方面是必不可少的。尤其是在多用户的平台（比如公有或私有的 PaaS）上，控制组十分重要。例如，当某些应用程序表现异常的时候，可以保证一致地正常运行和性能。

### Daemon sec
- 确保只有可信的用户才可以访问 Docker 服务。Docker 允许用户在主机和容器间共享文件夹，同时不需要限制容器的访问权限，这就容易让容器突破资源限制。例如，恶意用户启动容器的时候将主机的根目录/映射到容器的 /host 目录中，那么容器理论上就可以对主机的文件系统进行任意修改了。
- 终极目标是改进 2 个重要的安全特性：
- 将容器的 root 用户映射到本地主机上的非 root 用户，减轻容器和主机之间因权限提升而引起的安全问题；
- 允许 Docker 服务端在非 root 权限下运行，利用安全可靠的子进程来代理执行需要特权权限的操作。这些子进程将只允许在限定范围内进行操作，例如仅负责虚拟网络设定或文件系统管理、配置操作等。


### Kernel capabililty 内核能力机制
- 能力机制（Capability）是 Linux 内核一个强大的特性，可以提供细粒度的权限访问控制。 Linux 内核自 2.2 版本起就支持能力机制，它将权限划分为更加细粒度的操作能力，既可以作用在进程上，也可以作用在文件上。
- 例如，一个 Web 服务进程只需要绑定一个低于 1024 的端口的权限，并不需要 root 权限。那么它只需要被授权 net_bind_service 能力即可。此外，还有很多其他的类似能力来避免进程获取 root 权限。
- 默认情况下，Docker 启动的容器被严格限制只允许使用内核的一部分能力。
- 使用能力机制对加强 Docker 容器的安全有很多好处。
  - 通常，在服务器上会运行一堆需要特权权限的进程，包括有 ssh、cron、syslogd、硬件管理工具模块（例如负载模块）、网络配置工具等等。
  - 容器跟这些进程是不同的，因为几乎所有的特权进程都由容器以外的支持系统来进行管理。
  - ssh 访问被主机上ssh服务来管理；
  - cron 通常应该作为用户进程执行，权限交给使用它服务的应用来处理；
  - 日志系统可由 Docker 或第三方服务管理；
  - 硬件管理无关紧要，容器中也就无需执行 udevd 以及类似服务；
  - 网络管理也都在主机上设置，除非特殊需求，容器不需要对网络进行配置。
  - 从上面的例子可以看出，大部分情况下，容器并不需要“真正的” root 权限，容器只需要少数的能力即可。为了加强安全，容器可以禁用一些没必要的权限。
  - 完全禁止任何 mount 操作；
  - 禁止直接访问本地主机的套接字；
  - 禁止访问一些文件系统的操作，比如创建新的设备、修改文件属性等；
  - 禁止模块加载。
  - 这样，就算攻击者在容器中取得了 root 权限，也不能获得本地主机的较高权限，能进行的破坏也有限。
- 默认情况下，Docker采用白名单机制，禁用必需功能之外的其它权限。
- 当然，用户也可以根据自身需求来为 Docker 容器启用额外的权限。



> 总结
> 总体来看，Docker 容器还是十分安全的，特别是在容器内不使用 root 权限来运行进程的话。
> 另外，用户可以使用现有工具，比如 Apparmor, SELinux, GRSEC 来增强安全性；甚至自己在内核中实现更复杂的安全机制。


---


## Rules


---

### RULE #0 - Keep Host and Docker up to date

To prevent from known, container escapes vulnerabilities, which typically end in escalating to root/administrator privileges, patching Docker Engine and Docker Machine is crucial.

In addition, containers (unlike in virtual machines) share the kernel with the host, therefore kernel exploits executed inside the container will directly hit host kernel. For example, kernel privilege escalation exploit ([like Dirty COW](https://github.com/scumjr/dirtycow-vdso)) executed inside a well-insulated container will result in root access in a host.

---

### RULE #1 - Do not expose the Docker daemon socket (even to the containers)

Docker socket `/var/run/docker.sock` is the UNIX socket that Docker is listening to.
- This is the primary entry point for the Docker API.
- The owner of this socket is root.
- Giving someone access to it is equivalent to giving unrestricted root access to your host.

**Do not enable _tcp_ Docker daemon socket.**
- If you are running docker daemon with `-H tcp://0.0.0.0:XXX`
- or similar are exposing un-encrypted and un-authenticated direct access to the Docker daemon.
- If you really have to do this, you should secure it.
- Check how to do this [following Docker official documentation](https://docs.docker.com/engine/reference/commandline/dockerd/#daemon-socket-option).

**Do not expose _/var/run/docker.sock_ to other containers**.
- If you are running your docker image with `-v /var/run/docker.sock://var/run/docker.sock` or similar, you should change it.
- mounting the socket read-only is not a solution but only makes it harder to exploit.
- Equivalent in the docker-compose file is something like this:

```
volumes:
- "/var/run/docker.sock:/var/run/docker.sock"
```

---

### RULE #2 - Set a user

Configuring the container to use an unprivileged user is the best way to prevent privilege escalation attacks.

This can be accomplished in three different ways as follows:

1. During runtime
   - `docker run -u 4000 alpine`

2. During build time.
   - Simple add user in Dockerfile and use it. For example:

    ```dockerfile
    FROM alpine
    RUN groupadd -r myuser && useradd -r -g myuser myuser
    <HERE DO WHAT YOU HAVE TO DO AS A ROOT USER LIKE INSTALLING PACKAGES ETC.>
    USER myuser
    ```

3. Enable user namespace support (`--userns-remap=default`) in [Docker daemon](https://docs.docker.com/engine/security/userns-remap/#enable-userns-remap-on-the-daemon)

In kubernetes, this can be configured in [Security Context](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/) using `runAsNonRoot` field e.g.:

```
kind: ...
apiVersion: ...
metadata:
  name: ...
spec:
  ...
  containers:
  - name: ...
    image: ....
    securityContext:
          ...
          runAsNonRoot: true
          ...
```

As a Kubernetes cluster administrator, you can configure it using [Pod Security Policies](https://kubernetes.io/docs/concepts/policy/pod-security-policy/).

---

### RULE #3 - Limit capabilities (Grant only specific capabilities, needed by a container)

[Linux kernel capabilities](http://man7.org/linux/man-pages/man7/capabilities.7.html) are a set of privileges that can be used by privileged.
- Docker, by default, runs with only a subset of capabilities.
- You can change it and drop some capabilities (using `--cap-drop`) to harden your docker containers, or add some capabilities (using `--cap-add`) if needed.
- Remember not to run containers with the `--privileged` flag - this will add ALL Linux kernel capabilities to the container.

The most secure setup is to drop all capabilities `--cap-drop all` and then add only required ones.

For example:
- `docker run --cap-drop all --cap-add CHOWN alpine`

**And remember: Do not run containers with the _\--privileged_ flag!!!**

In kubernetes this can be configured in [Security Context](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/) using `capabilities` field e.g.:

```
kind: ...
apiVersion: ...
metadata:
  name: ...
spec:
  ...
  containers:
  - name: ...
    image: ....
    securityContext:
          ...
          capabilities:
            drop:
            - all
            add:
            - CHOWN
          ...
```

As a Kubernetes cluster administrator, you can configure it using [Pod Security Policies](https://kubernetes.io/docs/concepts/policy/pod-security-policy/).

---

### RULE #4 - Add –no-new-privileges flag

Always run your docker images with `--security-opt=no-new-privileges` in order to prevent escalate privileges using `setuid` or `setgid` binaries.

In kubernetes, this can be configured in [Security Context](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/) using `allowPrivilegeEscalation` field e.g.:

```
kind: ...
apiVersion: ...
metadata:
  name: ...
spec:
  ...
  containers:
  - name: ...
    image: ....
    securityContext:
          ...
          allowPrivilegeEscalation: false
          ...
```

As a Kubernetes cluster administrator, you can refer to Kubernetes documentation to configure it using [Pod Security Policies](https://kubernetes.io/docs/concepts/policy/pod-security-policy/).

---

### RULE #5 - Disable inter-container communication (--icc=false)

By default inter-container communication (icc) is enabled
- it means that all containers can talk with each other (using [`docker0` bridged network](https://docs.docker.com/v17.09/engine/userguide/networking/default_network/container-communication/#communication-between-containers)).

disabled it by running docker daemon with `--icc=false` flag. disabled icc.
-then it is required to tell which containers can communicate using `--link=CONTAINER\_NAME\_or\_ID:ALIAS` option.
- See more in [Docker documentation - container communication](https://docs.docker.com/v17.09/engine/userguide/networking/default_network/container-communication/#communication-between-containers)

In Kubernetes [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/) can be used for it.

---

### RULE #6 - Use Linux Security Module (seccomp, AppArmor, or SELinux)

**do not disable default security profile!**

Consider using security profile like [seccomp](https://docs.docker.com/engine/security/seccomp/) or [AppArmor](https://docs.docker.com/engine/security/apparmor/).

Instructions how to do this inside Kubernetes can be found at [Security Context documentation](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/) and in [Kubernetes API documentation](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.18/#securitycontext-v1-core)

---

### RULE #7 - Limit resources (memory, CPU, file descriptors, processes, restarts)

The best way to avoid DoS attacks is by limiting resources.
- You can limit
- [memory](https://docs.docker.com/config/containers/resource_constraints/#memory),
- [CPU](https://docs.docker.com/config/containers/resource_constraints/#cpu),
- maximum number of restarts (`--restart=on-failure:<number_of_restarts>`),
- maximum number of file descriptors (`--ulimit nofile=<number>`)
- and maximum number of processes (`--ulimit nproc=<number>`).

[Check documentation for more details about ulimits](https://docs.docker.com/engine/reference/commandline/run/#set-ulimits-in-container---ulimit)

Kubernetes:
- [Assign Memory Resources to Containers and Pods](https://kubernetes.io/docs/tasks/configure-pod-container/assign-memory-resource/),
- [Assign CPU Resources to Containers and Pods](https://kubernetes.io/docs/tasks/configure-pod-container/assign-cpu-resource/)
- and [Assign Extended Resources to a Container](https://kubernetes.io/docs/tasks/configure-pod-container/extended-resource/)


---

### RULE #8 - Set filesystem and volumes to read-only

1. **Run containers with a read-only filesystem**
   - using `--read-only` flag. For example:
   - `docker run --read-only alpine sh -c 'echo "whatever" > /tmp'`

2. If an application inside a container has to save something temporarily,
   - combine `--read-only` flag with `--tmpfs` like this:
   - `docker run --read-only --tmpfs /tmp alpine sh -c 'echo "whatever" > /tmp/file'`

Equivalent in the docker-compose file will be:

```
version: "3"
services:
  alpine:
    image: alpine
    read_only: true
```

kubernetes in [Security Context](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/) will be:

```
kind: ...
apiVersion: ...
metadata:
  name: ...
spec:
  ...
  containers:
  - name: ...
    image: ....
    securityContext:
          ...
          readOnlyRootFilesystem: true
          ...
```

In addition, if the volume is mounted only for reading **mount them as a read-only**
- It can be done by appending `:ro` to the `-v` like this:
  - `docker run -v volume-name:/path/in/container:ro alpine`
- Or by using `--mount` option:
  - `docker run --mount source=volume-name,destination=/path/in/container,readonly alpine`


---

### RULE #9 - Use static analysis tools

To detect containers with known vulnerabilities - scan images using static analysis tools.

- Free
  - [Clair](https://github.com/coreos/clair)
  - [Trivy](https://github.com/knqyf263/trivy)
- Commercial
  - [Snyk](https://snyk.io/) **(open source and free option available)**
  - [anchore](https://anchore.com/opensource/) **(open source and free option available)**
  - [Aqua Security's MicroScanner](https://github.com/aquasecurity/microscanner) **(free option available for rate-limited number of scans)**
  - [JFrog XRay](https://jfrog.com/xray/)
  - [Qualys](https://www.qualys.com/apps/container-security/)

To detect misconfigurations in Kubernetes:

- [kubeaudit](https://github.com/Shopify/kubeaudit)
- [kubesec.io](https://kubesec.io/)
- [kube-bench](https://github.com/aquasecurity/kube-bench)

To detect misconfigurations in Docker:

- [inspec.io](https://www.inspec.io/docs/reference/resources/docker/)
- [dev-sec.io](https://dev-sec.io/baselines/docker/)

---

### RULE #10 - Set the logging level to at least INFO

By default, the Docker daemon is configured to have a base logging level of `'info'`
- if this is not the case: set the Docker daemon log level to 'info'.
- Rationale: Setting up an appropriate log level, configures the Docker daemon to log events that you would want to review later.
- A base log level of 'info' and above would capture all logs except the debug logs.
- Until and unless required, you should not run docker daemon at the 'debug' log level.

To configure the log level in docker-compose:
- `docker-compose --log-level info up`

---

### Rule #11 - Lint the Dockerfile at build time

Many issues can be prevented by following some best practices when writing the Dockerfile.
- Adding a security linter as a step in the the build pipeline can go a long way in avoiding further headaches. Some issues that are worth checking are:

- Ensure a `USER` directive is specified
- Ensure the `base image version` is pinned
- Ensure the `OS packages versions` are pinned
- Avoid the use of `ADD` in favor of `COPY`
- Avoid the use of `apt/apk upgrade`
- Avoid curl bashing in `RUN` directives



---



Ref:
- [Docker Baselines on DevSec](https://dev-sec.io/baselines/docker/)
- [Use the Docker command line](https://docs.docker.com/engine/reference/commandline/cli/)
- [Overview of docker-compose CLI](https://docs.docker.com/compose/reference/overview/)
- [Configuring Logging Drivers](https://docs.docker.com/config/containers/logging/configure/)
- [View logs for a container or service](https://docs.docker.com/config/containers/logging/)
- [Dockerfile Security Best Practices](https://cloudberry.engineering/article/dockerfile-security-best-practices/)
