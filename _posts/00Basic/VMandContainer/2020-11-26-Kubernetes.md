---
title: Virtulization - Google Kubernetes
date: 2020-11-26 11:11:11 -0400
categories: [00Basic, VMyContainer]
tags: [Linux, VMs, Kubernetes]
math: true
image:
---

- [Virtulization - Google Kubernetes](#virtulization---google-kubernetes)
  - [basic](#basic)
  - [need of container orchestration tool](#need-of-container-orchestration-tool)
- [快速上手](#快速上手)
  - [基本概念](#基本概念)
    - [容器组 Pod](#容器组-pod)
      - [Pod 管理](#pod-管理)
      - [Pod 的使用](#pod-的使用)
      - [替代方案](#替代方案)
      - [Pod 的生命周期](#pod-的生命周期)
    - [services](#services)
    - [ConfigMap + secret](#configmap--secret)
    - [Data storgae - volume](#data-storgae---volume)
  - [架构设计考虑](#架构设计考虑)
  - [kubernetes architecture](#kubernetes-architecture)
    - [节点 Node](#节点-node)
      - [容器状态](#容器状态)
      - [主机IP](#主机ip)
      - [Node 周期](#node-周期)
      - [Node 状态](#node-状态)
      - [Node 管理](#node-管理)
      - [Node 控制](#node-控制)
  - [Productiom cluster setup](#productiom-cluster-setup)
    - [minikube](#minikube)
    - [kubectl 使用](#kubectl-使用)
  - [deploy](#deploy)
    - [Deployment](#deployment)
    - [StatefulSet](#statefulset)
  - [Network policy?](#network-policy)
    - [network policy security](#network-policy-security)



---


# Virtulization - Google Kubernetes



![k8s_architecture](https://i.imgur.com/ibs26v8.png)

## basic

- open source container orchestration tool
  - help to manage containerized application in different deployment environment (physical, virtual, cloud)

- Google 团队发起并维护的基于 Docker 的开源容器集群管理系统，


- 目标是管理跨多个主机的容器，提供基本的部署，维护以及运用伸缩，主要实现语言为 Go 语言


- 它不仅支持常见的云平台，而且支持内部数据中心。
  - 易学：轻量级，简单，容易理解
  - 便携：支持公有云，私有云，混合云，以及多种云平台
  - 可拓展：模块化，可插拔，支持钩子，可任意组合
  - 自修复：自动重调度，自动重启，自动复制


- 建于 Docker 之上的 Kubernetes 可以构建一个容器的调度服务，其目的是让用户透过 Kubernetes 集群来进行云端容器集群的管理，而无需用户进行复杂的设置工作。
- 系统会自动选取合适的工作Node 来执行具体的容器集群调度处理工作。
- 其核心概念是 Container Pod。
  - 一个 Pod 由一组工作于同一物理工作Node 的容器构成。
  - 这些组容器拥有相同的网络命名空间、IP以及存储配额，也可以根据实际情况对每一个 Pod 进行端口映射。
  - 此外，Kubernetes 工作Node 会由主系统进行管理，Node 包含了能够运行 Docker 容器所用到的服务。


---

## need of container orchestration tool

- trend from Monolith to Microservices

- increased usage of microservices

- proper way to manage it
  - High Availability - no down time
  - scalability or high performance - high response rate
  - Disaster recovery - backup and restore





---


# 快速上手
Kubernetes 支持在多种环境下的安装，包括本地主机(Fedora)、云服务(Google GAE、AWS 等)。
- 然而最快速体验 Kubernetes 的方式显然是本地通过 Docker 的方式来启动相关进程。

在单 Node 使用 Docker 快速部署一套 Kubernetes 的拓扑。

![k8s-singlenode-docker](https://i.imgur.com/MoGH8ac.png)

> Kubernetes 依赖 Etcd 服务来维护所有主Node 的状态。


```bash
# 启动 Etcd 服务。
$ docker run --net=host -d \
         gcr.io/google_containers/etcd:2.0.9 /usr/local/bin/etcd \
         --addr=127.0.0.1:4001 \
         --bind-addr=0.0.0.0:4001 \
         --data-dir=/var/etcd/data


# 启动主Node
# 启动 kubelet。
$ docker run --net=host -d -v /var/run/docker.sock:/var/run/docker.sock \
         gcr.io/google_containers/hyperkube:v0.17.0 /hyperkube kubelet --api_servers=http://localhost:8080 --v=2 --address=0.0.0.0 \
         --enable_server \
         --hostname_override=127.0.0.1 \
         --config=/etc/kubernetes/manifests


# 启动服务代理
$ docker run -d --net=host --privileged
         gcr.io/google_containers/hyperkube:v0.17.0 /hyperkube proxy --master=http://127.0.0.1:8080 --v=2


# 测试状态
# 在本地访问 8080 端口，可以获取到如下的结果：

$ curl 127.0.0.1:8080
{
  "paths": [
    "/api",
    "/api/v1beta1",
    "/api/v1beta2",
    "/api/v1beta3",
    "/healthz",
    "/healthz/ping",
    "/logs/",
    "/metrics",
    "/static/",
    "/swagger-ui/",
    "/swaggerapi/",
    "/validate",
    "/version"
  ]
}


# 查看服务
# 所有服务启动后，查看本地实际运行的 Docker 容器，有如下几个。
CONTAINER ID        IMAGE                                        COMMAND                CREATED             STATUS              PORTS               NAMES
ee054db2516c        gcr.io/google_containers/hyperkube:v0.17.0   "/hyperkube schedule   2 days ago          Up 1 days                               k8s_scheduler.509f29c9_k8s-master-127.0.0.1_default_9941e5170b4365bd4aa91f122ba0c061_e97037f5
3b0f28de07a2        gcr.io/google_containers/hyperkube:v0.17.0   "/hyperkube apiserve   2 days ago          Up 1 days                               k8s_apiserver.245e44fa_k8s-master-127.0.0.1_default_9941e5170b4365bd4aa91f122ba0c061_6ab5c23d
2eaa44ecdd8e        gcr.io/google_containers/hyperkube:v0.17.0   "/hyperkube control   2 days ago          Up 1 days                               k8s_controller-manager.33f83d43_k8s-master-127.0.0.1_default_9941e5170b4365bd4aa91f122ba0c061_1a60106f
30aa7163cbef        gcr.io/google_containers/hyperkube:v0.17.0   "/hyperkube proxy --   2 days ago          Up 1 days                               jolly_davinci
a2f282976d91        gcr.io/google_containers/pause:0.8.0         "/pause"               2 days ago          Up 2 days                               k8s_POD.e4cc795_k8s-master-127.0.0.1_default_9941e5170b4365bd4aa91f122ba0c061_e8085b1f
c060c52acc36        gcr.io/google_containers/hyperkube:v0.17.0   "/hyperkube kubelet    2 days ago          Up 1 days                               serene_nobel
cc3cd263c581        gcr.io/google_containers/etcd:2.0.9          "/usr/local/bin/etcd   2 days ago          Up 1 days                               happy_turing

# 这些服务大概分为三类：主Node 服务、工作Node 服务和其它服务。

# 主Node 服务
# apiserver 是整个系统的对外接口，提供 RESTful 方式供客户端和其它组件调用；

# scheduler 负责对资源进行调度，分配某个 pod 到某个Node 上；

# controller-manager 负责管理控制器，包括 endpoint-controller(刷新服务和 pod 的关联信息)和 replication-controller(维护某个 pod 的复制为配置的数值)。

# 工作Node 服务
# kubelet 是工作Node 执行操作的 agent，负责具体的容器生命周期管理，根据从数据库中获取的信息来管理容器，并上报 pod 运行状态等；

# proxy 为 pod 上的服务提供访问的代理。

# 其它服务
# Etcd 是所有状态的存储数据库；

# gcr.io/google_containers/pause:0.8.0 是 Kubernetes 启动后自动 pull 下来的测试镜像。
```



这些服务大概分为三类：`主Node 服务`、`工作Node 服务`和`其它服务`
- 主Node 服务
  - apiserver 是整个系统的对外接口，提供 RESTful 方式供客户端和其它组件调用；
  - scheduler 负责对资源进行调度，分配某个 pod 到某个Node 上；
  - controller-manager 负责管理控制器，包括 endpoint-controller(刷新服务和 pod 的关联信息)和 replication-controller(维护某个 pod 的复制为配置的数值)。

- 工作Node 服务
  - kubelet 是工作Node 执行操作的 agent，负责具体的容器生命周期管理，根据从数据库中获取的信息来管理容器，并上报 pod 运行状态等；
  - proxy 为 pod 上的服务提供访问的代理。

- 其它服务
  - Etcd 是所有状态的存储数据库；
  - `gcr.io/google_containers/pause:0.8.0` 是 Kubernetes 启动后自动 pull 下来的测试镜像。



---

## 基本概念

![k8s_architecture](https://i.imgur.com/ibs26v8.png)

* 节点(`Node`)：一个Node 是一个运行 Kubernetes 中的主机。
* 容器组 (`Pod`)：一个 Pod 对应于由若干Pod 成的一个Pod ，同个 Pod 内的容器共享一个存储卷(volume)。
* 容器组生命周期(`pos-states`)：包含所有容器状态集合，包括Pod 状态类型，Pod 生命周期，事件，重启策略，以及 replication controllers。
* `Replication Controllers`：主要负责指定数量的 pod 在同一时间一起运行。
* 服务(`services`)：一个 Kubernetes 服务是Pod 逻辑的高级抽象，同时也对外提供访问Pod 的策略。
* 卷(`volumes`)：一个卷就是一个目录，容器对其有访问权限。
* 标签(`labels`)：标签是用来连接一组对象的，比如Pod 。标签可以被用来组织和选择子对象。
* 接口权限(`accessing_the_api`)：端口，IP 地址和代理的防火墙规则。
* web 界面(`ux`)：用户可以通过 web 界面操作 Kubernetes。
* 命令行操作(`cli`)：`kubecfg`命令。

---



### 容器组 Pod



![Screen Shot 2022-04-05 at 12.14.12](https://i.imgur.com/m2i7iBm.png)


在 Kubernetes 中，使用的最小单位是Pod
- Pod 是创建，调度，管理的最小单位。
- 一个Pod 使用相同的 Docker 容器并共享卷(挂载点)。
- 一个Pod 是一个特定应用的打包集合，包含一个或多个容器。
- smallest unit

- abstraction over container

- create a layer of container
  - Pod : one application
  - Pod > container + container + container

- each Pod has one IP
  - pod communicate by using IP
  - new IP on re-creatyion


- Pod 被认为只有很短的运行周期。
  - Pod 被调度到一组Node 运行，直到容器的生命周期结束或者其被删除。
  - 如果Node 死掉，运行在其上的Pod 将会被删除而不是重新调度。(也许在将来的版本中会添加Pod 的移动)。


Pod 设计的初衷：资源共享和通信
- Pod 主要是为了数据共享和它们之间的通信。
- 在一个Pod 中，容器都使用相同的网络地址和端口，可以通过本地网络来相互通信。
- 每个Pod 都有独立的 IP，可用通过网络来和其他物理主机或者容器通信。
- Pod 有一组存储卷(挂载点)，主要是为了让容器在重启之后可以不丢失数据。


#### Pod 管理

Pod 是一个运用管理和部署的高层次抽象，同时也是一组容器的接口。Pod 是部署、水平放缩的最小单位。


#### Pod 的使用

Pod 可以通过组合来构建复杂的运用，其本来的意义包含：
* 内容管理，文件和数据加载以及本地缓存管理等。
* 日志和检查点备份，压缩，快照等。
* 监听数据变化，跟踪日志，日志和监控代理，消息发布等。
* 代理，网桥
* 控制器，管理，配置以及更新



---

#### 替代方案

为什么不在一个单一的容器里运行多个程序？

1. 透明化。为了使Pod 中的容器保持一致的基础设施和服务，比如进程管理和资源监控。这样设计是为了用户的便利性。
1. 解偶软件之间的依赖。每个容器都可能重新构建和发布，Kubernetes 必须支持热发布和热更新(将来)。
1. 方便使用。用户不必运行独立的程序管理，也不用担心每个运用程序的退出状态。
1. 高效。考虑到基础设施有更多的职责，容器必须要轻量化。


---

#### Pod 的生命周期

包括若干状态值：`pending`、`running`、`succeeded`、`failed`。

1. pending
   - Pod 已经被 Node 接受，但有一个或多个容器还没有运行起来。
   - 这将包含某些 Node 正在下载镜像的时间，这种情形会依赖于网络情况。

2. running
   - Pod 已经被调度到 Node ，并且所有的容器都已经启动。
   - 至少有一个容器处于运行状态(或者处于重启状态)。

3. succeeded
   - 所有的容器都正常退出。

4. failed
   - Pod 中所有容器都意外中断了。

通常来说，如果Pod 被创建了就不会自动销毁
- 除非被某种行为触发，而触发此种情况可能是人为，或者复制控制器所为。
- 唯一例外的是 Pod 由 succeeded 状态成功退出，或者在一定时间内重试多次依然失败。
- 如果某个 Node 死掉或者不能连接，那么Node 控制器将会标记其上的Pod 的状态为 `failed`。

举例如下。

* Pod 状态 `running`，有 1 容器，容器正常退出
  * 记录完成事件
  * 如果重启策略为：
    * 始终：重启容器，Pod 保持 `running`
    * 失败时：Pod 变为 `succeeded`
    * 从不：Pod 变为 `succeeded`

* Pod 状态 `running`，有1容器，容器异常退出
  * 记录失败事件
  * 如果重启策略为：
    * 始终：重启容器，Pod 保持 `running`
    * 失败时：重启容器，Pod 保持 `running`
    * 从不：Pod 变为 `failed`


* Pod 状态 `running`，有2容器，有1容器异常退出
  * 记录失败事件
  * 如果重启策略为：
    * 始终：重启容器，Pod 保持 `running`
    * 失败时：重启容器，Pod 保持 `running`
    * 从不：Pod 保持 `running`
        * 当有2容器退出
    * 记录失败事件
    * 如果重启策略为：
        * 始终：重启容器，Pod 保持 `running`
        * 失败时：重启容器，Pod 保持 `running`
        * 从不：Pod 变为 `failed`


* Pod 状态 `running`，容器内存不足
  * 标记容器错误中断
  * 记录内存不足事件
  * 如果重启策略为：
    * 始终：重启容器，Pod 保持 `running`
    * 失败时：重启容器，Pod 保持 `running`
    * 从不：记录错误事件，Pod 变为 `failed`



* Pod 状态 `running`，一块磁盘死掉
  * 杀死所有容器
  * 记录事件
  * Pod 变为 `failed`
  * 如果Pod 运行在一个控制器下，Pod 将会在其他地方重新创建


* Pod 状态 `running`，对应的Node 段溢出
  * Node 控制器等到超时
  * Node 控制器标记Pod  `failed`
  * 如果Pod 运行在一个控制器下，Pod 将会在其他地方重新创建




---


### services



![Screen Shot 2022-04-05 at 12.14.48](https://i.imgur.com/MDG9Llr.png)


- permanent IP attach to each pod
- lifecycle of Pof and service not connected

- App should be accessible through browser
  - external service:
    - http://a.b.c.d:port
    - http://myapp.com <- Ingree
  - internal service:
    - http:myDB


---



### ConfigMap + secret


- external configuration to application
  - URL_DB = mongo-db-service1
  - URL_DB = mongo-db-service2



![Screen Shot 2022-04-05 at 12.17.37](https://i.imgur.com/GLqFm3i.png)

secret
- to store credential
- base64 encoded



---


### Data storgae - volume

- data storage for Pod


- volumes attach to Pod (local/remote)

- keep storage permanent
  - k8s cluster do not manage data persistence





---

## 架构设计考虑

- 分布式架构，保证扩展性；
- 逻辑集中式的控制平面 + 物理分布式的运行平面；
- 一套资源调度系统，管理哪个容器该分配到哪个节点上；
- 一套对容器内服务进行抽象和 HA 的系统。


运行原理

![k8s_architecture](https://i.imgur.com/ibs26v8.png)

Kubernetes 是一套分布式系统
- 由多个节点组成，节点分为两类：
  - 一类是属于管理平面的 `主节点/控制节点(Master Node)`: 复杂的工作
  - 一类是属于运行平面的 `工作节点(Worker Node)`: 负责提供稳定的操作接口和能力抽象即可。

从这张图上，我们没有能发现 Kubernetes 中对于控制平面的分布式实现，但是由于数据后端自身就是一套分布式的数据库 Etcd，因此可以很容易扩展到分布式实现。



这些服务大概分为三类：`主Node 服务`、`工作Node 服务`和`其它服务`


![kube-proxy](https://i.imgur.com/KqhiUE6.png)


- **主Node 服务**
  - `apiserver` 是整个系统的对外接口，提供 RESTful 方式供客户端和其它组件调用；
  - `scheduler` 负责对资源进行调度，分配某个 pod 到某个Node 上；
  - `controller-manager`
    - 负责管理控制器，
    - 包括 endpoint-controller(刷新服务和 pod 的关联信息)
    - 和 replication-controller(维护某个 pod 的复制为配置的数值)。

- **工作Node 服务**
  - kubelet 是工作 Node 执行操作的 agent
  - 负责具体的容器生命周期管理，根据从数据库中获取的信息来管理容器，并上报 pod 运行状态等；
  - proxy 为 pod 上的服务提供访问的代理。
  - kube-proxy 是一个简单的网络访问代理，同时也是一个 Load Balancer。
  - 它负责将访问到某个服务的请求具体分配给工作节点上的 Pod（同一类标签）。


- **其它服务**
  - Etcd 是所有状态的存储数据库；
  - `gcr.io/google_containers/pause:0.8.0` 是 Kubernetes 启动后自动 pull 下来的测试镜像。








---


## kubernetes architecture

![Screen Shot 2022-04-05 at 12.28.56](https://i.imgur.com/hgtitTh.png)



---

### 节点 Node




**Node**
- 在 `Kubernetes` 中，Node 是实际工作的点
  - Worker Nodes do the work

- Node > Pod + Pod + Pod

- Node 可以是虚拟机或者物理机器，依赖于一个集群环境。

- 每个 Node 都有一些必要的服务以运行Pod ，并且它们都可以通过主 Node 来管理。


- 必要服务包括 Docker，kubelet 和代理服务。

- 3 processes mush be installed on every Node
  -  必要服务包括 Docker，kubelet 和代理服务。
  1. <font color=red> Container runtime </font>
     1. Container runtime: container ...
     2. Container runtime need to be installed on each pod
  2. <font color=red> Kubelet </font>
     1. get request from **scheduler**
     2. schedule those container
     3. interface with both the container and node
     4. kubelet starts the pod with a container inside
  3. <font color=red> Kube proxy </font>
     1. forward request from service to pod



![Screen Shot 2022-04-05 at 12.32.53](https://i.imgur.com/AtgjCvQ.png)


**Master**
- to interact with cluster
- need less resource than worker node


- 4 processes mush be installed on every Node
  - <font color=red> Api Server </font>
    - cluster gateway
    - get request and query
    - gatekeepper for Authentication :
      - validate the request
      - health request
  - <font color=red> scheduler </font>
    - decide
    - to schedule a new pod
    - decide the resource need
    - decide which worker node (check the **etcd**: A 30% used, B 60% used)
  - <font color=red> Controller manager </font>
    - detect cluster state change
      - pod dead -> recover cluster state
      - request to scheduler -> kubelet
  - <font color=red> etcd </font>
    - cluster brain
    - store actual application data
    - cluster state information
    - key value store
    - changes get stored in the key value store


![Screen Shot 2022-04-05 at 12.41.24](https://i.imgur.com/ZIu6XhR.jpg)






---




#### 容器状态

容器状态用来描述Node 的当前状态。现在，其中包含三个信息：


---


#### 主机IP

主机 IP 需要云平台来查询，`Kubernetes` 把它作为状态的一部分来保存。
- 如果 `Kubernetes` 没有运行在云平台上，Node  ID 就是必需的。
- IP 地址可以变化，并且可以包含多种类型的 IP 地址，如公共 IP，私有 IP，动态 IP，ipv6 等等。



#### Node 周期

通常来说Node 有 `Pending`，`Running`，`Terminated` 三个周期
- 如果 Kubernetes 发现了一个Node 并且其可用，那么 Kubernetes 就把它标记为 `Pending`。
- 然后在某个时刻，Kubernetes 将会标记其为 `Running`。
- Node 的结束周期称为 `Terminated`。
- 一个已经 `Terminated` 的Node 不会接受和调度任何请求，并且已经在其上运行的Pod 也会删除。



#### Node 状态

Node 的状态主要是用来描述处于 `Running` 的Node 。
- 可用的有 `NodeReachable` 和 `NodeReady`。以后可能会增加其他状态。
- `NodeReachable` 表示集群可达。
- `NodeReady` 表示 kubelet 返回 Status Ok 并且 HTTP 状态检查健康。


#### Node 管理

Node 并非 Kubernetes 创建，而是由云平台创建，或者就是物理机器、虚拟机。
- 在 Kubernetes 中，Node 仅仅是一条记录，
- Node 创建之后，Kubernetes 会检查其是否可用。
- 在 Kubernetes 中，Node 用如下结构保存：

```json
    {
      "id": "10.1.2.3",
      "kind": "Minion",
      "apiVersion": "v1beta1",
      "resources": {
        "capacity": {
          "cpu": 1000,
          "memory": 1073741824
        },
      },
      "labels": {
        "name": "my-first-k8s-node",
      },
    }
```

Kubernetes 校验Node 可用依赖于 ID。
- 在当前的版本中，有两个接口可以用来管理Node ：`Node 控制` 和 `Kube 管理`。

---

#### Node 控制

在 Kubernetes 主 Node 中，`Node 控制器` 是用来管理 Node 的组件。主要包含：
* 集群范围内 Node 同步
* 单 Node 生命周期管理

Node 控制有一个同步轮寻，主要监听所有云平台的虚拟实例，会根据Node 状态创建和删除。
- 可以通过 `--node_sync_period`标志来控制该轮寻。
- 如果一个实例已经创建，Node 控制将会为其创建一个结构。
- 同样的，如果一个Node 被删除，Node 控制也会删除该结构。
- 在 Kubernetes 启动时可用通过 `--machines`标记来显示指定Node 。
- 同样可以使用 `kubectl` 来一条一条的添加Node ，两者是相同的。
- 通过设置 `--sync_nodes=false`标记来禁止集群之间的Node 同步，你也可以使用 api/kubectl 命令行来增删Node 。


---





---


## Productiom cluster setup


- multiple master and worker nodes(VM or local)




---


### minikube

> Virtulization / Hypervisor is needed


to Test/local cluster setup

minikube
- create VM box on laptop
- nodes runs in CM
- master and node on one node
- 1 node K8s cluster
- for testing

![Screen Shot 2022-04-05 at 12.46.15](https://i.imgur.com/OCzNoKX.png)

```bash
$ brew install minikube


$ minikube start --driver=docker --alsologtostderr

$ minikube status
# minikube
# type: Control Plane
# host: Running
# kubelet: Running
# apiserver: Running
# kubeconfig: Configured
```


---

### kubectl 使用


- wat to interact to kubernet cluster
- API server:
  - enable interaction with cluster
  - way: UI, API, CLI (**kubectl**)
  - worker processes do the kebectl action

- for both minikube and cloud cluster



- kubectl 是 Kubernetes 自带的客户端，可以用它来直接操作 Kubernetes。



```bash
# 使用格式有两种：
kubectl [flags]
kubectl [command]


$ kubectl get nodes
# NAME       STATUS   ROLES                  AGE   VERSION
# minikube   Ready    control-plane,master   53s   v1.23.3
$ kubectl get pod
$ kubectl get services



$ kubectl version
# Client Version: version.Info{Major:"1", Minor:"23", GitVersion:"v1.23.5", GitCommit:"c285e781331a3785a7f436042c65c5641ce8a9e9", GitTreeState:"clean", BuildDate:"2022-03-16T15:51:05Z", GoVersion:"go1.17.8", Compiler:"gc", Platform:"darwin/arm64"}
# Server Version: version.Info{Major:"1", Minor:"23", GitVersion:"v1.23.3", GitCommit:"816c97ab8cff8a1c72eccca1026f7820e93e0d25", GitTreeState:"clean", BuildDate:"2022-01-25T21:19:12Z", GoVersion:"go1.17.6", Compiler:"gc", Platform:"linux/arm64"}



```

---


## deploy


distributed system and containers
- cluster is good for Stateless app

![Screen Shot 2022-04-05 at 12.25.27](https://i.imgur.com/eBRuULA.png)

- Pod:
  - not rely on one pod
  - replicate every in another node
  - <font color=red> Deployment </font>:
    - you create deployments
    - define blueprints for pod: duplicate=3


- service is also load balancer

- DB cannot be replicate
  - read and write on same share data storage
  - avoid data inconsistency
  - <font color=red> StatefulSet </font> for SateFul app
  - DB are oftern hosted outside of K8s cluster
    - use external database




### Deployment

- bluepring for creating pos
- most basic configuratiyon for deployment
- rest default

- Deployment
  - ReplicaSet manages Pods
  - Pod is an abstraction of container
  - container


```bash
$ kubectl create deployment NAME --image=image [-dry-run]

$ kubectl create deployment nginx-depl --image=nginx


$ kubectl get deployment
# NAME         READY   UP-TO-DATE   AVAILABLE   AGE
# nginx-depl   1/1     1            1           36s


$ kubectl get replicaset
# NAME                    DESIRED   CURRENT   READY   AGE
# nginx-depl-5ddc44dd46   1         1         1       10m


$ kubectl get pod
# NAME                          READY   STATUS    RESTARTS   AGE
# nginx-depl-5ddc44dd46-q5cmj   1/1     Running   0          44s




$ kubectl edit deployment nginx-depl
# Please edit the object below. Lines beginning with a '#' will be ignored,
# and an empty file will abort the edit. If an error occurs while saving this file will be
# reopened with the relevant failures.
#
apiVersion: apps/v1
kind: Deployment
metadata:
  annotations:
    deployment.kubernetes.io/revision: "1"
  creationTimestamp: "2022-04-05T22:01:23Z"
  generation: 1
  labels:
    app: nginx-depl
  name: nginx-depl
  namespace: default
  resourceVersion: "5536"
  uid: d1d87196-a2e4-44f8-b294-d27e112b37ee
spec:
  progressDeadlineSeconds: 600
  replicas: 1
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      app: nginx-depl
  strategy:
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
    type: RollingUpdate
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: nginx-depl
    spec:
      containers:
      - image: nginx:1.16
        imagePullPolicy: Always
        name: nginx
        resources: {}
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      terminationGracePeriodSeconds: 30
status:
  availableReplicas: 1
  conditions:
  - lastTransitionTime: "2022-04-05T22:01:33Z"
    lastUpdateTime: "2022-04-05T22:01:33Z"
    message: Deployment has minimum availability.
    reason: MinimumReplicasAvailable
    status: "True"
    type: Available
  - lastTransitionTime: "2022-04-05T22:01:23Z"
    lastUpdateTime: "2022-04-05T22:01:33Z"
    message: ReplicaSet "nginx-depl-5ddc44dd46" has successfully progressed.
    reason: NewReplicaSetAvailable
    status: "True"
    type: Progressing
  observedGeneration: 1
  readyReplicas: 1
  replicas: 1
  updatedReplicas: 1


# HorizontalPodAutoscalers
kubectl get hpa


# view on workers in a cold state without any traffic.
kubectl get nodes -L role
```







### StatefulSet


---

## Network policy?

- Network policy is the primary tool for securing a Kubernetes network.

- It restrict the network traffic in the cluster so only the traffic want to flow is allowed.

> how network security was typically achieved prior to network policy.
> - physical topology: Historically in enterprise networks, network security was provided by designing a physical topology of network devices (switches, routers, firewalls) and their associated configuration. The physical topology defined the security boundaries of the network.
> - virtualization: In the first phase of virtualization, the same network and network device constructs were virtualized in the cloud, and the same techniques for creating specific network topologies of (virtual) network devices were used to provide network security. Adding new applications or services often required additional network design to update the network topology and network device configuration to provide the desired security.

- In contrast, the Kubernetes network model defines a "flat" network in which every pod can communicate with all other pods in the cluster using pod IP addresses. This approach massively simplifies network design and allows new workloads to be scheduled dynamically anywhere in the cluster with no dependencies on the network design.

- In this model, rather than network security being defined by `network topology boundaries`, it is defined using `network policies` that are independent of the network topology.

- Network policies are further abstracted from the network by using `label selectors` as their primary mechanism for defining which workloads can talk to which workloads, rather than IP addresses or IP address ranges.

### network policy security

- While you can (and should) use firewalls to restrict traffic at the perimeters of the network (commonly referred to as north-south traffic), their ability to police Kubernetes traffic is often limited to a granularity of the cluster as a whole, rather than to specific groups of pods, due to the dynamic nature of pod scheduling and pod IP addresses. In addition, the goal of most attackers once they gain a small foothold inside the perimeter is to move laterally (commonly referred to as east-west) to gain access to higher value targets, which perimeter based firewalls can't police against.

Network policy on the other hand is designed for the dynamic nature of Kubernetes by following the standard Kubernetes paradigm of using label selectors to define groups of pods, rather than IP addresses. And because network policy is enforced within the cluster itself it can police both north-south and east-west traffic.

Network policy represents an important evolution of network security, not just because it handles the dynamic nature of modern microservices, but because it empowers dev and devops engineers to easily define network security themselves, rather than needing to learn low-level networking details or raise tickets with a separate team responsible for managing firewalls. Network policy makes it easy to define intent, such as "only this microservice gets to connect to the database", write that intent as code (typically in YAML files), and integrate authoring of network policies into git workflows and CI/CD processes.

















.
