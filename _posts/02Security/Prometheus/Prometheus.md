

- [Prometheus](#prometheus)
	- [Prometheus: 监控与告警](#prometheus-监控与告警)
		- [1:概要介绍](#1概要介绍)
			- [主要特性](#主要特性)
			- [整体架构](#整体架构)
		- [2:安装方法](#2安装方法)
			- [安装方式](#安装方式)
			- [安装示例](#安装示例)
			- [结果确认](#结果确认)
		- [Prometheus：监控与告警：3:指标监控示例](#prometheus监控与告警3指标监控示例)
			- [事前准备](#事前准备)
			- [启动监控对象进程](#启动监控对象进程)
			- [配置Prometheus](#配置prometheus)
			- [启动Prometheus服务](#启动prometheus服务)
			- [结果确认](#结果确认-1)
		- [Prometheus：监控与告警：4:使用Grafana进行可视化显示](#prometheus监控与告警4使用grafana进行可视化显示)
			- [事前准备](#事前准备-1)
		- [Prometheus：监控与告警：5:在Kubernetes上部署](#prometheus监控与告警5在kubernetes上部署)
		- [Prometheus：监控与告警：6: Exporter概要介绍](#prometheus监控与告警6-exporter概要介绍)
	- [monitoring with Prometheus](#monitoring-with-prometheus)
	- [client](#client)
		- [Instrumenting applications](#instrumenting-applications)
			- [Installation](#installation)
			- [How Go exposition works](#how-go-exposition-works)
			- [Add own metrics](#add-own-metrics)
			- [Other Go client features](#other-go-client-features)
		- [Client for the Prometheus HTTP API](#client-for-the-prometheus-http-api)

- https://github.com/prometheus/client_golang
- [Endpoint](https://prometheus.io/docs/guides/go-application/)
- [P go SDK](https://pkg.go.dev/github.com/prometheus/client_golang/prometheus)
- https://github.com/prometheus/prometheus
- https://prometheus.io/docs/introduction/overview/
- https://blog.csdn.net/liumiaocn/category_9561738.html

# Prometheus

![pic](https://img-blog.csdnimg.cn/20191203090913325.jpeg?x-oss-process=image/resize,m_fixed,h_224,w_224)

## Prometheus: 监控与告警


### 1:概要介绍


![pic](https://img-blog.csdnimg.cn/20191205135124878.png#pic_center)

Prometheus是开源的监控告警的解决方案
- 最早由SoundCloud公司所开发和开源，
- 从2012年产生至今已经7年，在技术不断变化的时代这已经是一个很长的时间了，在过去的7年里，Prometheus也得到了越来越多用户的使用和推崇，并且在2016年加入CNCF后，成为继Kubernetes之后最早从中毕业的项目。
- Prometheus的开发语言以Go为主

![pic](https://img-blog.csdnimg.cn/20191205142859465.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 同样在CNCF的landcape的同一领域的还有Grafana、Sensu、graphite、Zabbix等。

![pic](https://img-blog.csdnimg.cn/20191205143232549.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 适合的场景
  - 非常适合纯数字类型的时序列数据，无论是以机器为中心的监控场景，还是高度动态变化的面向服务架构下的监控场景，Prometheus都是非常适合的。在微服务的世界中，Prometheus对多维度数据收集和查询的支持能力，是其长袖善舞之处。
  - Prometheus被设计用来支撑可靠性，当系统出现问题时能够快速判断故障原因，因此，每个Prometheus服务器都是独立的，不会依赖于网络存储或者其他远程的服务。当你其他的基础设施出现问题的时候，它还是独立可用的，而且不需要昂贵的基础设施的支撑才能够使用。


- 不适合的场景
  - Prometheus看重的是可靠性而不是精确性，使用Prometheus，无论什么样的故障境况下，用户总是能够看到可以看到的统计信息，这部分信息往往来源于独立运行的没有发生故障的那些Prometheus服务器所提供。
  - 但是如果使用者需要100%的精确，比如实时的交易系统的每次请求的完整性与精确性的场景下，Prometheus可能就无法很好与完整地收集到所有的数据，这种系统或者场景之下，可以选取其他技术用于保证这些数据的完整性与精确性，而Prometheus则可以用来进行监控。


#### 主要特性

* 多维度数据模型: 可使用指标名称和键/值对结合进行数据的管理。
* 提供灵活的查询语言PromQL，利用PromQL能更好地为数据先可视化显示或者告警功能提供更好的数据。
* 不依赖于分布式存储，单个服务节点的数据是自主的，可独立使用。
* 时序列数据库可基于HTTP协议使用PULL模式拉取数据。
* 提供中间网关的方式，也支持向中间网关以PUSH模式推送数据，而Prometheus再从中间网关拉取数据。
* 可以通过服务发现或者静态配置的方式发现目标对象。
* 支持多种方式的图表和仪表盘展示，比如对Grafana的支持。

#### 整体架构

整体架构如下图所示

![pic](https://img-blog.csdnimg.cn/20191205150351782.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)
可以看到，Prometheus的构成非常清晰，主要由如下几个部分构成:

* **Pushgateway**: 支持时序列数据使用PUSH模式推送的中间网关
* **Prometheus Server**: 提供PromQL查询语言能力、负责数据的采集和存储等主要功能
* **Alertmanager**: 是一个独立的组件，提供灵活的报警方式，比如Email、webhook等。
* **Web UI**: Prometheus本身提供一个非常简单的Web UI界面，这也是其代码中由小比例的JavaScript和TypeScript的原因。当然也可以结合Grafana进行监控数据的可视化展示。



---

### 2:安装方法

#### 安装方式

Prometheus支持多种安装方式，比如：

```bash
# 1. Docker镜像方式
# 最简单的方式莫过于直接使用Prometheus提供的官方镜像，
# 使用如下执行命令即可在本地9090端口启动Prometheus服务。
docker run --name prometheus -d -p 127.0.0.1:9090:9090 prom/prometheus

# 2. 二进制文件方式
# 下载预先编译好的二进制文件然后进行安装和设置，下载地址可以是：
# - https://prometheus.io/download/
# - https://github.com/prometheus/prometheus/releases
# 启动方法
# - 解压操作系统相对应的二进制文件包，提供本地设定文件，
# - 使用如下命令即可启动Prometheus服务了
./prometheus --config.file=prometheus.yml

# 3. 源码编译方式
# 构建
# - Prometheus使用go语言开发，所以本地只需要提供go的开发环境
# - 然后git clone源码之后执行make即可构建出二进制文件
mkdir -p $GOPATH/src/github.com/prometheus
cd $GOPATH/src/github.com/prometheus
git clone https://github.com/prometheus/prometheus.git
cd prometheus
make build
# 启动服务
./prometheus --config.file=prometheus.yml


# 4. 其他安装方式
Ansible
* https://github.com/cloudalchemy/ansible-prometheus
Chef
* https://github.com/elijah/chef-prometheus
Puppet
* https://forge.puppet.com/puppet/prometheus
SaltStack
* https://github.com/saltstack-formulas/prometheus-formula
Helm
* 直接使用helm install进行安装，需要注意的是相关版本的Chart是对于Helm 2还是Helm 3的支持。
```

#### 安装示例

整体来说Prometheus的安装就是一个二进制文件，下载、解压并设定权限，然后通过config.file指定配置文件即可完成安装并启动服务

各种方式基本上都是简化这一原本就很简单的过程而已。

这里以Docker方式的服务启动为例进行说明：

配置文件准备
```yaml
# prometheus-demo.yml
global:
  scrape_interval: 10s
  evaluation_interval: 10s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["127.0.0.1:9090"]
        labels:
          group: "prometheus"
```

```bash
# 下载最新版本的Prometheus镜像
docker pull prom/prometheus

# 启动Prometheus服务
# 我们把配置文件放在本地 ~/docker/prometheus/prometheus.yml，这样可以方便编辑和查看
# 通过 -v 参数将本地的配置文件挂载到 /etc/prometheus/ 位置，这是 prometheus 在容器中默认加载的配置文件位置。
# 如果我们不确定默认的配置文件在哪，可以先执行上面的不带 -v 参数的命令，然后通过 docker inspect 命名看看容器在运行时默认的参数有哪些（下面的 Args 参数）：
docker run -d -p 9090:9090 \
	-v `pwd`/prometheus-demo.yml:/etc/prometheus/prometheus.yml \
	--name prometheus prom/prometheus

docker container start prometheus
```


#### 结果确认

通过`http://localhost:9090` 访问启动起来的Prometheus的UI界面，缺省进入的是Grap标签的所在页面：

![pic](https://img-blog.csdnimg.cn/20191231055620111.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

Alerts标签界面信息

![pic](https://img-blog.csdnimg.cn/20191231060311508.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

Status标签界面信息

![pic](https://img-blog.csdnimg.cn/20191231060332842.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)


Prometheus使用REACT重新改造了界面的显示，可以通过点
击Try experimental React UI进行体验

![pic](https://img-blog.csdnimg.cn/20191231060649331.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)


确认metrics内容 `http://localhost:9090/metrics`

![pic](https://img-blog.csdnimg.cn/20191231061551286.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)


稍微运行一会，选择一个指标，调节时间范围为15分钟，即可
看到指标的变化情况

![pic](https://img-blog.csdnimg.cn/20191231061807669.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)



---

### Prometheus：监控与告警：3:指标监控示例

#### 事前准备
- Prometheus可以对各种指标数据进行监控，

首先准备用于提供监控数据的应用程序。
```bash
git clone https://github.com/prometheus/client_golang.git
```

启动Docker容器进行编译
- 启动Docker并将待编译的工程进行卷的映射

```bash
# Automatically remove the container when it exits
# The -it instructs Docker to allocate a pseudo-TTY connected to the container’s stdin; creating an interactive bash shell in the container.
# Bind mount a volume
# CPU shares (relative weight)
docker run --rm -it \
	-v `pwd`/client_golang:/go/src liumiaocn/golang:1.13.5-alpine3.11 sh \
	-c 'cd /go/src/examples/random && go get -d && go build && ls -l random'

docker run --rm -it \
	-v `pwd`/client_golang:/go/src golang:1.13.5-alpine3.11 sh \
	-c 'cd /go/src/examples/random && go get -d && go build && ls -l random'
# go: downloading github.com/prometheus/common v0.7.0
# go: downloading github.com/prometheus/client_model v0.1.0
# go: extracting github.com/prometheus/client_model v0.1.0
# go: downloading github.com/golang/protobuf v1.3.2
# go: extracting github.com/prometheus/common v0.7.0
# go: downloading github.com/matttproud/golang_protobuf_extensions v1.0.1
# go: downloading golang.org/x/sys v0.0.0-20191220142924-d4481acd189f
# go: downloading github.com/cespare/xxhash/v2 v2.1.1
# go: downloading github.com/prometheus/procfs v0.0.8
# go: downloading github.com/beorn7/perks v1.0.1
# go: extracting github.com/golang/protobuf v1.3.2
# go: extracting github.com/cespare/xxhash/v2 v2.1.1
# go: extracting github.com/matttproud/golang_protobuf_extensions v1.0.1
# go: extracting github.com/beorn7/perks v1.0.1
# go: extracting github.com/prometheus/procfs v0.0.8
# go: extracting golang.org/x/sys v0.0.0-20191220142924-d4481acd189f
# go: finding github.com/prometheus/client_model v0.1.0
# go: finding github.com/prometheus/common v0.7.0
# go: finding github.com/beorn7/perks v1.0.1
# go: finding github.com/cespare/xxhash/v2 v2.1.1
# go: finding github.com/golang/protobuf v1.3.2
# go: finding github.com/prometheus/procfs v0.0.8
# go: finding github.com/matttproud/golang_protobuf_extensions v1.0.1
# github.com/prometheus/procfs/internal/util
# /go/pkg/mod/github.com/prometheus/procfs@v0.8.0/internal/util/parse.go:69:15: undefined: os.ReadFile
# /go/pkg/mod/github.com/prometheus/procfs@v0.8.0/internal/util/parse.go:78:15: undefined: os.ReadFile
# /go/pkg/mod/github.com/prometheus/procfs@v0.8.0/internal/util/readfile.go:36:9: undefined: io.ReadAll
# note: module requires Go 1.17

```


#### 启动监控对象进程

- 分别在8080-8082三个端口启动三个服务用于提供Prometheus监控的对象进程。

```bash
$ docker run -p 8080:8080 -d -it \
	-v `pwd`/random:/random \
	--rm alpine /random \
	-listen-address=:8080
# 22da3e4803b8fc7b31b0ebb7b8eac0afc188c62bfc1e1ae58f26ebf56178f3b8

$ docker run -p 8081:8081 -d -it \
	-v `pwd`/random:/random \
	--rm alpine /random \
	-listen-address=:8081
# ed35547ffb865df313236adab20d0c20164f051a45df9f59c93df6e1ddaec4b6

$ docker run -p 8082:8082 -d -it \
	-v `pwd`/random:/random \
	--rm alpine /random \
	-listen-address=:8082
# 4e7da2844b26b67b41ad43b52673fccb95d7631c88906d3dff1f244dff62a43e
```

结果确认：容器状态确认

```bash
$ docker ps
# CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                    NAMES
# 4e7da2844b26        alpine              "/random -listen-add…"   2 seconds ago       Up 1 second         0.0.0.0:8082->8082/tcp   ecstatic_hypatia
# ed35547ffb86        alpine              "/random -listen-add…"   10 seconds ago      Up 9 seconds        0.0.0.0:8081->8081/tcp   blissful_jennings
# 22da3e4803b8        alpine              "/random -listen-add…"   21 seconds ago      Up 20 seconds       0.0.0.0:8080->8080/tcp   nervous_bell
```

结果确认：指标确认

```bash
$ curl http://localhost:8080/metrics 2>/dev/null |wc -l
#  164

$ curl http://localhost:8081/metrics 2>/dev/null |wc -l
#  164

$ curl http://localhost:8082/metrics 2>/dev/null |wc -l
#  164
```


#### 配置Prometheus

- 现在我们在8080-8082三个端口都提供了可供Prometheus进行监控指标数据，将Prometheus进行如下配置：

```yaml
prometheus-random.yml
scrape_configs:
  - job_name: "example-random"
# Override the global default and scrape targets from this job every 5 seconds.
    scrape_interval: 5s
    static_configs:
      - targets: ["192.168.31.242:8080", "192.168.31.242:8081"]
        labels:
          group: "production"

      - targets: ["192.168.31.242:8082"]
        labels:
          group: "canary"
```

配置说明：
* 设定job名称为example-random
* 数据的抓取时间间隔设定为5秒
* 将8080-8082三个监控对象分成两组，
  * 8080和8081一组，组标签名称为production，
  * 8082为一组，组标签名称为canary
* ip（192.168.31.242）请修改为自己的IP，因为本文启动的内容均在容器之中，又没有使用link或者其他方式来使得各个容器之间的相互联通，这里直接使用IP方式使得Prometheus能够访问到这些对象机器。

#### 启动Prometheus服务

```bash

docker run -d -p 9090:9090 \
	-v `pwd`/prometheus-demo.yml:/etc/prometheus/prometheus.yml \
	--name prometheus prom/prometheus

# 使用如下命令启动Prometheus服务
$ docker run -d -p 9090:9090 \
	-v `pwd`/prometheus-random.yml:/etc/prometheus/prometheus.yml \
	--name prometheus prom/prometheus
# 1f9d3831c1d2c85d758cb4ff8af3054ec90a7f7b8f1f356431150ce6822253df

# 确认Promtheus容器状态
$ docker ps |grep prometheus
# 1f9d3831c1d2        prom/prometheus     "/bin/prometheus --c…"   About a minute ago   Up 59 seconds       0.0.0.0:9090->9090/tcp   prometheus
```

#### 结果确认

确认连接状态

Prometheus从`8080-8082`的端口获取监控数据，而这些连接是否正常，可以从如下界面进行确认

![pic](https://img-blog.csdnimg.cn/20200102145254446.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

获取的指标信息

![pic](https://img-blog.csdnimg.cn/2020010214533735.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

查看某一指标的实时状况

Prometheus也具有一点可视化的能力，比如可以直接确认选中的某项指标的一定时间段（缺省1个小时，这里选择为5分钟）的变化情况

![pic](https://img-blog.csdnimg.cn/20200102145520356.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)


---


### Prometheus：监控与告警：4:使用Grafana进行可视化显示


#### 事前准备
- 这里仍然使用前文所创建的random进程提供的指标数据信息，详细设定过程可参看：[link](https://liumiaocn.blog.csdn.net/article/details/103405873)

Grafana准备
- Grafana从2015年10发布的2.5.0版本开始支持对对Prometheus数据源，在本文的示例中将使用6.5.1版本进行可视化显示的演示。

```bash
# 拉取镜像
docker pull grafana/grafana:6.5.1

# 启动Grafana服务
docker run -d -p 3000:3000 \
	--name grafana grafana/grafana:6.5.1
```

设定Prometheus数据源
- 启动Grafana和Prometheus之后，使用如下步骤即可在Grafana中添加Prometheus的数据源。

步骤1: 登录Grafana
- 使用缺省的admin/admin账号登录Grafana，
- 登录之后可以修改缺省密码也可直接跳过  passwd

![pic](https://img-blog.csdnimg.cn/20200102154218450.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

步骤2: 添加Prometheus数据源
- 点击 Add data source进行数据源的添加

![pic](https://img-blog.csdnimg.cn/20200102154412917.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- Grafana支持很多可视化展示的数据源，这里选择Prometheus

![pic](https://img-blog.csdnimg.cn/20200102154525774.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 设定Prometheus的URL，由于本文示例中各个服务均使用单独的容器启动，所以这里直接使用可以访问的IP来进行设定，除了IP的设定之外其余均可使用缺省设定，点击Save & Test按钮确认连接是否正常

![pic](https://img-blog.csdnimg.cn/20200102154830793.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 这样数据源的添加就完成了，后续如果需要进行修改或者删除等操作，可通过下图左侧Configuration菜单中的data sources选项进行操作即可。

![pic](https://img-blog.csdnimg.cn/20200103050507289.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

设定可视化显示的仪表盘

- 在数据源配置和连通性测试成功之后，即可在Grafana中创建定制的仪表盘了。点击下图中的New Dashboard按钮或者左侧Dashboard菜单的Manage选项即可进行仪表盘的创建。

![pic](https://img-blog.csdnimg.cn/20200103050807417.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 在接下来的页面中选择Choose Visualization

![pic](https://img-blog.csdnimg.cn/20200103051111792.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 点击Panel Title下拉菜单的Edit选项即可进行编辑

![pic](https://img-blog.csdnimg.cn/20200103051412644.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 另外选择左侧的General按钮，则可对Panel Title的标题进行修改，比如此处修改为Random-Metrics-Info

![pic](https://img-blog.csdnimg.cn/20200103051700741.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 实际上这时并没有数据，所以这里点击左下侧四个按钮的第一个，是设定可视化来源的数据信息的。可以看到在Metrics页面可以看到监控的各项指标，这里选取其中一项进行显示，同时将时间范围设定为15分钟（这里只是为了方便示例结果的演示）

![pic](https://img-blog.csdnimg.cn/20200103052115278.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 点击保存按钮，将此Dashboard设定名称并点击Save按钮，缺省会保存到General目录下（也可以点击Gneral的下拉菜单创建新的保存目录）

![pic](https://img-blog.csdnimg.cn/20200103052441545.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 保存之后结果如下所示

![pic](https://img-blog.csdnimg.cn/20200103052610779.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 这时就可以随意拖拽进行可视化显示的调节了

![pic](https://img-blog.csdnimg.cn/20200103052723578.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 然后使用相同的步骤再添加两个指标进行设定和显示，

![pic](https://img-blog.csdnimg.cn/20200103053339838.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

![pic](https://img-blog.csdnimg.cn/20200103053349762.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 此时自定义的仪表盘已经变成了这样

![pic](https://img-blog.csdnimg.cn/2020010305343729.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

- 可以进行随意拖拽和调整大小与位置，比如可以调整成这样

![pic](https://img-blog.csdnimg.cn/20200103053609231.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)


---


### Prometheus：监控与告警：5:在Kubernetes上部署

Prometheus安装方法
- 在Kubernetes上直接部署Prometheus也非常简单，
- 使用ConfigMap管理配置文件，然后使用卷方式挂载，
- 然后创建Deployment和Service即可使用了。



事前准备

```bash
$ kubectl get node -o wide
# NAME              STATUS   ROLES    AGE   VERSION   INTERNAL-IP       EXTERNAL-IP   OS-IMAGE                KERNEL-VERSION          CONTAINER-RUNTIME
# 192.168.163.131   Ready    <none>   20h   v1.17.0   192.168.163.131   <none>        CentOS Linux 7 (Core)   3.10.0-957.el7.x86_64   docker://18.9.7
```

步骤1: 创建ConfigMap


```bash
# 创建Prometheus使用的ConfigMap配置，所使用的yaml文件如下所示
# prometheus.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-configmap
  namespace: default
data:
  prometheus.yml: |
    global:
      scrape_interval:     10s
      evaluation_interval: 10s
    scrape_configs:
      - job_name: 'prometheus'
        static_configs:
        - targets: ['localhost:9090']


# 创建并确认ConfigMap配置
$ kubectl create -f prometheus.yml
# configmap/prometheus-configmap created

$ kubectl get cm
# NAME                   DATA   AGE
# prometheus-configmap   1      4s
$ kubectl describe cm prometheus-configmap
# Name:         prometheus-configmap
# Namespace:    default
# Labels:       <none>
# Annotations:  <none>

# Data
# ====
# prometheus.yml:
# ----
# global:
#   scrape_interval:     10s
#   evaluation_interval: 10s
# scrape_configs:
#   - job_name: 'prometheus'
#     static_configs:
#     - targets: ['localhost:9090']

# Events:  <none>
```



步骤2: 创建Service与Deployment
- 首先准备如下的yaml文件，包含了Prometheus所需的Deployment和Service的配置

```yml
prometheus-deployment.yml
    ---
    apiVersion: v1
    kind: "Service"
    metadata:
      name: prometheus
      labels:
        name: prometheus
    spec:
      ports:
      - name: prometheus
        protocol: TCP
        port: 9090
        targetPort: 9090
      selector:
        app: prometheus
      type: NodePort
    ...
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      labels:
        name: prometheus
      name: prometheus
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: prometheus
      template:
        metadata:
          labels:
            app: prometheus
        spec:
          containers:
          - name: prometheus
            image: prom/prometheus:v2.15.1
            command:
            - "/bin/prometheus"
            args:
            - "--config.file=/etc/prometheus/prometheus.yml"
            ports:
            - containerPort: 9090
              protocol: TCP
            volumeMounts:
            - mountPath: "/etc/prometheus"
              name: prometheus-configmap
          volumes:
          - name: prometheus-configmap
            configMap:
              name: prometheus-configmap
```

```bash
# 创建Service与Deployment
$ kubectl create -f prometheus-deployment.yml
# service/prometheus created
# deployment.apps/prometheus created


# 确认Service信息
$ kubectl get service -o wide
# NAME         TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)          AGE   SELECTOR
# kubernetes   ClusterIP   10.254.0.1       <none>        443/TCP          20h   <none>
# prometheus   NodePort    10.254.229.211   <none>        9090:30944/TCP   7s    app=prometheus


# 确认Pod信息
$ kubectl get pods
# NAME                         READY   STATUS    RESTARTS   AGE
# prometheus-fcd87fbf4-ljzrb   1/1     Running   0          13s
```


步骤3: 结果确认
- 在30944端口即可确认刚刚部署的Prometheus的运行状况

![pic](https://img-blog.csdnimg.cn/20200104063822495.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)


---


### Prometheus：监控与告警：6: Exporter概要介绍

![pic](https://img-blog.csdnimg.cn/20200113145037917.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9saXVtaWFvY24uYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

Exporter
- 为Prometheus提供监控数据源的应用都可以被成为Exporter，
- 比如Node Exporter则用来提供节点相关的资源使用状况，而Prometheus从这些不同的Exporter中获取监控数据，然后可以在诸如Grafana这样的可视化工具中进行结果的显示。

Exporter的类型
* Exporter根据来源可以分为：社区提供的Exporter和自定义的Exporter两种
* Exporter根据支持方式可以分为：很多软件现在已经内嵌支持Prometheus，比如kubernetes或者etcd，简单来说这种类型的软件中不需要单独的Exporter用于提供给Prometheus的监控数据的功能，这是其本身的功能特性之一。当然更多的情况则是通过独立运行的Exporter来进行，比如Node Exporter，操作系统本身由于不像kubernetes那样提供对于Prometheus的支持，所以需要单独运行Node Exporter用于提供节点自身的信息给Prometheus进行监控。

社区常见的Exporter

数据库
常见的主流数据库几乎逗留相应的Exporter，详细如下所示：
* MongoDB exporter
* MSSQL server exporter
* MySQL server exporter (official)
* OpenTSDB Exporter
* Oracle DB Exporter
* PostgreSQL exporter
* Redis exporter
* ElasticSearch exporter
* RethinkDB exporter
* Consul exporter (official)

消息队列
* Kafka exporter
* IBM MQ exporter
* RabbitMQ exporter
* RocketMQ exporter
* NSQ exporter
* Gearman exporter


存储
* Ceph exporter
* Gluster exporter
* Hadoop HDFS FSImage exporter


硬件相关
* Node/system metrics exporter (official)
* Dell Hardware OMSA exporter
* IoT Edison exporter
* IBM Z HMC exporter
* NVIDIA GPU exporter


问题追踪与持续集成
* Bamboo exporter
* Bitbucket exporter
* Confluence exporter
* Jenkins exporter
* JIRA exporter

HTTP服务
* Apache exporter
* HAProxy exporter (official)
* Nginx metric library
* Nginx VTS exporter
* Passenger exporter
* Squid exporter
* Tinyproxy exporter
* Varnish exporter
* WebDriver exporter

API服务
* AWS ECS exporter
* AWS Health exporter
* AWS SQS exporter
* Cloudflare exporter
* DigitalOcean exporter
* Docker Cloud exporter
* Docker Hub exporter
* GitHub exporter
* InstaClustr exporter
* Mozilla Observatory exporter
* OpenWeatherMap exporter
* Pagespeed exporter
* Rancher exporter
* Speedtest exporter
* Tankerkönig API Exporter

日志
* Fluentd exporter
* Google’s mtail log data extractor
* Grok exporter

监控系统
* Akamai Cloudmonitor exporter
* Alibaba Cloudmonitor exporter
* AWS CloudWatch exporter (official)
* Azure Monitor exporter
* Cloud Foundry Firehose exporter
* Collectd exporter (official)
* Google Stackdriver exporter
* Graphite exporter (official)
* Huawei Cloudeye exporter
* InfluxDB exporter (official)
* JavaMelody exporter
* JMX exporter (official)
* Nagios / Naemon exporter
* Sensu exporter
* SNMP exporter (official)
* TencentCloud monitor exporter
* ThousandEyes exporter


其他
* BIND exporter
* Bitcoind exporter
* cAdvisor
* Dnsmasq exporter
* Ethereum Client exporter
* JFrog Artifactory Exporter
* JMeter plugin
* Kibana Exporter
* kube-state-metrics
* OpenStack exporter
* PowerDNS exporter
* Script exporter
* SMTP/Maildir MDA blackbox prober
* WireGuard exporter
* Xen exporter


使用方式
- Prometheus Server提供PromQL查询语言能力、负责数据的采集和存储等主要功能，
- 而数据的采集主要通过周期性的从Exporter所暴露出来的HTTP服务地址（一般是/metrics）来获取监控数据。
- 而Exporter在实际运行的时候根据其支持的方式也会分为：
  * 独立运行的Exporter应用，通过HTTP服务地址提供相应的监控数据（比如Node Exporter）
  * 内置在监控目标中，通过HTTP服务地址提供相应的监控数据（比如kubernetes）





---

## monitoring with Prometheus


## client


Package prometheus is the core instrumentation package.
- It provides metrics primitives to instrument code for monitoring.
- It also offers a registry for metrics.
- Sub-packages allow to expose the registered metrics via HTTP (package promhttp) or push them to a Pushgateway (package push).
- There is also a sub-package promauto, which provides metrics constructors with automatic registration.
- All exported functions and methods are safe to be used concurrently unless specified otherwise.

### Instrumenting applications


- The [prometheus directory](https://github.com/prometheus/client_golang/tree/main/prometheus) contains the instrumentation library.
- See the [guide on the Prometheus website](https://prometheus.io/docs/guides/go-application/) to learn more about instrumenting applications.
- For comprehensive API documentation, see the [GoDoc](https://godoc.org/github.com/prometheus/client_golang) for Prometheus' various Go libraries.


In this guide, you created two sample Go applications that expose metrics to Prometheus
- one that exposes only the default Go metrics
- and one that also exposes a custom Prometheus counter and configured a Prometheus instance to scrape metrics from those applications.

This documentation is [open-source](https://github.com/prometheus/docs#contributing-changes). Please help improve it by filing issues or pull requests.

#### Installation
- install the `prometheus`, `promauto`, and `promhttp` libraries necessary for the guide using [`go get`](https://golang.org/doc/articles/go_command.html):

```bash
go get github.com/prometheus/client_golang/prometheus
go get github.com/prometheus/client_golang/prometheus/promauto
go get github.com/prometheus/client_golang/prometheus/promhttp
```

#### How Go exposition works
- To expose Prometheus metrics in a Go application, you need to provide a `/metrics` HTTP endpoint.
- You can use the [`prometheus/promhttp`](https://godoc.org/github.com/prometheus/client_golang/prometheus/promhttp) library's HTTP [`Handler`](https://godoc.org/github.com/prometheus/client_golang/prometheus/promhttp#Handler) as the handler function.

```go
// This minimal application, for example, would expose the default metrics for Go applications via `http://localhost:2112/metrics`:
package main
import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)
func main() {
	http.Handle("/metrics", promhttp.Handler())
	http.ListenAndServe(":2112", nil)
}

// To start the application:
go run main.go

// To access the metrics:
curl http://localhost:2112/metrics
```

#### Add own metrics

- The application [above](#how-go-exposition-works) exposes only the default Go metrics.
- You can also register your own custom application-specific metrics.


```go
// This example application exposes a `myapp_processed_ops_total` [counter](/docs/concepts/metric_types/#counter) that counts the number of operations that have been processed thus far.
// Every 2 seconds, the counter is incremented by one.
package main

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func recordMetrics() {
	go func() {
		for {
			opsProcessed.Inc()
			time.Sleep(2 * time.Second)
		}
	}()
}

var (
	opsProcessed = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "myapp_processed_ops_total",
			Help: "The total number of processed events",
		})
)

func main() {
	recordMetrics()

	http.Handle("/metrics", promhttp.Handler())
	http.ListenAndServe(":2112", nil)
}


// To start the application:
go run main.go

// To access the metrics:
curl http://localhost:2112/metrics
```

In the metrics output, you'll see the help text, type information, and current value of the `myapp_processed_ops_total` counter:

```go
// # HELP myapp_processed_ops_total The total number of processed events
// # TYPE myapp_processed_ops_total counter
// myapp_processed_ops_total 5
```


[configure](/docs/prometheus/latest/configuration/configuration/#scrape_config) a locally running Prometheus instance to scrape metrics from the application. Here's an example `prometheus.yml` configuration:

```yaml
scrape_configs:
    - job_name: myapp
      scrape_interval: 10s
      static_configs:
      - targets:
        - localhost:2112
```

#### Other Go client features

In this guide we covered just a small handful of features available in the Prometheus Go client libraries. You can also expose other metrics types, such as
- [gauges](https://godoc.org/github.com/prometheus/client_golang/prometheus#Gauge)
- and [histograms](https://godoc.org/github.com/prometheus/client_golang/prometheus#Histogram),
- [non-global registries](https://godoc.org/github.com/prometheus/client_golang/prometheus#Registry),
- functions for [pushing metrics](https://godoc.org/github.com/prometheus/client_golang/prometheus/push) to Prometheus [PushGateways](/docs/instrumenting/pushing/),
- bridging Prometheus and [Graphite](https://godoc.org/github.com/prometheus/client_golang/prometheus/graphite), and more.




The [examples directory](https://github.com/prometheus/client_golang/tree/main/examples) contains simple examples of instrumented code.

### Client for the Prometheus HTTP API


The api/prometheus directory contains the client for the Prometheus HTTP API. It allows you to write Go applications that query time series data from a Prometheus server. It is still in alpha stage.
