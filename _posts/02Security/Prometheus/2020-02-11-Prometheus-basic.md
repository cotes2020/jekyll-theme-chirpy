---
title: Monitor - Prometheus basic
date: 2020-02-11 11:11:11 -0400
categories: [02Security, Prometheus]
tags: [02Security, Prometheus]
math: true
image:
---

- [Prometheus basic](#prometheus-basic)
    - [主要的特色](#主要的特色)
  - [kafka with Prometheus](#kafka-with-prometheus)
  - [Prometheus架构剖析](#prometheus架构剖析)
    - [pull vs push](#pull-vs-push)
    - [Job/Exporter](#jobexporter)
      - [Telegraf](#telegraf)
    - [Pushgateway](#pushgateway)
    - [Service Discovery 服务发现](#service-discovery-服务发现)
    - [Prometheus Server](#prometheus-server)
  - [Dashboard](#dashboard)
  - [Alertmanager](#alertmanager)
  - [数据模型](#数据模型)
  - [metric](#metric)


---


# Prometheus basic

> Prometheus和Kubernetes不仅在使用过程中紧密相关,而且在历史上也有很深的渊源.
> Google公司里曾经有两款系统——Borg系统和它的监控Borgmon系统.Borg系统是Google内部用来管理来自不同应用、不同作业的集群的管理器,每个集群都会拥有数万台服务器及上万个作业；Borgmon系统则是与Borg系统配套的监控系统.
> Borg系统和Borgmon系统都没有开源,但是目前开源的Kubernetes、Prometheus的理念都是对它们的理念的传承.

> Prometheus官网上的自述是:“From metrics to insight. Power your metrics and alerting with a leading open-source monitoring solution.”
> 从指标到洞察力,Prometheus通过领先的开源监控解决方案为用户的指标和告警提供强大的支持.




- a timeseries database that scrapes targets and stores metrics.
  - 一个时序数据库,又是一个监控系统,更是一套完备的监控生态解决方案.
  - 作为时序数据库,在2020年2月的排名中,Prometheus已经跃居到第三名
  - 超越了老牌的时序数据库OpenTSDB、Graphite、RRDtool、KairosDB等

- Recorded metrics can be queried and a multitude of operators and functions are provided for queries.

- Prometheus' model stores all recorded values in the database, in contrast with systems such as Graphite and RRD that store data in a custom —usually lower- resolution that degrades over time.

- This permits fine grained results in queries at the expense of storage.
  - Although Prometheus' storage engine is very efficient, it is better to keep the metrics retention period shorter rathen than longer.
  - Typical values span from 15 days to a few months.


Prometheus' model scrapes data from exporters. This means that Prometheus should have a list of targets to scrape. This list may be set manually or automatically via supported backends —such as consul, zookeeper and kubernetes.



### 主要的特色

- 一站式监控告警平台,依赖少,功能齐全.
- 支持对云或容器的监控,其他系统主要对主机监控.
- 数据查询语句表现力更强大,内置更强大的统计函数.
- 在数据存储扩展性以及持久性上没有 InfluxDB,OpenTSDB,Sensu 好.


**什么时候用它合适**
- Prometheus可以很好地记录任何纯数字时间序列.
  - 既适合以机器为中心的监视,也适合高度动态的面向服务的体系结构的监视.
  - 在微服务的世界中,它对多维数据收集和查询的支持是一个特别的优势.
- Prometheus是为可靠性而设计的
  - 在服务宕机的时候,你可以快速诊断问题.
  - 每台Prometheus服务器都是独立的,不依赖于网络存储或其他远程服务.

**什么时候用它不合适**
- Prometheus的值的可靠性.你总是可以查看有关系统的统计信息,即使在出现故障的情况下也是如此.
- 如果你需要100%的准确性,例如按请求计费,Prometheus不是一个好的选择,因为收集的数据可能不够详细和完整.在这种情况下,最好使用其他系统来收集和分析用于计费的数据,并使用Prometheus来完成剩下的监视工作.





与Nagios、Zabbix、Ganglia、Open-Falcon等很多监控系统相比,Prometheus最主要的特色有4个:
- 通过PromQL实现`多维度数据模型的灵活查询`.
- 定义了`开放指标数据的标准,自定义探针`(如Exporter等),编写简单方便.
- PushGateway组件让这款监控系统`可以接收监控数据`.
- 提供了`VM和容器化的版本`.



除了上述4种特色之外,Prometheus还有如下特点:
- Go语言编写,拥抱云原生.
- 采用拉模式为主、推模式为辅的方式采集数据.
- 二进制文件直接启动,也支持容器化部署镜像.
- 支持多种语言的客户端
  - 如Java、JMX、Python、Go、Ruby、.NET、Node.js等语言.
- 支持本地和第三方远程存储,单机性能强劲,可以处理上千target及每秒百万级时间序列.
- 高效的存储.
  - 平均一个采样数据占3.5B左右,共320万个时间序列,每30秒采样一次,如此持续运行60天,占用磁盘空间大约为228GB(有一定富余量,部分要占磁盘空间的项目未在这里列出).
- 可扩展.
  - 可以在每个数据中心或由每个团队运行独立Prometheus Server.
  - 也可以使用联邦集群让多个Prometheus实例产生一个逻辑集群,当单实例Prometheus Server处理的任务量过大时,通过使用功能分区(sharding)+联邦集群(federation)对其进行扩展.
- 出色的可视化功能.
  - Prometheus拥有多种可视化的模式,比如内置表达式浏览器、Grafana集成和控制台模板语言.
  - 它还提供了HTTP查询接口,方便结合其他GUI组件或者脚本展示数据.
- 精确告警.
  - Prometheus基于灵活的PromQL语句可以进行告警设置、预测等,
  - 另外它还提供了分组、抑制、静默等功能防止告警风暴.
- 支持静态文件配置和动态发现等自动发现机制
  - 目前已经支持了Kubernetes、etcd、Consul等多种服务发现机制,
  - 这样可以大大减少容器发布过程中手动配置的工作量.
- 开放性.
  - Prometheus的client library的输出格式不仅支持Prometheus的格式化数据,还可以在不使用Prometheus的情况下输出支持其他监控系统(比如Graphite)的格式化数据.


Prometheus 也存在一些局限性,主要包括如下方面:
- 主要针对性能和可用性监控
  - 不适用于针对日志(Log)、事件(Event)、调用链(Tracing)等的监控.
- 关注的是近期发生的事情,而不是跟踪数周或数月的数据.
  - 因为大多数监控查询及告警都针对的是最近(通常不到一天)的数据.
  - 监控数据默认保留15天.
- 本地存储有限,存储大量的历史数据需要对接第三方远程存储.
- 采用联邦集群的方式,并没有提供统一的全局视图.
- 监控数据并没有对单位进行定义.
- 对数据的统计无法做到100%准确,如订单、支付、计量计费等精确数据监控场景.
- 默认是拉模型,建议合理规划网络,尽量不要转发.

---

## kafka with Prometheus


- The Kafka ecosystem is based on the JVM and as such exports metrics via JMX.

- **Prometheus** expects its own format for metrics and thus provides small applications called `exporters` that can translate metrics from various software.

- **jmx_exporter** is such an application, that converts JMX metrics to the Prometheus format.
  - It comes in two flavors: server (standalone) and agent.

    - In agent mode, it runs as a `java agent` within the application to be monitored.
      - The suggested mode for most applications. easier to setup and can provide operational metrics as well (CPU usage, memory usage, open file descriptors, JVM   statistics, etc).

    - In server mode, it runs as a `separate application` that `connects` to the monitored application via JMX, `reads` the metrics and then `serves` them in Prometheus format.

      - In special cases such as the brokers we suggest the server mode.
      - under load (hundreds or thousands of topics, clients, etc) the brokers can expose tens or even hundreds of thousands of metrics,
      - we have identified a few cases where the jmx_exporter agent can’t keep up and may cause trouble to the broker as well.
      - The jmx_exporter server as a standalone application will not affect the broker. Jmx_exporter server instances should be co-hosted when possible with the application they monitor, especially for software such as the brokers that expose too many metrics.


---

## Prometheus架构剖析

![91e28b3a8475155e8748fbc91b3998c2](https://i.imgur.com/nTSKO7p.png)


![Screen Shot 2022-09-07 at 11.06.10](https://i.imgur.com/OOAu2Em.png)

**6个核心模块构成**
- Prometheus Server
  - 主要用于抓取数据和存储时序数据,
  - 另外还提供查询和 Alert Rule 配置管理.
- client libraries
  - 用于对接 Prometheus Server, 可以查询和上报数据.
- Pushgateway
  - 用于批量,短期的监控数据的汇总节点,主要用于业务数据汇报等.
- Job/Exporter
  - 汇报数据
  - 例如汇报机器数据的 node_exporter,
  - 汇报 MongoDB 信息的 MongoDB exporter 等等.
- Service Discovery、
- Alertmanager
  - 用于告警通知管理
- Dashboard


**架构**
- Prometheus通过服务发现机制发现target
  - Prometheus server 定期从 `静态配置的 targets` 或者 `服务发现的 targets` 拉取数据
  - targets:
    - 可以是长时间执行的Job,
    - 也可以是短时间执行的Job,
    - 还可以是通过Exporter监控的第三方应用程序.

- 被抓取的数据会存储起来,
  - 当新拉取的数据大于配置内存缓存区的时候,
    - Prometheus 会将数据持久化到磁盘
    - 如果使用 remote storage 将持久化到云端

  - Prometheus 可以配置 rules,然后通过 `PromQL语句` 定时查询数据

    - 在仪表盘等可视化系统中供查询,
      - 可以使用 API, Prometheus Console 或者 Grafana 查询和聚合数据.

    - 当条件触发的时候向 `Alertmanager` 发送告警信息,
      - Alertmanager 收到警告的时候,可以根据配置,聚合,去重,降噪,最后发送警告.
      - 告警会通过页面、电子邮件、钉钉信息或者其他形式呈现.

Prometheus不仅是一款时间序列数据库,在整个生态上还是一套完整的监控系统.
- 对于时间序列数据库,在进行技术选型的时候,往往需要从宽列模型存储、类SQL查询支持、水平扩容、读写分离、高性能等角度进行分析.
- 而监控系统的架构,除了在选型时需要考虑的因素之外,往往还需要考虑通过减少组件、服务来降低成本和复杂性以及水平扩容等因素.


很多企业自己研发的监控系统中往往会
- 使用消息队列Kafka和Metrics parser、Metrics process server等Metrics解析处理模块,
- 再辅以Spark等流式处理方式.
- 应用程序将Metric推到消息队列(如Kafaka),然后经过Exposer中转,再被Prometheus拉取.
- 之所以会产生这种方案,是因为考虑到有历史包袱、复用现有组件、通过MQ(消息队列)来提高扩展性等因素.
- 这个方案会有如下几个问题:
  - 增加了查询组件,比如基础的sum、count、average函数都需要额外进行计算.
    - 这一方面多了一层依赖,在查询模块连接失败的情况下会多提供一层故障风险；
    - 另一方面,很多基本的查询功能的实现都需要消耗资源.而在Prometheus的架构里,上述这些功能都是得到支持的.
  - 抓取时间可能会不同步,延迟的数据将会被标记为陈旧数据.
    - 如果通过添加时间戳来标识数据,就会失去对陈旧数据的处理逻辑.
  - Prometheus适用于监控大量小目标的场景,而不是监控一个大目标,
    - 如果将所有数据都放在Exposer中,那么Prometheus的单个Job拉取就会成为CPU的瓶颈.
    - 这个架构设计和Pushgateway类似,因此如果不是特别必要的场景,官方都不建议使用. 缺少服务发现和拉取控制机制
  - Prometheus只能识别Exposer模块,不知道具体是哪些target,也不知道每个target的UP时间,所以无法使用Scrape_*等指标做查询,也无法用scrape_limit做限制.

对于上述这些重度依赖,可以考虑将其优化掉,而Prometheus这种采用以拉模式为主的架构,在这方面的实现是一个很好的参考方向.

同理,很多企业的监控系统对于cmdb具有强依赖,通过Prometheus这种架构也可以消除标签对cmdb的依赖.


---


### pull vs push

pull方式
- Prometheus采集数据是用的 pull 拉模型
- 通过HTTP协议去采集指标，
  - 只要应用系统能够提供HTTP接口就可以接入监控系统，
  - 相比于私有协议或二进制协议来说开发、简单。

push方式
- 对于定时任务这种短周期的指标采集，如果采用pull模式，可能造成任务结束了，Prometheus还没有来得及采集，这个时候可以使用加一个中转层，客户端推数据到Push Gateway缓存一下，由Prometheus从push gateway pull指标过来。
- 需要额外搭建Push Gateway，同时需要新增job去从gateway采数据




---

### Job/Exporter

`Job/Exporter`属于Prometheus target,是Prometheus监控的对象.

**Job**
- 分为长时间执行和短时间执行两种.
  - 对于长时间执行的Job,可以使用Prometheus Client集成进行监控；
  - 对于短时间执行的Job,可以将监控数据推送到Pushgateway中缓存.

**Exporter**
- Prometheus收录的`Exporter`有上千种,它可以用于第三方系统的监控.
- Exporter的机制是将第三方系统的监控数据按照Prometheus的格式暴露出来,没有Exporter的第三方系统可以自己定制Exporter
- Prometheus是一个白盒监视系统,它会对应用程序内部公开的指标进行采集.
- blackbox_exporter
  - 假如用户想从外部检查,这就会涉及黑盒监控,Prometheus中常用的黑盒Exporter就是blackbox_exporter.
  - blackbox_exporter 包括一些现成的模块,例如HTTP、TCP、POP3S、IRC和ICMP.
  - blackbox.yml 可以扩展其中的配置,以添加其他模块来满足用户的需求.
  - blackbox_exporter一个令人满意的功能是,如果模块使用TLS/SSL,则Exporter将在证书链到期时自动公开,这样可以很容易地对即将到期的SSL证书发出告警.

#### Telegraf
- Exporter种类繁多,每个Exporter又都是独立的,每个组件各司其职.但是Exporter越多,维护压力越大,尤其是内部自行开发的Agent等工具需要大量的人力来完成资源控制、特性添加、版本升级等工作,可以考虑替换为Influx Data公司开源的Telegraf统一进行管理.
- Telegraf是一个用Golang编写的用于数据收集的开源Agent,其基于插件驱动.Telegraf提供的输入和输出插件非常丰富,当用户有特殊需求时,也可以自行编写插件(需要重新编译),它在Influx Data架构中的位置如图所示.

![5058438d75a9a69e2ddfc760146b018a](https://i.imgur.com/VYaFZix.png)


- Telegraf就是Influx Data公司的时间序列平台**TICK**(一种高性能时序中台)技术栈中的“T”
- 主要用于收集时间序列型数据,比如服务器CPU指标、内存指标、各种IoT设备产生的数据等.
- Telegraf支持各种类型Exporter的集成,可以实现Exporter的多合一.
- 还有一种思路就是通过主进程拉起多个Exporter进程,仍然可以跟着社区版本进行更新.

- Telegraf的CPU和内存使用率极低,支持几乎所有的集成监控和丰富的社区集成可视化,如Linux、Redis、Apache、StatsD、Java/Jolokia、Cassandra、MySQL等.
- 由于Prometheus和InfluxDB都是时间序列存储监控系统,可以变通地将Telegraf对接到Prometheus中.在实际POC环境验证中,使用Telegraf集成Prometheus比单独使用Prometheus会拥有更低的内存使用率和CPU使用率.



### Pushgateway

- Prometheus是`拉模式`为主的监控系统,它的`推模式`就是通过Pushgateway组件实现的.
- Pushgateway是支持临时性Job主动推送指标的中间网关,它本质上是一种用于监控Prometheus服务器无法抓取的资源的解决方案.
- 它也是用Go语言编写的,在Apache 2.0许可证下开源.

- Pushgateway作为一个独立的服务,位于被采集监控指标的应用程序和Prometheus服务器之间.
  - 应用程序主动推送指标到Pushgateway,
  - Pushgateway接收指标,
  - 然后Pushgateway也作为target被Prometheus服务器抓取.它
- 的使用场景主要有如下几种:
  - 临时/短作业
  - 批处理作业

- 应用程序与Prometheus服务器之间有网络隔离,
  - 如安全性(防火墙)、
  - 连接性(不在一个网段,服务器或应用程序仅允许特定端口或路径访问).
- Pushgateway与网关类似,在Prometheus中被建议作为临时性解决方案,主要用于监控不太方便访问到的资源.
- 它会丢失很多Prometheus服务器提供的功能,比如UP指标和指标过期时进行实例状态监控.


Pushgateway的常见问题

- 它存在单点故障问题.如果Pushgateway从许多不同的来源收集指标时宕机,用户将失去对所有这些来源的监控,可能会触发许多不必要的告警.

- Pushgateway不会自动删除推送给它的任何指标数据.
  - 因此,必须使用Pushgateway的API从推送网关中删除过期的指标.

```bash
curl -X DELETE http://pushgateway.example.org:9091/metrics/job/some_job/instance/ some_instance
```

- 防火墙和NAT问题.推荐做法是将Prometheus移到防火墙后面,让Prometheus更加接近采集的目标.

- 注意,Pushgateway会丧失Prometheus通过UP监控指标检查实例健康状况的功能,此时Prometheus对应的拉状态的UP指标只是针对单Pushgateway服务的.



---


### Service Discovery 服务发现

- 作为下一代监控系统的首选解决方案,Prometheus通过`服务发现`机制对云以及容器环境下的监控场景提供了完善的支持.

- 除了支持`文件的服务发现`(Prometheus会周期性地从文件中读取最新的target信息)外,Prometheus还支持多种常见的`服务发现组件`,
  - 如`Kubernetes、DNS、Zookeeper、Azure、EC2和GCE`等.
  - 例如,Prometheus可以使用Kubernetes的API获取容器信息的变化(如容器的创建和删除)来动态更新监控对象.

- 对于支持文件的服务发现,实践场景下可以衍生为与自动化配置管理工具(Ansible、Cron Job、Puppet、SaltStack等)结合使用.

- 通过服务发现的方式,管理员可以在不重启Prometheus服务的情况下动态发现需要监控的target实例信息.

- 服务发现中有一个高级操作,就是 `Relabeling` 机制.
  - Relabeling机制会从Prometheus包含的target实例中获取默认的元标签信息,从而对不同开发环境(测试、预发布、线上)、不同业务团队、不同组织等按照某些规则(比如标签)从服务发现注册中心返回的target实例中有选择性地采集某些Exporter实例的监控数据.


- 相对于直接使用文件配置,在云环境以及容器环境下更多的监控对象都是动态的.
- 实际场景下,Prometheus作为下一代监控解决方案,更适合云及容器环境下的监控需求,在服务发现过程中也有很多工作(如Relabeling机制)可以加持.

---

### Prometheus Server

- Prometheus服务器是Prometheus最核心的模块.
- 它主要包含抓取、存储和查询这3个功能

![a9185d697de249cce58df1299fc5e7e1](https://i.imgur.com/0PU96mT.png)


<font color=red> 抓取 </font>:

- Prometheus Server通过服务发现组件,周期性地从上面介绍的`Job、Exporter、Pushgateway`这3个组件中通过 `HTTP轮询` 的形式拉取监控指标数据.

<font color=red> 存储 </font>:

- 抓取到的监控数据通过一定的规则清理和数据整理
  - 抓取前使用服务发现提供的 `relabel_configs` 方法
  - 抓取后使用作业内的 `metrics_relabel_configs` 方法
- 会把得到的结果存储到新的时间序列中进行持久化.
- 多年来,存储模块经历了多次重新设计,Prometheus 2.0版的存储系统是第三次迭代.
  - 该存储系统每秒可以处理数百万个样品的摄入,使得使用一台Prometheus服务器监控数千台机器成为可能.
  - 使用的压缩算法可以在真实数据上实现每个样本1.3B.建议使用SSD,但不是严格要求.

- Prometheus的存储分为本地存储和远程存储.
  - **本地存储**:
    - 会直接保留到本地磁盘
    - 性能上建议使用SSD且不要保存超过一个月的数据.
    - 任何版本的Prometheus都不支持NFS.一些实际生产案例告诉我们,Prometheus存储文件如果使用NFS,则有损坏或丢失历史数据的可能.
  - **远程存储**:
    - 适用于存储大量的监控数据.
    - Prometheus支持的远程存储包括`OpenTSDB、InfluxDB、Elasticsearch、Graphite、CrateDB、Kakfa、PostgreSQL、TimescaleDB、TiKV`等.
    - 远程存储需要配合中间层的适配器进行转换,主要涉及Prometheus中的 `remote_write` 和 `remote_read` 接口.
    - 在实际生产中,远程存储会出现各种各样的问题,需要不断地进行优化、压测、架构改造甚至重写上传数据逻辑的模块等工作.

<font color=red> 查询 </font>:

- Prometheus持久化数据以后,客户端就可以通过 `PromQL` 语句对数据进行查询了.

---

## Dashboard

- Web UI、Grafana、API client可以统一理解为Prometheus Dashboard.

- Prometheus服务器除了内置查询语言PromQL以外,还支持表达式浏览器及表达式浏览器上的数据图形界面.

- 实际工作中使用Grafana等作为前端展示界面,用户也可以直接使用Client向Prometheus Server发送请求以获取数据.

---

## Alertmanager

- Alertmanager是独立于Prometheus的一个告警组件,需要单独安装部署.

- Prometheus可以将多个Alertmanager配置为一个集群,通过服务发现动态发现告警集群中节点的上下线从而`避免单点问题`,Alertmanager也支持集群内多个实例之间的通信,


![cbf3cc219ca9a431076763bdcdeb7978](https://i.imgur.com/Fcv3sIi.png)

- Alertmanager接收Prometheus推送过来的告警,用于管理、整合和分发告警到不同的目的地.

- Alertmanager
  - 提供了多种内置的第三方告警通知方式,
  - 同时还提供了对Webhook通知的支持,通过Webhook用户可以完成对告警的更多个性化的扩展.
- Alertmanager除了提供基本的告警通知能力以外,还提供了如分组、抑制以及静默等告警特性


---

## 数据模型

Prometheus基本上将所有数据存储为时间序列：属于同一指标和同一组标记维度的时间戳值流.除了存储时间序列外,Prometheus还可以根据查询结果生成临时派生的时间序列.

时间序列 time series: streams of timestamped values belonging to the same metric and the same set of labeled dimensions

**Metric names and labels**

- Every time series is uniquely identified by its metric name and optional key-value pairs called labels.
  - 每个时间序列都由其指标名称和称为标签的可选键值对唯一标识)

- 指标名称
  - 指定要度量的系统的一般特性(例如, `http_requests_total` 表示接收的HTTP请求的总数).
  - 它可能包含ASCII字母和数字,以及下划线和冒号.
  - 它必须匹配正则表达式`[a-zA-Z_:][a-zA-Z0-9_:]*`
- 标签名称
  - 可以包含ASCII字母、数字和下划线.
  - 它们必须匹配正则表达式`[a-zA-Z_][a-zA-Z0-9_]*`.
  - 以__开头的标签名称保留内部使用.

- 标签值可以包含任何Unicode字符.



**Sample(样本)**

- 样本构成实际的时间序列数据.

- 每个样本包括：
  - a float64 value
  - a millisecond-precision timestamp


**notation(记法)**
- 给定一个度量名称和一组标签,时间序列通常使用以下符号标识：
- `<metric name>{<label name>=<label value>,...}`
- 例如
  - 有这样一个时间序列
  - 指标名称是api_http_requests_total
  - 有两个标签method="POST"和handler="/messages"
  - 那么这个时间序列可以这样写：
  - `api_http_requests_total{method="POST", handler="/messages"}`


**metric types(指标类型)**
- Counter(计数器)
  - 计数器是一个累积指标,它表示一个单调递增的计数器,其值只能在重启时递增或重置为零.
  - 例如,可以使用计数器来表示已服务的请求数、已完成的任务数或错误数.
  - 不要使用计数器来反映一个可能会减小的值.
  - 例如,不要使用计数器表示当前正在运行的进程的数量,这种情况下,你应该用gauge.


**Gauge(计量器)**

- 计量器表示一个可以任意上下移动的数值.
- 计量器通常用于测量温度或当前内存使用量等,也用于“计数”,比如并发请求的数量.


**Histogram(直方图、柱状图)**

- 直方图对观察结果(通常是请求持续时间或响应大小之类的东西)进行采样,并在可配置的桶中计数.它还提供了所有观测值的和.
- 直方图用一个基本的指标名`<basename>`暴露在一个抓取期间的多个时间序列：
- 观察桶的累积计数器,格式为`<basename>_bucket{le="<upper inclusive bound>"}`
- 所有观测值的总和,格式为`<basename>_sum`
已观察到的事件的计数,格式为`<basename>_count`

**Summary(摘要)**
- 与柱状图类似,摘要样例观察结果(通常是请求持续时间和响应大小之类的内容).
- 虽然它还提供了观测值的总数和所有观测值的总和,但它计算了一个滑动时间窗口上的可配置分位数.


---


## metric


```yaml


- job_name: a

  kubernetes_sd_configs:
  - role: endpoints

  relabel_configs:


  - action: drop
    source_labels: [__AA]
    regex: X
    # drop __AA when it = to X
    # not show

  - action: keep
    source_labels:
    - A
    - B
    - C
    regex: 1;2;3
    # change ABC to 123 separately
    # not show

  - action: labelmap
    regex: _A_(.+)
    # keep _A_123456 label without _A_ prefix
    # show

  - action: replace
    source_labels: [__A]
    target_label: A
    # change label name from __A to A
    # show

  - action: replace
    source_labels:
    - __A__
    - __B

    regex: (https?) # value if exsiste
    regex: (.+)     # .value

    regex: ([^:]+)(?::\d+)?;(\d+)
    replacement: $1:$2 # value1:value2

    target_label: __A__
    # change label name from __A__, __B to
    # A:B
    # https
    # .value
    # show


  bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
  scheme: https
  tls_config:
    ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    insecure_skip_verify: true

```






.
