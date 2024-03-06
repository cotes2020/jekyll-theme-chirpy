

- [Falco](#falco)
  - [overall](#overall)
  - [Falco vs Linux系统](#falco-vs-linux系统)
  - [component](#component)
    - [Falco规则文件](#falco规则文件)
      - [列子](#列子)
    - [falco 的配置](#falco-的配置)
    - [falcosidekick](#falcosidekick)
  - [code](#code)
    - [install](#install)
    - [批量部署&更新规则](#批量部署更新规则)

---

# Falco


## overall


- 开源云原生运行时安全工具
- 事实上也是Kubernetes威胁检测引擎。

- 是一种旨在检测异常活动的系统行为监控程序。

- 是一种开源的审计工具，
  - 在用户空间中运行，使用**内核模块**拦截系统调用，通过设置一组规则实现连续监视和检测容器，应用程序，主机和网络活动。
  - 因此，它既能够检测传统主机上的应用程序，
  - 也能够检测Docker容器环境或者PaaS容器云平台。


- Falco可以对Linux系统`调用行为`进行监控
  - 提供了**lkm内核模块**驱动和**eBPF**驱动。

- Falco的主要功能如下：
  - 从内核运行时采集Linux系统调用，提供了一套强大的规则引擎，用于对Linux系统调用行为进行监控
  - 当系统调用违反规则时，会触发相应的告警。








- Falco能够检测或者告警所有涉及系统调用的进程行为。
- 例如：

  - 某容器中启动了一个shell

  - 容器正在特权模式下运行，或者在从主机挂载敏感路径如/proc

  - 某服务进程创建了一个非预期类型的子进程

  - 意外读取敏感文件，例如/etc/shadow文件被读写

  - /dev目录下创建了一个非设备文件

  - ls之类的常规系统工具向外进行了对外网络通信


---


## Falco vs Linux系统



Falco与Linux内核的安全检测工具的**不同**在于：

1. Falco通过底层**内核模块**提供的`系统调用事件流`，实现连续式实时监控功能；

2. Falco运行在用户空间内，通过**内核模块**拦截系统调用，Falco设置的规则方式也比较灵活；

3. Falco的规则语法比较简单，支持Docker容器环境或者PaaS容器云平台。

4. Falco提供了丰富的告警/错误输出方式，支持与其他工具协同工作。

---

## component

![t01bdeacd1c67ac8719](https://i.imgur.com/sClFyxr.png)


### Falco规则文件

Falco规则文件是包含三种类型元素的YAML文件：
- Rules, 就是生成告警的条件以及一下描述性输出字符串。
- Macros, 是可以在规则或者其他宏中重复使用的规则条件片段。
- Lists, 类似Python 列表，定义了一个变量集合。

Falco 使用了Sysdig
- 在rule的 condition里面,任何 Sysdig 过滤器都可以在 Falco 中使用。



#### 列子

1. https://github.com/draios/sysdig/wiki/sysdig-user-guide#filtering

这是一个rule的 condition条件示例，在容器内运行 bash shell 时发出警报：

```yaml
container.id != host and proc.name = bash

# 第一个子句检查事件是否发生在容器中（Sysdig 事件有一个container字段，该字段等于”host”事件是否发生在host主机上）。

# 第二个子句检查进程名称是否为bash。
```

2. 列子

```yaml
- list: my_programs
  items: [ls, cat,  bash]

- macro: access_file
  condition: evt.type=open

- rule: program_accesses_file
  desc: track whenever a set of programs opens a file
  condition: proc.name in (my_programs) and (access_file)
  output: a tracked program opened a file (user=%user.name command=%proc.cmdline file=%fd.name)
  priority: INFO
```


3. web应用进程java，php，apache，httpd，tomcat 中运行其他进程falco demo.

![t01326f4124c0ab1339](https://i.imgur.com/4NDSHEV.png)

4. web应用进程java，php，apache，httpd，tomcat 中读取查看敏感文件falco demo，图片来自，字节沙龙

![t0155da32381e4e3a06-1](https://i.imgur.com/2wmo0Q0.png)


---

### falco 的配置

下面，我们修改falco 的配置，`/etc/falco/falco.yaml`

```yaml
json_output: true
json_include_output_property: true
http_output:
  enabled: true
  url: "https://localhost:2801"
```

启动falco
```bash
systemctl enable falco
systemctl start falco
```


---


### falcosidekick

https://github.com/falcosecurity/falcosidekick.git

- falcosidekick 是一个管道工具
- 接受 Falco的事件并将它们发送到不同的持久化工具中。
  - 使用falcosidekick把 falco post 过来的数据写入es or kafka。
  - 读取kafka里面的东西完成告警
  - 也可以用 Prometheus 和falco-exporter 完成告警。

```yaml
elasticsearch:
   hostport: "https://10.10.116.177:9200"
   index: "falco"
   type: "event"
   minimumpriority: ""
   suffix: "daily"
   mutualtls: false
   checkcert: true
   username: ""
   password: ""


kafka:
  hostport: ""
  topic: ""
  # minimumpriority: "debug"
```

![t010e47727cebe5ca9b](https://i.imgur.com/3WT8nRV.jpg)




---

## code


### install

```bash

curl -s https://falco.org/repo/falcosecurity-3672BA8F.asc | apt-key add -
echo "deb https://download.falco.org/packages/deb stable main" | tee -a /etc/apt/sources.list.d/falcosecurity.list

apt-get update -y
apt-get -y install linux-headers-$(uname -r)
apt-get install -y falco

rpm --import https://falco.org/repo/falcosecurity-3672BA8F.asc
curl -s -o /etc/yum.repos.d/falcosecurity.repo https://falco.org/repo/falcosecurity-rpm.repo
yum -y install kernel-devel-$(uname -r)
yum -y install falco
```


---

### 批量部署&更新规则

- 批量部署和更新规则
- 可以使用saltstack 或者 ansible 下发对应shell脚本来完成

批量部署

```bash
#!/bin/bash

if [  -n "$(uname -a | grep Ubuntu)" ]; then       # 按实际情况修改
        curl -s https://falco.org/repo/falcosecurity-3672BA8F.asc | apt-key add -
        echo "deb https://download.falco.org/packages/deb stable main" | tee -a /etc/apt/sources.list.d/falcosecurity.list
        apt-get update -y
        apt-get install -y falco
else
        rpm --import https://falco.org/repo/falcosecurity-3672BA8F.asc
        curl -s -o /etc/yum.repos.d/falcosecurity.repo https://falco.org/repo/falcosecurity-rpm.repo
        yum -y install falco
fi

systemctl enable falco && systemctl start falco
```



批量更新规则

```bash
#!/bin/bash

BDATE=`date +%Y%m%d%H%M%S`
URL=https://8.8.8.8:8888/falco_update.tar.gz

if [ -d /etc/falco_bak ]
then
        cp -r /etc/falco  /etc/falco_bak/${BDATE}
        rm -rf /etc/falco_bak/falco_update.tar.gz
else
        mkdir /etc/falco_bak
        cp -r /etc/falco  /etc/falco_bak/${BDATE}
fi

curl -o /etc/falco_bak/falco_update.tar.gz ${URL}
rm -rf /etc/falco
tar -xzvf /etc/falco_bak/falco_update.tar.gz -C /etc
systemctl restart falco

# 把规则 falco_update.tar.gz，提前准备好
# 使用saltstack 推下去即可.
# saltstack demo 如下：
[root@localhost ~]$ cat /srv/salt/top.sls
base:
  '*':
    - exec_shell_install

[root@localhost ~]$ cat /srv/salt/exec_shell_install.sls

exec_shell_install:
  cmd.script:
    - source: salt://falco_install.sh
    - user: root

[root@localhost ~]$ salt '*' state.highstate




# 也可以使用ansible 推下去即可.
# ansible demo 如下：

[root@server81 work]$ ansible servers -m shell -a "mkdir -p /var/falco_sh"

[root@server81 ansible]$ ansible servers -m copy -a "src=/root/ansible/falco_install.sh  dest=/var/falco_sh/falco_install.sh mode=0755"
172.16.5.193 | CHANGED => {

[root@server81 ansible]$ ansible servers -m shell -a "/var/falco_sh/falco_install.sh"
172.16.5.193 | CHANGED | rc=0 >>
```












.
