---
title: Meow's CyberAttack - Application/Server Attacks - DDos Dos - Sinkholing
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, DDos]
tags: [CyberAttack, DDos]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - DDos Dos - Sinkholing](#meows-cyberattack---applicationserver-attacks---ddos-dos---sinkholing)
  - [Sinkholing](#sinkholing)

---

# Meow's CyberAttack - Application/Server Attacks - DDos Dos - Sinkholing

book: 
<font color=LightSlateBlue></font>
<font color=OrangeRed></font>

---

## Sinkholing

- Sinkholing可以将网络中的数据流量进行有目的的转发

- 既可以是针对正常的数据流量也可以在发生网络攻击时作为一种防御措施。

- 当一个僵尸网络中的肉鸡向服务器发送数据时，就可以使用sinkholing技术将这些数据流量进行有目的的转发，以此来对僵尸网络进行监控、查找受到影响的域IP地址，最终瓦解僵尸网络的攻击，让那些肉鸡无法接受命令。

- 政府执法部门可以用这种技术来调查大规模的网络犯罪活动。

- 日常生活中，各类互联网基础设施运营商和内容分发网络都会使用这种技术来保护自己的网络和用户免受攻击，调整网络内的数据流量分布情况。

- `Law enforcement seize domain. Then point domain to null IP. All requests to that domain just get "lost" are dropped.`

Example
- 最著名的例子就是Marcus Hutchins利用sinkholing技术成功阻止了WannaCry恶意勒索软件的传播。
- 当WannaCry正在大规模发作时，他和其他的研究人员对这个恶意勒索软件进行反编译，在其中找到了一个弱点。WannaCry会指向一个特定的网址，但该网址域名不属于任何人，因此他就用10.69美元买下并启用了该域名。
- 在WannaCry的代码中，如果检测到该域名被启用，那么程序自动停止运行。正是由于这样的操作挽救了全球无数电脑免受攻击，WannaCry背后的开发人员疏忽了，只会对静态域名进行检测而不是动态域名的检测。
- Marcus Hutchins通过设置买下的域名，将所有WannCry的数据流量转发到自己建立的sinkhole服务器上，然后研究这些数据流量。
- 当他注册了该域名后，由于WannaCry的大量数据导致了sinkhole服务器已经逼近最大负载。
- 虽然他的sinkhole服务器不能挽救已经中招的电脑，也不能阻止WannaCry的传播，但是给网络安全人员留出了宝贵时间来采取相应的防护措施。


