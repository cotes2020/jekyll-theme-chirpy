# AWS service categories

- [AWS service categories](#aws-service-categories)
  - [overall](#overall)
  - [Compute 计算](#compute-计算)
  - [Storage 存储](#storage-存储)
  - [Database 数据库](#database-数据库)
  - [Migration \& Transfer 数据迁移和传输](#migration--transfer-数据迁移和传输)
  - [Networking \& Content Delivery 网络和内容传送](#networking--content-delivery-网络和内容传送)
  - [Developer Tools 开发工具](#developer-tools-开发工具)
  - [Robotics 机器人](#robotics-机器人)
  - [Customer Enablement 针对每个客户的优化](#customer-enablement-针对每个客户的优化)
  - [Blockchain 区块链](#blockchain-区块链)
  - [Satellite 卫星](#satellite-卫星)
  - [Quantum Technologies 量子技术](#quantum-technologies-量子技术)
  - [Management \& Governance 管理与政府治理](#management--governance-管理与政府治理)
  - [Identity \& access management](#identity--access-management)
  - [Detection](#detection)
  - [Infrastructure protection](#infrastructure-protection)
  - [Data protection](#data-protection)
  - [Incidence response](#incidence-response)
  - [Compliance](#compliance)
  - [Media Services | 媒体服务](#media-services--媒体服务)
  - [Machine Learning 机器学习](#machine-learning-机器学习)
  - [Analytics 分析](#analytics-分析)
  - [Mobile 移动设备](#mobile-移动设备)
  - [AR \& VR 增强现实和虚拟现实](#ar--vr-增强现实和虚拟现实)
  - [Application Integration 应用程序集成](#application-integration-应用程序集成)
  - [AWS Cost Management AWS的成本管理](#aws-cost-management-aws的成本管理)
  - [Customer Engagement 客户交互](#customer-engagement-客户交互)
  - [Business Applications 商业应用程序](#business-applications-商业应用程序)
  - [End User Computing 终端用户计算机技术](#end-user-computing-终端用户计算机技术)
  - [Internet Of Things 物联网](#internet-of-things-物联网)
  - [Game Development 游戏开发](#game-development-游戏开发)
  - [Containers 容器](#containers-容器)


---

## overall

![Screenshot 2023-10-01 at 21.10.30](/assets/img/post/Screenshot%202023-10-01%20at%2021.10.30.png)

![Screenshot 2023-10-01 at 21.10.38](/assets/img/post/Screenshot%202023-10-01%20at%2021.10.38.png)

![Screenshot 2023-10-01 at 21.10.46](/assets/img/post/Screenshot%202023-10-01%20at%2021.10.46.png)

![Screenshot 2023-10-01 at 22.30.58](/assets/img/post/Screenshot%202023-10-01%20at%2022.30.58.png)


---

## Compute 计算

- EC2
  - 虚拟私有服务器

- Lightsail
  - 亚马逊提供的托管服务商(VPS、DNS、存储)

- Lambda
  - 可以运行用Python，Node.js，Go等语言编写的代码，还可以并行运行。

- Batch
  - 在EC2机器的Docker容器中运行软件指令

- Elastic Beanstalk
  - 在托管的虚拟机上运行软件

- Serverless Application Repository
  - (在Lambda)可部署的无服务器应用程序的存储库

- AWS Outposts
  - 可以在您的数据中心使用亚马逊服务

- EC2 Image Builder
  - 自动创建EC2图像

![Screen Shot 2020-06-24 at 13.24.29](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.24.29.png)

## Storage 存储

- S3
  - 不能直接挂载，但可以通过HTTP下载的文件存储

- EFS
  - 可以将网络上的磁盘挂载到您的机器上使用的网络文件系统

- FSx
  - 可以从EC2机器连接的Windows或Lustre的文件系统

- S3 Glacier
  - 用于备份和归档的低成本存储系统

- Storage Gateway
  - 可以把S3连接到自有（或远程控制）的机器上使用的iSCSI

- AWS Backup
  - 自动备份不同AWS服务(如EC2和RDS)

![Screen Shot 2020-06-24 at 13.24.46](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.24.46.png)


## Database 数据库


- RDS
  - 托管的MySQL，PostgreSQL数据库

- DynamoDB
  - 大规模可扩展的非关系数据库

- ElastiCache
  - 托管的分布式的memcache高速缓存系统和redis高性能Key-Value数据库

- Neptune
  - 图表数据库

- Amazon Redshift
  - 用来大量存储流计算可处理数据的数据仓

- Amazon QLDB
  - 可供选择的用于加密验证的数据(如货币交易)的数据库

- Amazon DocumentDB
  - MongoDB的克隆(不完全兼容)

- Amazon Keyspaces
  - 托管的Apache Cassandra的克隆

![Screen Shot 2020-06-24 at 13.25.03](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.25.03.png)


## Migration & Transfer 数据迁移和传输

- Migration Hub
  - 从数据中心迁移到AWS

- Application Discovery Service
  - 在您的数据中心提供检测服务

- Database Migration Service
  - 可以将运行中的数据库迁移到RDS，不同的数据结构之间也可以实施

- Server Migration Service
  - 可以将虚拟机迁移到AWS

- AWS Transfer Family
  - 以S3为基础的(s)FTP服务。可以通过FTP将数据传输到S3存储桶

- Snowball
  - 可申领一台AWS机器并连接到您的数据中心，将数据快速传输到AWS后再归还机器

- DataSync(DataSync)
  - 在数据中心和AWS之间同步数据

![Screen Shot 2020-06-24 at 13.25.15](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.25.15.png)


## Networking & Content Delivery 网络和内容传送

- [VPC](https://ocholuo.github.io/posts/VPC/)
  - 在AWS中创建您自己的VPN

- [VPC Flow]
  - to monitor VPC traffic

- [CloudFront](https://ocholuo.github.io/posts/CloudFront/)
  - 内容传送网络

- Route 53
  - 管理域名和记录

- [API Gateway](https://ocholuo.github.io/posts/Gatway-API-Gateway/)
  - 创建HTTP API并将它们连接到不同的后端

- Direct Connect
  - (物理)连接您的系统(或数据中心)和AWS

- AWS App Mesh
  - 可以作为您的容器(ECS或EKS)的sidecar自动运行Envoy

- AWS Cloud Map
  - 为您的容器提供检测服务

- Global Accelerator
  - 在边缘位置运行应用程序（CDN的应用程序版本）

![Screen Shot 2020-06-24 at 13.25.29](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.25.29.png)

---

## Developer Tools 开发工具
| service                                                        | info                                                               |
| -------------------------------------------------------------- | ------------------------------------------------------------------ |
| CodeStar(CodeStar)                                             | 可以使用template code、CodeCommit和CodeBuild的模板快速开发应用程序 |
| [CodeCommit](https://ocholuo.github.io/posts/1CodeCommit/)     | 亚马逊资源存储库(如git存储库等)                                    |
| [CodeBuild](https://ocholuo.github.io/posts/2CodeBuild/)       | 持续集成服务                                                       |
| [CodeDeploy](https://ocholuo.github.io/posts/3CodeDeploy/)     | 部署服务                                                           |
| [CodePipeline](https://ocholuo.github.io/posts/0CodePipeline/) | 按照定义的工作流进行代码传送                                       |
| Cloud9                                                         | 在线的集成开发环境IDE                                              |
| X-Ray                                                          | 可以分析和调试应用程序，支持Python、Node.js、Go等开发语言          |

## Robotics 机器人
| service       | info                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------ |
| AWS RoboMaker | Cloud solution for robotic developers to simulate, test and securely deploy robotic applications |
| AWS RoboMaker | 为机器人工程师提供的云端解决方案，可用于模拟、测试和安全部署机器人的应用程序                     |

## Customer Enablement 针对每个客户的优化
| service          | info                            |
| ---------------- | ------------------------------- |
| AWS IQ           | 可以根据需要聘请AWS专家的招工板 |
| Support          | AWS支持中心                     |
| Managed Services | 委托AWS为您运行管理AWS服务      |

## Blockchain 区块链
| service                   | info   |
| ------------------------- | ------ |
| Amazon Managed Blockchain | 区块链 |

## Satellite 卫星
| service        | info                           |
| -------------- | ------------------------------ |
| Ground Station | 分时无线电和指向太空的大型天线 |

## Quantum Technologies 量子技术
| service       | info         |
| ------------- | ------------ |
| Amazon Braket | 一些量子技术 |

## Management & Governance 管理与政府治理

- [CloudWatch](https://ocholuo.github.io/posts/AWS-CloudWatch/)
  - 从不同的AWS组件获取日志

- AWS Auto Scaling
  - 根据您自己设置的输入和规则对资源进行缩放

- [CloudFormation](https://ocholuo.github.io/posts/CloudFormation/)
  - 使用模板创建和设置AWS组件

- OpsWorks
  - 通过Ansible实现自动化运维

- Service Catalog
  - 管理云端的项目或代码的列表

- Systems Manager
  - 可以自由对资源进行分组和查看数据，例如单个应用程序

- AWS AppConfig
  - 可以保存或发布应用程序的配置数据

- Trusted Advisor
  - 检查账户成本和安全性等问题

- Control Tower
  - 管理多个帐户

- AWS License Manager
  - 管理许可证

- AWS Well-Architected Tool
  - 创建关于系统的问卷调查，并查看其是否符合最佳实践路线

- Personal Health Dashboard
  - AWS状态页面

- AWS Chatbot
  - 可以使AWS与Slack联动

- Launch Wizard
  - 用来部署MS SQL或SAP的软件

- AWS Compute Optimizer
  - 发现最佳资源并指导您如何降低成本

![Screen Shot 2020-06-24 at 13.25.43](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.25.43.png)

---

## Identity & access management

- [AWS Identity & Access Management (IAM)](https://ocholuo.github.io/posts/IAM/)
  - AWS权限系统，可管理用户和AWS服务
  - Securely manage access to services and resources
  - securely control access

- [AWS CloudTrail](https://ocholuo.github.io/posts/AWS-CloudTrail/)
  - 记录在您的AWS服务中谁做了什么
  - Track user activity and API usage
  - Track app related/infrastructure access


- [AWS Single Sign-On (SSO)](https://ocholuo.github.io/posts/SSO/)
  - 可以在应用程序中使用单点登录功能
  - Cloud single-sign-on (SSO) service

- Amazon Cognito
  - Identity management for the apps
  - 用户和密码管理系统，方便应用程序的用户管理


- AWS Directory Service:
  - Managed Microsoft Active Directory
  - SaaS的动态目录

- AWS Resource Access Manager:
  - Simple, secure service to share AWS resources
  - 与其他账户共享AWS资源，如Route 53和EC2

- [AWS Organizations](https://ocholuo.github.io/posts/AWS-organizations/)
  - 设置多个组织和帐户
  - Central governance and management across AWS accounts

![Screen Shot 2020-06-24 at 13.26.48](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.26.48.png)

---

## Detection

- AWS Security Hub:
  - Unified security and compliance center
  - 利用GuardDuty、Inspector、Macie等的综合安全检查器


- [Amazon GuardDuty](https://ocholuo.github.io/posts/AWS-Duty/)
  - CloudTrail自动扫描VPC日志以应对威胁
  - Managed threat detection service


- [Amazon Inspector](https://ocholuo.github.io//posts/AWS-Inspector/)
  - 自动检测网络和机器的(安全)问题
  - Analyze application security


- AWS Config:
  - Record and evaluate configurations of the AWS resources
  - 审核您的AWS资源配置

---

## Infrastructure protection
- AWS Shield:
  - DDoS protection
  - Web应用防火墙,可以设置规则或指定预先准备的规则

- AWS Web Application Firewall (WAF):
  - Filter malicious web traffic, mitigate cross-site scripting attacks and also SQL injection attacks on the application

- AWS Firewall Manager:
  - Central management of firewall rules
  - 组织内不同帐户的防火墙管理

---

## Data protection

- Amazon Macie:
  - Discover and protect the sensitive data at scale
  - 分析S3存储桶中的数据并检查您的个人信息

- AWS Key Management Service (KMS):
  - Key storage and management
  - 管理加密密钥

- AWS CloudHSM:
  - Hardware based key storage for regulatory compliance
  - 硬件安全模块,可以生成和操作加密密钥

- AWS Certificate Manager:
  - Provision, manage, and deploy public and private SSL/TLS certificates
  - 管理SSL证书和颁发(免费)证书

- AWS Secrets Manager:
  - Rotate, manage and retrieve secrets
  - 保护加密数据，如密钥。也可以自动旋转秘密

---

## Incidence response
- Amazon Detective:
  - Investigate potential security issues
  - (来自Security Hub等)将安全问题留在日志中

- Cloud Endure Disaster Recovery:
  - Fast, automated, cost- effective disaster recovery

---

## Compliance
- AWS Artifact:
  - No cost, self-service portal for on-demand access to AWS’ compliance reports
  - 云合规性文档(ISO/IEC 27001类似的东西)

---

## Media Services | 媒体服务

- Elastic Transcoder
  - 将S3的文件转换为不同的格式或者以S3格式存储

- Kinesis Video Streams
  - 捕获媒体流

- Elemental MediaConnect
  - 截止目前内容不明

- Elemental MediaConvert
  - 将媒体转换为不同的格式

- Elemental MediaLive
  - 分享实时视频

- Elemental MediaPackage
  - 截止目前内容不明

- Elemental MediaStore
  - 截止目前内容不明

- Elemental MediaTailor
  - 在视频广播中插入广告

- Elemental Appliances & Software
  - 可以在本地创建视频，基本上是上述服务的组合

![Screen Shot 2020-06-24 at 13.26.01](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.26.01.png)

---


## Machine Learning 机器学习
| service                                                              | info                                             |
| -------------------------------------------------------------------- | ------------------------------------------------ |
| Amazon SageMaker                                                     | 机器学习工具                                     |
| [Amazon DataWrangler](https://ocholuo.github.io/posts/DataWrangler/) | Tableau + ELK for stored data                    |
| Amazon CodeGuru                                                      | 在机器学习中配置Java代码                         |
| Amazon Comprehend                                                    | 理解并对邮件和推文的内容进行分类                 |
| Amazon Forecast                                                      | 根据数据进行预测                                 |
| Amazon Fraud Detector                                                | 截止目前内容不明                                 |
| Amazon Kendra                                                        | 通过问题搜索服务                                 |
| Amazon Lex                                                           | 可以创建语音对话和聊天机器人                     |
| Amazon Machine Learning                                              | 不推荐，SageMaker是后继产品                      |
| Amazon Personalize                                                   | 可以根据数据创建针对个人做最优化的推荐           |
| Amazon Polly                                                         | 可以从文本转换为不同语种的语音                   |
| Amazon Rekognition                                                   | 识别图像中的物体或人物                           |
| Amazon Textract                                                      | 识别图像中的文本并将其作为文本输出(光学字符识别) |
| Amazon Transcribe                                                    | 将音声转换为文本                                 |
| Amazon Translate                                                     | 将文本翻译成其他语言                             |
| AWS DeepLens                                                         | 进行机器学习的摄像机                             |
| AWS DeepRacer                                                        | 一种在机器学习中编程竞赛的赛车游戏               |
| Amazon Augmented AI                                                  | 让人类参与学习流程，使机器学习更好               |
| AWS DeepComposer                                                     | 用电脑作曲，听上去相当的厉害                     |

## Analytics 分析

- Athena
  - 将查询数据保存在S3存储桶中

- EMR
  - 大数据框架可以执行缩放

- CloudSearch
  - 托管文档搜索系统(Elasticsearch的AWS版本)

- Elasticsearch Service
  - SaaS的Elasticsearch

- Kinesis
  - 以可分析的形式收集大量数据(可能类似ELK)

- QuickSight
  - 商业智能服务

- Data Pipeline
  - 将数据移动或变换格式到DynamoDB、RDS或S3等

- AWS Data Exchange
  - 寻找那些数据可以加以利用的API，但这可能会非常昂贵

- AWS Glue
  - ETL提高和验证服务和数据质量

- AWS Lake Formation
  - 数据湖(数据湖)创建(创建)

- MSK
  - SaaS的Apache Kafka

![Screen Shot 2020-06-24 at 13.26.33](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.26.33.png)


---


## Mobile 移动设备
| service     | info                                                            |
| ----------- | --------------------------------------------------------------- |
| AWS Amplify | 在AWS上自动生成并自动部署前端和后端应用程序                     |
| Mobile Hub  | 现在Amplify的一部分                                             |
| AWS AppSync | 可以创建可连接的后端API，也可以通过Amplify创建                  |
| Device Farm | AWS的BrowserStack，可以在不同的移动设备和浏览器上自动进行测试。 |

## AR & VR 增强现实和虚拟现实
| service         | info             |
| --------------- | ---------------- |
| Amazon Sumerian | 截止目前内容不明 |


---

## Application Integration 应用程序集成

- Step Functions
  - 可以用亚马逊自己的语言描述机器配置

- Amazon AppFlow
  - 可以自动绑定多个应用程序(可能类似zapier)

- Amazon EventBridge
  - 类似eventbus系统

- Amazon MQ
  - 由亚马逊管理的ActiveMQ

- Simple Notification Service
  - 通过电子邮件、短信等方式通知系统

- Simple Queue Service
  - 消息队列(消息队列)系统的系统

- SWF
  - 可以创建工作流程

![Screen Shot 2020-06-24 at 13.27.04](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.27.04.png)

---

## AWS Cost Management AWS的成本管理
| service                        | info                      |
| ------------------------------ | ------------------------- |
| AWS Cost Explorer              | 可视化AWS成本状况         |
| AWS Budgets                    | 创建AWS预算               |
| AWS Marketplace Subs criptions | 查找并购买已安装软件的AMI |

## Customer Engagement 客户交互
| service              | info                                         |
| -------------------- | -------------------------------------------- |
| Amazon Connect       | AWS呼叫中心平台                              |
| Pinpoint             | 通过模板创建交易用的电子邮件、短信或语音电话 |
| Simple Email Service | 邮件提供商，可以发送邮件                     |

## Business Applications 商业应用程序
| service            | info                     |
| ------------------ | ------------------------ |
| Alexa for Business | 将业务与Alexa联系起来    |
| Amazon Chime       | Zoom的AWS版本            |
| WorkMail           | AWS版本的Gmail和谷歌日历 |

---

## End User Computing 终端用户计算机技术

- WorkSpaces
  - 提供Windows或Linux的虚拟桌面服务

- AppStream 2.0
  - 可以将应用程序分发到浏览器

- WorkDocs
  - 可以在线保存和管理文档

- WorkLink
  - 可以将移动端用户连接到内联网

![Screen Shot 2020-06-24 at 13.27.22](/assets/img/post/Screen%20Shot%202020-06-24%20at%2013.27.22.png)

---

## Internet Of Things 物联网

- IoT Core
  - 通过MQTT代理管理IoT设备组

- FreeRTOS
  - 用于微型控制器的RTOS操作系统，可自动连接到IoT Core或Greengrass

- IoT1-Click
  - 一键连接和管理Lambda等系统

- IoT Analytics
  - 可以结构化和存储各种消息进行分析

- IoT Device Defender
  - 检测设备异常并采取行动

- IoT Device Management
  - 对IoT设备进行分组，为作业安排和远程访问设置

- IoT Events
  - 监控设备使用情况，并自行执行AWS服务和作业

- IoT Greengrass
  - 如果到IoT Core的连接是断断续续的，消息代理可以对最多200台能够相互通信的本地设备进行数据缓冲

- IoT SiteWise
  - 收集、结构化、分析和视觉化来自工业设备的数据

- IoT Things Graph
  - 类似CloudFormatation的设计工具，用于将设备与其他AWS服务的通信方式视觉化

## Game Development 游戏开发
| service         | info                  |
| --------------- | --------------------- |
| Amazon GameLift | 在AWS上部署游戏服务器 |

## Containers 容器
| service                                                                     | info                                                        |
| --------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [Elastic Container Registry]()                                              | 可以像在Docker Hub一样保存Docker映像                        |
| [Elastic Container Service - ECS](https://ocholuo.github.io/posts/AWS-ECS/) | 可以在您自己的EC2机器或者所管理的Fargate机器上运行container |
| [Elastic Kubernetes Service]()                                              | SaaS的Kubernetes                                            |

.
