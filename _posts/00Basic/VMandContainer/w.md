

[toc]

---


# Preferred Qualifications:


# • Application Security/ Threat Assessment with/without tools and Recommendation

# • Secure Coding Review and Analysis

- applications require a “last look” to ensure that the application and its’ components, are free of security flaws.
- A secure code review serves to `detect all the inconsistencies` that weren’t found in other types of security testing – and to ensure the `application’s logic and business code` is sound.
- Reviews can be done via both manual and automated methods
- **cut down on time and resources it would take if vulnerabilities were detected after release**. The security bugs being looked for during a secure code review have been the cause of countless breaches which have resulted in billions of dollars in lost revenue, fines, and abandoned customers.
- **focus on finding flaws in areas**:
  - Authentication, authorization, security configuration,
  - session management, logging,
  - data validation, error handling, and encryption.
- Code reviewers should be well-versed in the language of the application they’re testing, as well as knowledgeable on the secure coding practices and security controls that they need to be looking out for.
- **need to understand the full context of the application**,
  - including its intended audience and use cases
  - Without that context, code may look secure at first glance, but easily be attacked.
  - Knowing the context by which an app is going to be used and how it will function is the only way to certify that the code adequately protects whatever you’ve relegating to it.

- **5 Tips to a Better Secure Code Review**:
  1. **Produce code review checklists** to ensure consistency between reviews and by different developers
     1. all reviewers are working by the same `comprehensive checklist`. reviewers can forget to certain checks without a well-designed checklist.
     2. enforce time constraints as well as `mandatory breaks` for manual code reviewers. especially when looking at high value applications.

  2. Ensure a **positive security culture by not singling out developers**
     1. It can be easy, especially with reporting by some tools being able to compare results over time, to point the finger at developers who routinely make the same mistakes. It’s important when building a security culture to refrain from playing the blame game with developers; this only serves to deepen the gap between security and development. Use your findings to help guide your security education and awareness program, using those common mistakes as a jumping off point and relevant examples developers should be looking out for.
     2. Again, developers aren’t going to improve in security if they feel someone’s watching over their shoulder, ready to jump at every mistake made. Facilitate their security awareness in more positive ways and your relationship with the development team, but more importantly the organization in general, will reap the benefits.

  3. **Review code each time a meaningful change** in the code has been introduced
     1. If you have a secure SDLC in place, you understand the value of testing code on a regular basis. Secure code reviews don’t have to wait until just before release. For major applications, we suggest performing manual code reviews when new changes are introduced, saving time and human brainpower by having the app reviewed in chunks.

  4. A **mix of human review and tool use is best** to detect all flaws
     1. Tools aren’t (yet) armed with the mind of a human, and therefore can’t detect issues in the logic of code and are hard-pressed to correctly estimate the risk to the organization if such a flaw is left unfixed in a piece of code. Thus, as we discussed above, a mix of static analysis testing and manual review is the best combination to avoid missing blind spots in the code. Use your teams’ expertise to review more complicated code and valuable areas of the application and rely on automated tools to cover the rest.

  5. **Continuously monitor and track patterns** of insecure code
     1. By tracking repetitive issues you see between reports and applications, help inform future reviews by modifying your secure code review checklist, as well as your AppSec awareness training. Monitoring your code offers great insight into the patterns that could be the cause of certain flaws, and will help you when you’re updating your review guide.



• Drive the adoption of secure CI/CD technologies using Java, C#, SQL/NoSQL on Jenkins, Dockers, Ansible, Nginx etc

---

## • Strong experience in Container Security and security orchestration using – WSO2, Docker, Kubernates, Mesos

评估 Docker 的安全性时，主要考虑三个方面:
- 由内核的命名空间和控制组机制提供的 容器内在安全
- Docker 程序（特别是服务端）本身的抗攻击性
- 内核安全性的加强机制 对 容器安全性的影响

**namespace**
- Docker 容器和 LXC 容器很相似，所提供的安全特性也差不多。
- 当用 docker run 启动一个容器时，在后台 Docker 为容器创建了一个独立的`命名空间 namespace 和控制组 cgroups`集合。
- 命名空间提供了最基础也是最直接的隔离，在容器中运行的进程不会被运行在主机上的进程和其它容器发现和作用。
- 每个容器都有自己独有的网络栈，意味着它们不能访问其他容器的 sockets 或接口。
  - 不过，如果主机系统上做了相应的设置，容器可以像跟主机交互一样的和其他容器交互。
  - 当指定公共端口或使用 links 来连接 2 个容器时，容器就可以相互通信了（可以根据配置来限制通信的策略）。
- 从网络架构的角度来看，所有的容器通过本地主机的网桥接口相互通信，就像物理机器通过物理交换机通信一样。
- 那么，内核中实现命名空间和私有网络的代码是否足够成熟？

**cgroup**
- 它提供了很多有用的特性；以及确保各个容器可以公平地分享主机的内存、CPU、磁盘 IO 等资源；当然，更重要的是，控制组确保了当容器内的资源使用产生压力时不会连累主机系统。
- 尽管控制组不负责隔离容器之间相互访问、处理数据和进程，它在防止拒绝服务（DDOS）攻击方面是必不可少的。尤其是在多用户的平台（比如公有或私有的 PaaS）上，控制组十分重要。例如，当某些应用程序表现异常的时候，可以保证一致地正常运行和性能。

**Daemon sec**
- 确保只有可信的用户才可以访问 Docker 服务。Docker 允许用户在主机和容器间共享文件夹，同时不需要限制容器的访问权限，这就容易让容器突破资源限制。例如，恶意用户启动容器的时候将主机的根目录/映射到容器的 /host 目录中，那么容器理论上就可以对主机的文件系统进行任意修改了。
- 终极目标是改进 2 个重要的安全特性：
- 将容器的 root 用户映射到本地主机上的非 root 用户，减轻容器和主机之间因权限提升而引起的安全问题；
- 允许 Docker 服务端在非 root 权限下运行，利用安全可靠的子进程来代理执行需要特权权限的操作。这些子进程将只允许在限定范围内进行操作，例如仅负责虚拟网络设定或文件系统管理、配置操作等。


**Kernel capabililty 内核能力机制**
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



总结
总体来看，Docker 容器还是十分安全的，特别是在容器内不使用 root 权限来运行进程的话。

另外，用户可以使用现有工具，比如 Apparmor, SELinux, GRSEC 来增强安全性；甚至自己在内核中实现更复杂的安全机制。

---

• Experience in secure software development and integration of CI/CD pipe line with Security Tools like Fortify, Checkmarx, • Nessus and Container Security Tools

• Work with Development/ Architecture team ensuring secure design principles

• Lead development team during design and build phase
• Work on new solutions/ research to bring innovative IP solutions
• Work with other practices/ alliance partners to build new solutions in Application Security

# • Exposure to Application Security threat models

Threat modeling works to `identify, communicate, and understand threats` and `mitigations within the context` of protecting something of value.
- Threat modeling can be applied to a wide range of things, including software, applications, systems, networks, distributed systems, things in the Internet of things, business processes, etc.
- There are very few technical products which cannot be threat modeled; more or less rewarding, depending on how much it communicates, or interacts, with the world.
- Threat modeling can be done at any stage of development

Most of the time, a threat model includes:
- A description / design / model of what you’re worried about
- A list of assumptions that can be checked or challenged in the future as the threat landscape changes
- A list of potential threats to the system
- A list of actions to be taken for each threat
- A way of validating the model and threats, and verification of success of actions taken

> Threat modeling: the sooner the better, but never too late.

The inclusion of threat modeling in the SDLC can help

Build a secure design
- Efficient investment of resources; appropriately prioritize security, development, and other tasks
- Bring Security and Development together to collaborate on a shared understanding, informing development of the system
- Identify threats and compliance requirements, and evaluate their risk
- Define and build required controls.
- Balance risks, controls, and usability
- Identify where building a control is unnecessary, based on acceptable risk
- Document threats and mitigation
- Ensure business requirements (or goals) are adequately protected in the face of a malicious actor, accidents, or other causes of impact
- Identification of security test cases / security test scenarios to test the security requirements

4 Questions
Most threat model methodologies answer one or more of the following questions in the technical steps which they follow:

**What are we building?**
- As a starting point
- define the scope of the Threat Model. need to understand the application you are building, examples of helpful techniques are:
  - Architecture diagrams
  - Dataflow transitions
  - Data classifications
- You will also need to gather people from different roles with sufficient technical and risk awareness to agree on the framework to be used during the Threat modeling exercise.

1. **Identify threats**.
   1. model the system either with data flow diagrams (DFDs) or UML deployment diagrams.
   2. From these diagrams, identify entry points to your system such as data sources, application programming interfaces (APIs), Web services and the user interface itself.
   3. Because an adversary gains access to your system via entry points, they are your starting points for understanding potential threats.
   4. To help identify security threats you should add "privilege boundaries" with dotted lines onto your diagrams.
   5. Figure 1 depicts an example deployment diagram used to explain the boundaries applicable to testing a relational database. A privilege boundary separates processes, entities, nodes and other elements that have different trust levels. Wherever aspects of your system cross a privilege boundary, security problems can arise. For example, your system's ordering module interacts with the payment processing module. Anybody can place an order, but only manager-level employees can credit a customer's account when he or she returns a product. At the boundary between the two modules, someone could use functionality within the order module to obtain an illicit credit.
2. **Understand the threat(s)**.
   1. To understand the potential threats at an entry point, you must identify any security-critical activities that occur and imagine what an adversary might do to attack or misuse your system. Ask yourself questions such as "How could the adversary use an asset to modify control of the system, retrieve restricted information, manipulate information within the system, cause the system to fail or be unusable, or gain additional rights. In this way, determine the chances of the adversary accessing the asset without being audited, skipping any access control checks, or appearing to be another user. To understand the threat posed by the interface between the order and payment processing modules, you would identify and then work through potential security scenarios. For example, an adversary who makes a purchase using a stolen credit card and then tries to get either a cash refund or a refund to another card when he returns the purchase.
3. Categorize the threats. To categorize security threats, consider the STRIDE (Spoofing, Tampering, Repudiation, Information disclosure, Denial of Service, and Elevation of privilege) approach. Classifying a threat is the first step toward effective mitigation. For example, if you know that there is a risk that someone could order products from your company but then repudiate receiving the shipment, you should ensure that you accurately identify the purchaser and then log all critical events during the delivery process.
4. Identify mitigation strategies. To determine how to mitigate a threat, create a diagram called a threat tree. At the root of the tree is the threat itself, and its children (or leaves) are the conditions that must be true for the adversary to realize that threat. Conditions may in turn have subconditions. For example, under the condition that an adversary makes an illicit payment. The fact that the person uses a stolen credit card or a stolen debit/check card is a subcondition. For each of the leaf conditions, you must identify potential mitigation strategies; in this case, to verify the credit card using the XYZ verification package and the debit card with the issuing financial institution itself. Every path through the threat tree that does not end in a mitigation strategy is a system vulnerability.
5. Test. Your threat model becomes a plan for penetration testing. Penetration testing investigates threats by directly attacking a system, in an informed or uninformed manner. Informed penetration tests are effectively white-box tests that reflect knowledge of the system's internal design , whereas uninformed tests are black box in nature.



# • Experience with project management


# • Experience and desire to work in a management consulting environment that requires regular travel
