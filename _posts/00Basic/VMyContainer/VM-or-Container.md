---
title: containers vs virtual machines (VM’s)
date: 2020-11-26 11:11:11 -0400
categories: [VMyContainer]
tags: [Linux, VMs]
math: true
image:
---

- [containers vs virtual machines (VM’s)](#containers-vs-virtual-machines-vms)
  - [Application Cell / Container Virtualization](#application-cell--container-virtualization)
  - [containers vs virtual machines (VM’s)](#containers-vs-virtual-machines-vms-1)
  - [Containers 101](#containers-101)

---

# containers vs virtual machines (VM’s)

---


## Application Cell / Container Virtualization



**Application cell / container / Container-based virtualization**:

- runs services or applications within isolated `application cells / containers`.
  - the containers don’t host an entire operating system.
    - Instead, the host’s operating system and kernel run the service or app within each of the containers.
    - However, because they are running in separate containers, none of the services or apps can interfere with services and apps in other containers.

- Benefit:
  - uses fewer resources
    - can be more efficient than a system using a traditional Type II hypervisor virtualization.
    - Internet Service Providers (ISPs) often use it for customers who need specific applications.
- Drawback
  - containers must use the operating system of the host.
    - example
    - if the host is running Linux, all the containers must run Linux.
    - `Docker is different`







---





## containers vs virtual machines (VM’s)



Virtual Machines provide a very strong isolation on the host level and don’t share the OS. The primary reason for developers to move to a microservices based architecture is to break up the app stack into smaller pieces, thus providing a more agile environment. In doing so your application services will now be connected thru the network and this opens up a myriad a potential security issues.



some people believe that containers are inherently more secure than vm’s.

- The argument is that `breaking applications` into `microservices with well-defined interfaces and limited packaged services` **reduces the attack surface** overall.

- The key point is that a well-implemented container deployment which includes security precautions can be at least as secure, if not more secure, than vm’s.





Containers and container deployments face a multitude of different threats, weaknesses and vulnerabilities, which must be considered, and dealt with prior to production.



Unlike VMs, the fundamental risks of open network traffic across services and a shared kernel cannot be ignored and deserves real security concern.



Recommendations can be made for any container platform, and in almost any deployment scenario.  They generally come down to similar security recommendations for almost any system, platform or service:

- Scan your container before and during run-time. Reduce all attack surfaces in the App/OS etc. to only those required and harden what surfaces must be exposed.

- Use network micro segmentation to isolate application clusters based on trust, risk, and exposure.

- Apply and enable all security relevant and supported configuration options for the platform such as registry scanning, access controls, and container privileges.

- Always keep the host and container versions up-to-date.

- Know your app and how it’s supposed to behave. Create visibility to the intra-container and intra-host network traffic (east-west) as well as normal north-south monitoring.

- Continuously test and review the above security recommendations in your CI/CD process.

- Log threats and vulnerabilities from your Docker host in your regular SIEM tool such as Splunk or Nagios

- Implement a third party – container specific security platform such as NeuVector. Remember traditional firewalls can’t keep up with the rapid pace and fluidity of container deployments so status quo is not an option.



Today, security is much more of a concern with containers than it is with virtual machines. In fact, according to a Forrestor Research study, 53% of enterprises deploying containers cite Security as top concern.

This is likely due to the fact that vm’s have reached maturity in their deployment and the attack surfaces are fairly well understood. Containers, on the other hand, are the wild west and no one really knows where attacks could come from.

What About VMs AND Containers?

Many enterprises with existing applications running on a stable vm infrastructure are choosing to take a ‘toe in the water’ approach. By deploying containers on vm’s they get the benefits of mature monitoring and isolation with more rapid DevOpsprocesses. Compared to containers running on bare metal, they do give up some performance, scalability, and cost. But it’s certainly a valid way to transition.

It’s easy to get excited about this new microservice era and all it’s clear benefits. However with all new technologies come new threats that MUST be considered and understood prior to production. At NeuVector we believe that the best protection for containers happens during run-time and as close to container network traffic as possible.

It’s the last line of defense in a new and often changing environment.

Analogously – if it was at all practical and affordable every apartment building would have it’s own doorman and check all visitors ID – right?

 







## Containers 101



Thanks to Docker, containers are now the future of web development. According to DataDog, 15% of hosts run Docker, which is significantly up both from the 6% of hosts running it at this point in 2015 and the 0% of hosts running it before it was released in March of 2013. LinkedIn has also seen a 160% increase in profile references to Docker in just the past year alone, indicating it’s becoming much more important to know something about Docker when looking for work.

What exactly are containers? And why are they so rapidly grabbing developer market share from virtual machines?

To answer these questions, it’s helpful to consider containers in contrast to VMs.

A virtual machine is an emulation of an entire operating system managed by a hypervisor. A virtual machine may be run over the top of another OS or directly off the hardware. In either case, one VM can and usually will be run alongside other VMs, all of which are allocated their own set static space and resources by the hypervisor, with each VM acting as its own independent computer.

A container is a self-contained (it’s right there in the name) execution environment with its own isolated network resources. At a quick glance, a container may appear very similar to a VM. The key difference is that a container does not emulate a separate OS. Containers instead create separate, independent user spaces that have their own bins and libs, but that share the host operating system’s kernel with all other containers running on a machine. This being the case, containers do not need to be assigned their own set amount of RAM and other resources; they simply make use of whatever they need while they’re running.

In short: a virtual machine virtualizes the hardware, while a container virtualizes the OS.

This means containers are significantly more lightweight than VMs. They can be spun up in seconds instead of minutes and you can run as many as 8x as many of them on a single machine. And since the OS has been abstracted away, they can be easily moved from one machine to another.

What is contained in a container?

A container is made up of container images that bundle up the code and all its dependencies. One of these container images will be the app itself. The other images will be the libraries and binaries needed to run that app. All the images that make up the container are then turned into an image template that can be reused across multiple hosts.

It may sound like a lot of effort to add all the necessary individual images to a container, but all your images are stored and run out of a registry. If your application needs PHP 7.0, Apache, and Ubuntu to run then you’ll reference these in your config file and your container manager will pull them from this registry (assuming they’re there).

Where does Docker come into all of this?

Containers are nothing new, having been part of Linux since the creation of chroot in 1982. But to run them, you need a container manager like the one referenced above. Docker is by far the most popular of these (it’s nearly synonymous with containers at this point) and has been at the forefront of the rapid surge in container usage. What sets is apart?

The Docker Hub – This hub is not only the registry where your various images are privately stored, it’s also a rich ecosystem of public images built by other Docker users that you can pull down and use for your own projects. Why do all the grunt work when there are other people out there who have already done it?

Easy Version History and Rollbacks – Docker containers are read-only post-creation. That doesn’t mean you can’t make changes, what it does mean is that any changes are used to create new images whenever you run the “docker commit” command. These new images become new containers that you run just like the original. If an alteration leads to a problems in a new image, then you can simply go back to the previous one.

Portability – Containers are already portable just by the nature of their design, but Docker guarantees the environment will be exactly the same when moving an image from one Docker host to another so you can build once and run anywhere.

Docker is open source and its technology is the basis of the Open Containers Initiative, which is a Linux Foundation initiative focused on creating “industry standards around container formats and runtime.” Google, Amazon, Microsoft and other industry leaders are also part of the OCI.

Are containers available only on Linux?

Until very recently the answer was yes, but Microsoft added container support to Windows Server 2016. This can be managed using Docker for Windows.

Will containers replace virtual machines?

Though containers will absolutely continue to rise in popularity, it’s incredibly unlikely they’ll replace VMs. More likely the two will be used in concert with each other, each being put to use where most appropriate. Occasionally, containers might even be run within virtual machines, warping the space time continuum and confusing partisans of both approaches. We truly live in the future.

There are particular concerns about containers and security, as the varied images provide more points of entry for attackers and a container’s direct access to the OS kernel creates a larger surface area for attack than would be found in a hypervisor controlled virtual machine. As a security company, we’ll certainly have a lot more to say about that in due time.





￼



In computer systems, the attack surface includes anything where the attacker (or software acting on his behalf) can “touch” the target system.

Network interfaces, hardware connections, and shared resources are all possible attack points. Note that the attack surface doesn’t imply that an actual vulnerability exists. All 10 doors might be perfectly secure. But a larger attack surface means more places to protect and the greater likelihood the attacker will find a weakness in at least one.

Structural view

For the structural approach I’ll compare the attack surface of both systems. An attack surface represents the number of points at which a system can be attacked. It isn’t precisely defined (as a number, for example) but is useful for comparisons. To a burglar, a house with 10 doors has a greater attack surface than a house with one door, even if the doors are identical. One door might be left unlocked; one lock might be defective; doors in different locations might offer an intruder more privacy, and so on.



In computer systems, the attack surface includes anything where the attacker (or software acting on his behalf) can “touch” the target system. Network interfaces, hardware connections, and shared resources are all possible attack points. Note that the attack surface doesn’t imply that an actual vulnerability exists. All 10 doors might be perfectly secure. But a larger attack surface means more places to protect and the greater likelihood the attacker will find a weakness in at least one.

The total attack surface depends on the number of different touch points and the complexity of each. Let’s look at a simple example. Imagine an old-fashioned system that serves up stock market quotes. It has a single interface, a simple serial line. The protocol on that line is simple too: A fixed length stock symbol, say five characters, is sent to the server, which responds with a fixed-length price quotation -- say, 10 characters. There's no Ethernet, TCP/IP, HTTP, and so on. (I actually worked on such systems long ago in a galaxy far, far away.)



The attack surface of this system is very small. The attacker can manipulate the electrical characteristics of the serial line, send incorrect symbols, send too much data, or otherwise vary the protocol. Protecting the system would involve implementing appropriate controls against those attacks.

Now imagine the same service, but in a modern architecture. The service is available on the Internet and exposes a RESTful API. The electrical side of the attack is gone -- all that will do is fry the attacker’s own router or switch. But the protocol is enormously more complex. It has layers for IP, TCP, possibly TLS, and HTTP, each offering the possibility of an exploitable vulnerability. The modern system has a much larger attack surface, though it still looks to the attacker like a single interface point.

Bare-metal attack surface

For an attacker not physically present in the data center, the initial attack surface is the network into the server. This led to the “perimeter view” of security: Protect the entry points into the data center and nothing gets in. If the attacker cannot get in, it doesn’t matter what happens between systems on the inside. It worked well when the perimeter interfaces were simple (think dial-up), but fostered weaknesses on internal interfaces. Attackers who found a hole in the perimeter would often discover that the internal attack surface of the server farm was much larger than the external one, and they could do considerable damage once inside.

This internal attack surface included network connections between servers but also process-to-process interactions within a single server. Worse, since many services run with elevated privileges (“root” user), successfully breaking into one would effectively mean unfettered access to anything else on that system, without having to look for additional vulnerabilities. A whole industry grew up around protecting servers -- firewalls, antimalware, intrusion detection, and on and on -- with less than perfect results.



There are also interesting “side channel” attacks against servers. Researchers have shown examples of using power consumption, noise, or electromagnetic radiation from computers to extract information, sometimes very sensitive data such as cryptographic keys. Other attacks have leveraged exposed interfaces like wireless keyboard protocols. In general, however, these attacks are more difficult -- they might require proximity to the server, for example -- so the main path of coming “down the wire” is more common.

VM attack surface

When VMs are used in the same way as bare metal, without any difference in the architecture of the application (as they often are), they share most of the same attack points. One additional attack surface is potential failure in the hypervisor, OS, or hardware to properly isolate resources between VMs, allowing a VM to somehow read the memory of another VM. The interface between the VM and the hypervisor also represents an attack point. If a VM can break through and get arbitrary code running in the hypervisor, then it can access other VMs on the same system. The hypervisor itself represents a point of attack since it exposes management interfaces.

There are additional attack points depending on the type of VM system. Type 2 VM systems use a hypervisor running as a process on an underlying host OS. These systems can be attacked by attacking the host OS. If the attacker can get code running on the host system, he can potentially affect the hypervisor and VMs, especially if he can get access as a privileged user. The presence of an entire OS, including utilities, management tools, and possibly other services and entry points (such as SSH) provides a number of possible attack points. Type 1 VM systems, where the hypervisor runs directly on the underlying hardware, eliminate these entry points and therefore have a smaller attack surface.

Container attack surface

As with VMs, containers share the fundamental network entry attack points of bare-metal systems. In addition, like Type 2 virtual machines, container systems that use a “fully loaded” host OS are subject to all of the same attacks available against the utilities and services of that host OS. If the attacker can gain access to that host, he can try to access or otherwise affect the running containers. If he gets privileged (“root”) access, the attacker will be able to access or control any container. A “minimalist” OS (such as Apcera’s KurmaOS) can help reduce this attack surface but cannot eliminate it entirely, since some access to the host OS is required for container management.

The basic container separation mechanisms (namespaces) also offer potential attack points. In addition, not all aspects of processes on Linux systems are namespaced, so some items are shared across containers. These are natural areas for attackers to probe. Finally, the process to kernel interface (for system calls) is large and exposed in every container, as opposed to the much smaller interface between a VM and the hypervisor. Vulnerabilities in system calls can offer potential access to the kernel. One example of this is the recently reported vulnerability in the Linux key ring.

Architectural considerations

For both VMs and containers, the size of the attack surface can be affected by the application architecture and how the technology is used.

Many legacy VM applications treat VMs like bare metal. In other words, they have not adapted their architectures specifically for VMs or for security models not based on perimeter security. They might install many services on the same VM, run the services with root privileges, and have few or no security controls between services. Rearchitecting these applications (or more likely replacing them with newer ones) might use VMs to provide security separation between functional units, rather than simply as a means of managing larger numbers of machines.

Containers are well suited for microservices architectures that “string together” large numbers of (typically) small services using standardized APIs. Such services often have a very short lifetime, where a containerized service is started on demand, responds to a request, and is destroyed, or where services are rapidly ramped up and down based on demand. That usage pattern is dependent on the fast instantiation that containers support. From a security perspective it has both benefits and drawbacks.

The larger number of services means a larger number of network interfaces and hence a larger attack surface. However, it also allows for more controls at the network layer. For example, in the Apcera Platform, all container-to-container traffic must be explicitly permitted. A rogue container cannot arbitrarily reach out to any network endpoint.

Short container lifetime means that if an attacker does get in, the time he has to do something is limited, as opposed to the window of opportunity presented by a long-running service. The downside is that forensics are harder. Once the container is gone, it cannot be probed and examined to find the malware. These architectures also make it more difficult for an attacker to install malware that survives after container destruction, as he might on bare metal by installing a driver that loads on boot. Containers are usually loaded from a trusted, read-only repository, and they can be further secured by cryptographic checks.

Now let’s consider what goes on during a breach.

Protection against breaches

Attackers typically have one or two goals in cracking into a server system. They want to get data or to do damage.

If they’re after data, they want to infiltrate as many systems as possible, with the highest privileges possible, and maintain that access for as long as possible. Achieving this gives them time to find the data, which might be already there -- a poorly secured database, for example -- or might require slow collection over time as it trickles in, such as collecting transactions as they come in from users. Maintaining access for a long time requires stealth. The attack also requires a way to get the data out.

If the attacker is trying simply to do damage, the goal again is to access as many systems and privileges as possible. But there is a balancing act: Once the damage starts it will presumably be noticed, but the longer the attacker waits to start (while the malware filters from system to system), the greater the chance of being detected. Getting data out is less important than coordinated control of the malware. The idea is to infect as many systems as possible, then damage them at a synchronized point, either prearranged or on command.

Breaches involve a number of elements. Let’s look at each and see if VMs and containerized architectures can affect the attack surface for each.

Gaining initial access. The attacker needs a point of entry into the system, ideally to a point as “deep inside” as possible and with the highest possible privileges. One potential channel is to find a vulnerability in a public-facing endpoint such as a Web server. Since these often run as a privileged user, under an architecture where each VM runs multiple services, such an attack can be very powerful. Containerizing the Web server, or isolating it in its own VM, helps limit the attack by separating the Web server process from other processes. User namespacing separates the root user of the container from the root user of the host OS. For this reason containers offer better protection at this stage of an attack than a legacy VM application, and they are probably on a par with VM architectures that use separate VMs by function.

Server vulnerabilities happen, but the much more common attack point is to gain access to a user’s workstation, leveraging browser vulnerabilities or social engineering. This might get attack code running inside the corporate network, if that’s where the system is located; ideally (from the attacker’s standpoint) the company has done a poor job of segregating the production and employee networks.

Inside the corporate network or not, the malware can monitor the user’s activities and look for credentials for other systems that might be better attack points. If the user is an administrator or other privileged user, with access to sensitive systems, the attacker has hit the jackpot. The security postures of containers and VMs (at the server level) are largely the same here, although there are interesting applications of both VM and container technology at the individual workstation that try to better secure the browser.

Move laterally from system to system. Unless the first system attacked has what the attacker wants -- unlikely -- he will try to spread the attack through the corporate network. The most effective entry point into servers (from, say, an infected workstation) would be a vulnerable host OS for containers or Type 2 VMs. If the attacker compromises such a host and gets root, he will be able to access every container or Type 2 VM on the system. The equivalent attack on a Type 1 virtualization host would be much more difficult, due to the hypervisor’s much smaller attack surface. Direct attacks on interfaces exposed by services have similar attack surfaces for services in containers and VMs. 

Once inside a server, the attacker will want to move throughout that machine and to others. Here containers are more vulnerable to OS attacks, due to the larger attack surface presented by the OS system call interface. The attacker’s ability to extend his reach by calling out to other services depends on how well network controls are applied. If the network is open, the attacker will proceed to search for other systems. However, strict network controls (like those provided in the Apcera Platform) can help limit access in containerized systems.

Escalate privilege. If the attacker already has administrative credentials, especially with access to server side management interfaces (hypervisor management, container management, host OS access), neither approach can provide much protection. In this case the attacker has pretty much won.

Inside a server, the interface between containers and the kernel makes for a larger attack surface than the interface between a VM and a hypervisor. This vulnerability is mitigated by user namespaces, which constrain the power of root inside the container. Privileged access inside the container can affect the app itself and potentially shared resources (like data sources), but can’t access root resources outside the container.

Again, if the VM architecture has many services inside a single VM, privilege escalation can cause more damage, since code running as root inside one service will effectively have unlimited access to the other services. Similar logic argues against container architectures with multiple processes or services inside the same container.

Find sensitive data. Services running together inside a VM often share or have similar access privileges to data, so they're vulnerable to an attack. VMs often have virtual disks used by many processes. On the other hand, the container practice of “no data inside containers” helps isolate and protect sensitive data. Microservices architecture standardizes such access with RESTful APIs that can have standardized controls (such as authentication and authorization) applied to them. Other controls, such as Apcera’s Semantic Pipelines, can provide advanced security between containers and data sources.

Embed a long-lived presence. Both VMs and containers can be booted from trusted repositories, so they are similar in their resistance to an attacker infecting an image with malware that survives across a boot. The underlying host OS, if one is present, may be vulnerable to such an attack. Mechanisms are evolving for secured system boot, starting from an embedded hardware root of trust, that can prevent these kinds of attacks, but they are not yet widely deployed. The long-lived nature of VMs gives containers an edge here, since the container may come and go before the malware has a chance to do much.

Do damage. This is similar to finding sensitive data, but with write access: The attacker wants to change something, wipe a disk, insert transactions, and so on. Container-based microservices architectures encourage better isolation and, hence, better protection against damage than typical VM systems.

Exfiltrate data. A more closed and closely controlled network of containers (such as that in the Apcera Platform) has a smaller attack surface for exfiltration than one with open VMs. However, the ability to apply the controls is key. VMs can be similarly protected (at least VM to VM, if not between services inside a single VM) by appropriate configuration of networking infrastructure, but the process is often manual, tedious, and error prone. Similarly, a container network without inherent platform support for network controls would be difficult to make both operational and secure.

Comparing container and VM security yields no runaway winner. Much depends on how the containers and VMs are used, and specifically on the architecture of the applications they support. In this regard, containers often have an edge because they are more likely to be used for new applications. In some sense it is unfair to compare VMs running within legacy architectures with containers and microservices, but that is often the reality of how they are used. Perimeter controls cannot contain modern attacks. We need to evolve our approach to security and adapt to new architectures. Containers, along with solid platforms for securing and managing these architectures, will be an important part of that evolution.
