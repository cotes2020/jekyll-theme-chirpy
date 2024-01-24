---
title: Lab - NDG - PAN8 Cybersecurity-Gateway
date: 2020-07-16 11:11:11 -0400
description: Learning Path
categories: [Lab, NDG]
img: /assets/img/sample/rabbit.png
tags: [Lab, NDG]
---

# 2020-07-16-PAN8-Cybersecurity-Gateway

[toc]

From the NDG: [PAN8 Cybersecurity Essentials](https://portal.netdevgroup.com/learn/pan8-ce-pilot/wb79h483WM)

![Screen Shot 2020-05-27 at 23.24.24](https://i.imgur.com/AvhpUOu.png)

---

## Lab 1: Configuring TCP/IP and a Virtual Router

### 1.0 Load Lab Configuration
load the Firewall configuration file.
1. Client PC.
2. Google Chrome: `https://192.168.1.254`
3. Login to the Firewall web interface
4. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-01`
6.  Click the Commit link

### 1.1 Configure Ethernet Interfaces with Layer 3 Information
confirm you have no connectivity to the Firewall from the inside network.
configure the Firewall with Layer 3 information.
1. **CMD** > `ping 192.168.1.1`: no host
2. **Firewall administrator page**: Network > Interfaces > `Ethernet` > interface `ethernet1/2`
3. **Ethernet Interface window**
   1. Interface Type dropdown, select Layer3.
   2. Security Zone dropdown, select inside.
4. In the Ethernet Interface window, click on the IPv4 tab and click on the Add button at the bottom-left. Type 192.168.1.1/24 in the address field.
5. Click on the Advanced tab and under the Management Profile dropdown, select allow-mgmt and click OK.
6. Click the Commit link located at the top-right of the web interface. The allow-mgmt Management Profile allows the interface to accept pings and to accept management functions such as configuring the Firewall with SSH or a web browser.
7.  In the Commit window, click Commit to proceed with committing the changes.
8.  When the commit operation successfully completes, click Close to continue.


### 1.2 Create a Virtual Router
create a Virtual Router allows the Firewall to do routing functions so that the Firewall and devices behind it can access other networks and the Internet.
1. Navigate to Network > Virtual Routers > add
2. Name field: VR-1
3. Add: ethernet1/1, ethernet1/2.
4. Static Routes: Add
5. **Virtual Router** – Static Route – Ipv4 window, type default-route in the
   1. Name: `default-route`
   2. Destination: `0.0.0.0/0`
   3. Interface: `ethernet1/1`
   4. Next Hop: `IP Address`
   5. field below it: `203.0.113.1`
   6. click OK

![Screen Shot 2020-05-28 at 19.17.55](https://i.imgur.com/5olV14o.png)

![Screen Shot 2020-05-28 at 19.17.37](https://i.imgur.com/6oi8bK4.png)

![Screen Shot 2020-05-28 at 19.18.41](https://i.imgur.com/u3atRL2.png)

### 1.3 Verify Network Connectivity
confirm you now have connectivity to the Firewall from the inside network by utilizing ping and connecting to the web interface.
1. CMD icon: `ping 192.168.1.1`
2. In Google Chrome: https://192.168.1.1
3. see the Firewall web interface on the 192.168.1.1 IP address that was configured earlier.


---

## Lab 2: Configuring DHCP

### 2.0 Load Lab Configuration
load the Firewall configuration file.
1. Client PC.
2. Google Chrome: https://192.168.1.254
3. Login to the Firewall web interface as username admin, password admin.
4. In the web interface, navigate to Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-02`
5.  Click the Commit link located at the top-right of the web interface.

### 2.1 Configure DHCP Server
configure a DHCP Server on the Firewall, clients behind the Firewall will not have to manually configure IP addresses.
A client that is configured for DHCP and connected to the same network as the Firewall will receive an IP address automatically, reducing network configuration errors.

1. Navigate to Network > DHCP > DHCP Server > Add
2. **DHCP Server window**: Lease
    - Interface: `ethernet1/2`
    - Mode dropdown: `enabled`
    - Lease radio: `Timeout, 2 days`
    - IP Pools: `192.168.1.100-192.168.1.110`

> 2-day timeout, the client will need to request a new IP address every 2 days

![Screen Shot 2020-05-28 at 19.37.06](https://i.imgur.com/uTtFWSD.png)

3. **DHCP Server window**: Option
    - Gateway: `192.168.1.1`
    - Subnet Mask: `255.255.255.0`
    - Primary DNS: `8.8.8.8`

![Screen Shot 2020-05-28 at 19.38.20](https://i.imgur.com/Hwe3HPO.png)

4.  Click the Commit link


### 2.2 Configure Client for DHCP

1. CMD window: ipconfig /all
    - client IP: `192.168.1.20`

![Screen Shot 2020-05-28 at 19.44.28](https://i.imgur.com/lDaoYFo.png)

![Screen Shot 2020-05-28 at 19.41.55](https://i.imgur.com/oKl4OET.png)

2.  Start Menu > Control
Panel > Network and Sharing Center >  Change adapter > **internal** > `Properties` > **Internet
Protocol Version 4(TCP/IPv4)** > `Properties`
    - `Obtain an IP address automatically`
    - `Obtain DNS server address automatically`.

![Screen Shot 2020-05-28 at 19.42.51](https://i.imgur.com/y88SzPG.png)

![Screen Shot 2020-05-28 at 19.43.51](https://i.imgur.com/2LYPbOI.png)

2. CMD window: ipconfig /all
    - client IP: `192.168.1.100`
![Screen Shot 2020-05-28 at 19.45.51](https://i.imgur.com/JqWi3aj.png)


### 2.3 Configure a `DHCP Client Reservation`
configure a DHCP Client Reservation.
- statically assign an IP address to a client via the DHCP Server. assist the DHCP server in leasing the proper address.

1. Network > DHCP > DHCP Server > ethernet1/2
    - reserved address + MAC ADDRESS

![Screen Shot 2020-05-28 at 19.49.39](https://i.imgur.com/qaPBhCq.png)

CMD:
- ipconfig /release
release the current DHCP lease.
- ipconfig /renew
request a new lease from the DHCP server.


### 2.4 Configure the Firewall Outside Interface for DHCP
configure the Firewall outside interface for DHCP. the Firewall will obtain an IP address from a DHCP server on the network.

1. Network > Interfaces > Ethernet > **ethernet1/1** > **IPv4**: `DHCP Client`

> The `DHCP Client` setting
> allows the Firewall interface to receive a dynamic IP Address.
> IP address via DHCP, Firewall need to be configured to receive a dynamic IP Address.

![Screen Shot 2020-05-28 at 19.54.08](https://i.imgur.com/N6WeDbn.png)

1. Network > Interfaces > Ethernet > **ethernet1/1** > `Dynamic-DHCP Client link` under the IP Address field for ethernet1/1.
8. receive an IP Address of 203.0.113.51, obtained from the DHCP Server running on the VRouter between the Firewall and the External Network.

![Screen Shot 2020-05-28 at 22.50.15](https://i.imgur.com/PxluGx6.png)

---

## Lab 3: Configuring Virtual IP Addresses


1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-03`
4.  Click the Commit link located at the top-right of the web interface.

### 3.1 Configure a Virtual IP Address
configure a virtual IP address 192.168.20.1 on the Firewall.
- virtual IP address allows the Firewall to communicate with multiple IP networks from a single physical interface.

1. CMD: confirm nothing assigned with the IP address 192.168.20.1: `ping 192.168.20.1`

>  `Destination not unreachable` and `possibly Request timed out`: the Client cannot reach anyone at that IP address.

> By default, the Client’s default gateway is 192.168.1.1, which is the Firewall `inside interface`.
> The responses come from 203.0.113.1: means the Firewall had no routes to the 192.168.20.0 network and forwarded those requests to its default gateway 203.0.113.1. From this information you can reasonably assume 192.168.20.1, for this lab environment, does not exist on the network.

![Screen Shot 2020-05-28 at 22.57.59](https://i.imgur.com/0ahzr02.png)

2. Network > Interfaces > Ethernet > `ethernet1/2`.
   - IPv4 tab: add `192.168.20.1/24` in the IP address field.
3. Click the Commit link
4. To confirm the Firewall is configured with IP address 192.168.20.1: `ping 192.168.20.1`
   - now receive replies from 192.168.20.1, the Firewall, even though it is on a different network because it is a virtual network on the Palo Alto interface.

![Screen Shot 2020-05-28 at 23.04.46](https://i.imgur.com/A29Gl63.png)

5. Network > Interfaces > Ethernet > `ethernet1/2`.
   - IPv4 tab: Click on `192.168.20.1/24` edit to `192.168.20.1/29`
6. Click on the Commit link


7. Start Menu > Control Panel > Network and Sharing Center > Network and Sharing Center > Change adapter settings > **internal**:
    - Properties: Internet Protocol Version 4(TCP/IPv4): Properties button
    - IP address: change it to `192.168.20.20`
    - Default gateway:` 192.168.20.1`
8.  CMD: `ping 192.168.20.1`

> The ping will fail
> because the Firewall’s virtual IP address, 192.168.20.1, has a network mask of /29 (255.255.255.248).
> The 192.168.20.0/29 network can only have an IP range of 192.168.20.1 – 192.168.20.6, 192.168.20.0 being the network address, 192.168.20.7 being the broadcast address.
> For the ping to succeed, the Client, configured for IP address of 192.168.20.20 does not fall in the IP range.

7. Start Menu > Control Panel > Network and Sharing Center > Network and Sharing Center > Change adapter settings > **internal**:
    - Properties: Internet Protocol Version 4(TCP/IPv4): Properties button
    - IP: `192.168.20.6`
    - Subnet mask: 255.255.255.248

8.  CMD: `ping 192.168.20.1`

> The ping will now respond because the Client is in the same network as the Firewall’s virtual IP address.

![Screen Shot 2020-05-28 at 23.14.25](https://i.imgur.com/R4pD8hC.png)


---

## Lab 4: Creating Packet Captures

1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-04`
4.  Click the Commit link located at the top-right of the web interface.

### 4.1 Create a Wireshark Packet Capture
using Wireshark

1. Start Menu > Wireshark: **internal** interface: Capture > Start.
2. Google Chrome: https://www.panlabs.com
3. Wait 5 to 10 seconds and reopen Wireshark and then click the Stop capturing
4. save the Wireshark packet capture: File > Save As > Desktop > packetcapture > Save

---

## Lab 5: Analyzing Packet Captures

1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-05`
4.  Click the Commit link located at the top-right of the web interface.

### 5.1 Create a Packet Capture within the Palo Alto Networks Firewall
create a packet capture on the Firewall and download it to the Client for inspection. This will capture all traffic going through the Firewall.

1.  Monitor > **Packet Capture**.
2.  **Configure Capturing**:
   - `Add`
   - Packet Capture Stage window**
       - Stage: `firewall`
       - File: `pcap-1`
   - `OFF` to turn Packet Capture on.
3.  New tab: https://www.panlabs.com
4.  Click on the firewall-a tab in the upper-left to switch back to the Firewall administrator page.
5.  **Configure Capturing**: click `ON` to turn Packet Capture off.
6.  Click the `Refresh` icon in the upper-right of the Firewall administrator page to refresh the Captured Files section.
7.  Captured Files section, download the packet capture by clicking the pcap-1 filename in the File Name column.
8.  Once the pcap-1 file downloads, click on the pcap-1.pcap file, it will open in Wireshark.


### 5.2 Analyze PCAP Files with Wireshark

1. The first protocol you will analyze is DNS.
   - Review packets 1 and 2.
2. Observe packet 1.
   - Source is the Client (192.168.1.20)
   - Destination is 192.168.50.10.

> The Client is configured to use 127.0.0.1 as its DNS server.
> In this lab environment, the Client is running its own DNS server with the ability to forward requests to 192.168.50.10. This is the DMZ server, which is also running a DNS server.
> The Info column: a Standard query asking for the A record for www.panlabs.com.
> used Google Chrome to navigate to https://www.panlabs.com. The first step the Client does is to attempt to resolve www.panlabs.com to an IP address.

3. Observe packet 2.
   - the Source is the DMZ Server (192.168.50.10)
   - Destination is 192.168.1.20.

> is a Standard query response indicating the A record for www.panlabs.com has an IP address of 192.168.50.10.
> That is the DMZ server, which is also running a Web server hosting www.panlabs.com.
> Now that the Client knows the IP address of the original request, it can begin the request for a 3-way TCP handshake.

4. Review packets 3, 4, and 5. `TCP 3-way handshake`.

![Screen Shot 2020-05-28 at 23.43.39](https://i.imgur.com/tL3ee2o.png)


5. Observe packet 3.
   - the Source (the Client, 192.168.1.20) sends a TCP packet with the `flags SYN`, `ECN`, and `CWR` set in the header, to the Destination (the DMZ server, 192.168.50.10).
   - This establishes a `SYN (SYNchronize) packet` along with window size information.


6. Observe packet 4.
   - the Source (the DMZ server, 192.168.50.10) sends a TCP packet with the flags `SYN` and `ACK` set in the header, to the Destination (the Client, 192.168.1.20).
   - This establishes a `SYN-ACK (SYNchronize-ACKnowledgement) packet`. The DMZ server acknowledges the Client and sends back its own synchronization packet.

7. Observe packet 5.
   - the Source (the Client, 192.168.1.20) sends a TCP packet with the flag `ACK` set in the header, to the Destination (the DMZ server, 192.168.50.10).
   - This establishes an `ACK (ACKnowledgement) packet`. The Client acknowledges the DMZ server.
   - The Client and the DMZ server may begin communicating over TCP.


8.  Packets 3 – 45 represent a TCP Stream. When put together this represents the website, https://www.panlabs.com that you visited.
   - right-click on packet 3 and select Follow > `TCP Stream`.
   - Wireshark will assemble the packets associated with this TCP stream.

9. TCP Stream.
    - Notice the assembled packets represent the HTML website you visited.

![Screen Shot 2020-05-28 at 23.48.13](https://i.imgur.com/sVaAROG.png)


---

## Lab 6: Using the Application Command Center to Find Threats

1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-06`
4.  Click the Commit link located at the top-right of the web interface.

### 6.1 Generate Malware Traffic to the Firewall
generate malware traffic to the Firewall using a script that is replaying previously captured traffic.

1. PuTTY
2. **Saved Sessions**: traffic-generator > Load > Open.

![Screen Shot 2020-05-28 at 23.56.10](https://i.imgur.com/l9Sojy0.png)

3. At the prompt
   - passwd: `Pal0Alt0` and press Enter.
   - `sh /tg/malware.sh`
   - Wait 10 minutes to let the script generate malware traffic.

4. The script will generate test malware traffic to the Firewall so that you can see malware traffic in the Firewall. You will see the following output when the script has generated the traffic.

![Screen Shot 2020-05-29 at 00.09.47](https://i.imgur.com/5VmXPQf.png)


### 6.2 Find Malware Threat in the Application Command Center
In this section, you will review Threat Activity and Blocked Activity in the Application Command Center.
1. FW > ACC > **Threat Activity**.

![Screen Shot 2020-05-29 at 00.11.01](https://i.imgur.com/imtvxU6.png)

1. FW > ACC > **Blocked Activity**
    - the Blocked User Activity section.

![Screen Shot 2020-05-29 at 00.11.56](https://i.imgur.com/auRLuGH.png)


---

## Lab 7: Analyzing Firewall Logs


1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-07`
4.  Click the Commit link located at the top-right of the web interface.

### 7.1 Generate Traffic to the Firewall

same as lab 6.

`sh /tg/malware.sh`

### 7.2 Review Traffic in the Firewall Logs
1. Monitor > Logs > Traffic.
2. You will see traffic from the firewall.
3. Look under the Application column and find traffic that is categorized as `webbrowsing`

![Screen Shot 2020-05-29 at 00.37.05](https://i.imgur.com/PYeFlFy.png)

4. Click on the Magnifying Glass icon on the left to view the traffic.

![Screen Shot 2020-05-29 at 00.38.05](https://i.imgur.com/kfzdZEP.png)

---

## Lab 8: Protecting Sensitive Data

Data Pattern ->  Data Filtering Profile -> Security Policy

1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-08`
4.  Click the Commit link located at the top-right of the web interface.

### 8.1 Create a New Data Pattern
`Data pattern` objects detect the information that needs to be filtered.
- Three types of data patterns are utilized for scanning sensitive information.
  - `Predefined patterns` are preset patterns used to detect Social Security and credit card numbers.
  - `Regular expressions` are used to create custom data patterns.
  - `File properties` are used to scan files for specific file properties and values.

For this lab, you will use predefined patterns.

1. Objects > Custom Objects > Data Patterns > Add.
2. **Data Patterns window**, type SSN in the
   - Name: `SSN`
   - Description: `Sensitive`
   - Pattern: `Predefined Pattern`
   - Add.
     - select Social Security Numbers
     - select Social Security Numbers (without dash separator).

![Screen Shot 2020-05-29 at 00.47.51](https://i.imgur.com/fgYZuB1.png)

### 8.2 Create a Data Filtering Security Profile
`Data Filtering Security Profiles` prevent sensitive information such as credit card and Social Security numbers from leaving a secured network.

1. Objects > Security Profiles > Data Filtering > Add.
2. **Data Filtering Profile window**
   - Name: `SSNs`
   - Description: `Protecting Sensitive Data`
   - check: Data Capture.

![Screen Shot 2020-05-29 at 00.50.16](https://i.imgur.com/iTNFJFY.png)

3. **Data Filtering Profile window**, click Add.
   - Data Pattern: `SSN`
   - Alert and Block Threshold: `1`
   - Log Severity: `high`

![Screen Shot 2020-05-29 at 00.52.10](https://i.imgur.com/Q6lBINY.png)


### 8.3 Apply the Data Filtering Profile to the Security Policy

1. Policies > Security > `Allow-Inside-DMZ`.
2. **Security Policy Rule window**
   - Action: `Allow`
   - Profile Type: `Profiles`
   - Data Filtering: `SSNs`

![Screen Shot 2020-05-29 at 00.56.01](https://i.imgur.com/01ohjtR.png)

3. Click the Commit link located at the top-right of the web interface.


### 8.4 Create a Text File with Fake Social Security Numbers
1. Notepad

![Screen Shot 2020-05-29 at 00.57.36](https://i.imgur.com/uysx4Yh.png)

2. Save As > `SSN` in Desktop


### 8.5 Monitor Sensitive Data in the Palo Alto Networks Firewall

1. Internet Explorer: https://192.168.50.10/fileupload
2. Upload Files > SSN.txt file.
3. FW > Monitor > Logs > Data Filtering
4. Notice that the SSN.txt was blocked by the SSN Data Filtering Profile.
5. Click on the Detailed Log View button.
6. On the Detailed Log View window, click on the second row.
   - the Application web-browsing was reset and the Severity was high as applied by the Data Security Policy.
   - The **General** section show the Application, Protocol, and the Category it was assigned.
   - The **Source** section: identify where the source originated
   - the **Designation** section: identify where the file was designated.

![Screen Shot 2020-05-29 at 01.07.57](https://i.imgur.com/QdCUrI2.png)

123.txt `block`
123-22-2345

233.txt `not block`
uuu 123-22-2345 is


---

## Lab 9: Preventing Threats from the Internet with File Blocking

1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-09`
4.  Click the Commit link located at the top-right of the web interface.

### 9.1 Create a File Blocking Security Profile
create a File Blocking Security Profile to block PDF files.
1. Objects > Security Profiles > **File Blocking** > Add.

![Screen Shot 2020-05-29 at 01.24.28](https://i.imgur.com/zgz0w8W.png)

### 9.2 Apply the File Blocking Profile to a Security Policy
1. Policies > Security and click on Allow-Inside-DMZ.

![Screen Shot 2020-05-29 at 01.25.18](https://i.imgur.com/RgyvaKm.png)

2. In the Commit window, click Commit to proceed with committing the changes.

### 9.3 Test the File Blocking Profile
1. New tab: https://192.168.50.10/pan-os.pdf
2. Notice the File Transfer was blocked via the File Blocking Profile that was created in a previous section.
3. Monitor > Logs > Data Filtering.

![Screen Shot 2020-05-29 at 01.26.47](https://i.imgur.com/yWKopEE.png)

4. pan-os.pdf is being logged.
    - Action: `deny`

---

## Lab 10: Log Forwarding to Linux (Setup syslog to DMZ Server)

1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-10`
4.  Click the Commit link located at the top-right of the web interface.


> - setup syslog Server
> - `Log Forwarding` add syslog Server under syslog
> - `Log Settings: System, Configuration, User-ID, HIP Match sections` add syslog Server under syslog
> - `Security Policy Rule` add Log Forwarding: `Syslog Server`
> - finish


### 10.1 Configure Syslog Monitoring via Palo Alto FW
In this section, you will configure the Palo Alto Firewall for `Syslog monitoring`.
- `Syslog` is a standard log transport mechanism that enables the aggregation of log data from different network devices—such as routers, firewalls, printers—from different vendors into a central repository for archiving, analysis, and reporting.
- Palo Alto Networks firewalls can `forward every type of log they generate to an external Syslog server`.
  - You can use TCP or SSL for reliable and secure log forwarding, or UDP for non-secure forwarding

1. Device > Server Profiles > **Syslog**: Add.
   - **Syslog Server Profile** window

![Screen Shot 2020-05-29 at 13.29.35](https://i.imgur.com/eLkN8OK.png)


2. Objects > **Log Forwarding**: Add
   - **Log Forwarding Profile Match List**

![Screen Shot 2020-05-29 at 13.31.23](https://i.imgur.com/7X5ShWZ.png)

![Screen Shot 2020-05-29 at 13.35.30](https://i.imgur.com/XkklNwq.png)



3. Device > **Log Settings** > System section: Add
   - **Log Settings - System**

![Screen Shot 2020-05-29 at 13.36.40](https://i.imgur.com/Zh2mSeX.png)


4. Repeat step 3 by clicking Add for Configuration, User-ID, and HIP Match sections.


5. Policies > Security > Allow-Any.
   - **Security Policy Rule**
   - Actions tab
   - checkbox: Log at Session Start
   - Log Forwarding: `Syslog Server`

![Screen Shot 2020-05-29 at 13.41.16](https://i.imgur.com/ukBeT1g.png)


6.  Click the Commit link located at the top-right of the web interface.

7.  In the Commit window, click Commit to proceed with committing the changes.


### 10.2 Verify Syslog Forwarding
connect to the DMZ server and verify syslogs are being forwarded.
1.  CMD: ping 192.168.50.10

2.  PuTTy: `traffic-generator` > Load button
   - type Pal0Alt0
   - sh /tg/traffic.sh

![Screen Shot 2020-05-29 at 14.23.00](https://i.imgur.com/kfXVZRF.png)

3.  second PuTTY session: To verify traffic for the Firewall, `traffic-generator` > Load button
   - type Pal0Alt0
   - tail -f /var/log/messages

![Screen Shot 2020-05-29 at 14.23.13](https://i.imgur.com/dioYUQN.jpg)


---

## Lab 11: Backing up Firewall Logs

1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-cg-lab-11`
4.  Click the Commit link located at the top-right of the web interface.


### 11.1 Back Up Firewall Logs
export Firewall logs to another location. Exporting firewall logs to an `FTP Server`
- beneficial for keeping logs in the event that the logs are overwritten or an unforeseen event happens to the Firewall and the logs cannot be retrieved.


1.  Device > **Scheduled Log Export** > Add.
    - **Scheduled Log Export** window

![Screen Shot 2020-05-29 at 14.46.41](https://i.imgur.com/AYTntaG.png)


1. Click the Commit link

2. Monitor > Logs > System.

3. Change the Refresh dropbox to 10 Seconds at the top-right.

4. a log entry that shows a completed `log export of the traffic log to the FTP server`.

![Screen Shot 2020-05-29 at 14.49.40](https://i.imgur.com/wWhPRgN.png)



















.
