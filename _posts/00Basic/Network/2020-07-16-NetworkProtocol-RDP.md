---
title: Network protocol - RDP
date: 2020-07-16 11:11:11 -0400
categories: [00Basic, Network]
tags: [SSH]
math: true
image:
---

- [Network protocol - RDP](#network-protocol---rdp)
  - [basic](#basic)
  - [RDP Breakdown](#rdp-breakdown)

---

# Network protocol - RDP

---

## basic


Remote Desktop Protocol is used within Windows OS. It is used to access physical or virtual servers.
- Unlike SSH, RDP has a graphical user interface.
- RDP is designed to connect over the internet to another machine.
- Its functionality is to transmit data from the output device, e.g., monitor screen/display, mouse, and keyboard logs, to the input device (local machine).


---

## RDP Breakdown

- The RDP protocol stack consists of
  - ISO Transport Service (TKPT),
  - connection-Oriented Transport Protocol (X.224),
  - and the Multipoint Communication Service (T.125 MCS).

- TPKT enables the exchange of information units,
- while X.224 provides the connection-mode transport service used in the initial connection request and response within the RDP.

- The sending and receiving process is quite similar to that of the Seven-Layer OSI model, where we see data from an application or service being transmitted and passed through the protocol stack.

![RDP-Connection-3-1536x1536](https://i.imgur.com/5zTKODg.png)




The RDP connection goes through several stages between the Client and the Server.

1. Connection – occurs by using the `X.224 Connectio`n request PDU (protocol data unit). The Client sends a X.224 connection request to the Server where the packet sent contains RDP Negotiation Request and the security protocols that the Client supports, i.e., RSA RC4 encryption, TLS, RDSTLS, or CredSSP. Once the Server confirms the connection, it chooses the security protocol from the Client’s protocol selection. All subsequent data is wrapped in X.224 Data PDU from this point.

2. Basic Setting Exchange – After connecting with the Server, an exchange of basic settings occurs between the Client and the Server using MSC Connect Initial and Connect Response PDU. Information such as Core Data, Security Data, and Network Data fall under the basic settings category.
Core Data – Version of RDP, keyboard information, hostname, client software information, Desktop resolution, etc.
Security Data – encryption methods, size of session keys, server certificate, and Server random for session key creation.
Network Data – holds information regarding the allocated virtual channels and requested channels, along with details of the request type, IDs, responses, etc.

3. Channel Connection – Once basic settings have been exchanged and a list of virtual channels has been established, the following stage makes sure that all channels connection occurs.
MCS Erect Domain Request
MCS Attach User Request
MCS Attach User Confirm
MSC Channel Join Request and Confirmations

4. Security Commencement
The security commencement stage initiates the Security Exchange PDU, which contains the random encrypted public key from the Client with the Server. Both use random numbers to create encrypted session keys, where the subsequent RDP traffic can be encrypted in further stages.

5. Security Settings Exchange
Enables encrypted client Info PDU data to be sent from the Client to the Server. The data consists of user domain, username, password, working directory, compression types, etc.

6. Licensing
Licensing stage allows for authorized users to connect to the terminal server.

7. Capabilities Exchange
The Server sends Demand Active PDU where supported capabilities are sent to the Client. This usually covers OS version, compression input, bitmap codecs, virtual channels, etc. Then the Server can send a Monitor Layout PDU to understand the display monitors on the Server. At this stage, the Client responds with a Confirm Active PDU.

8. Connection Finalization
The second to last stage finalizes the connection between the Client and the Server. At this stage, the Client sends specific PDUs to the Server and vice versa.
The following PDUS are:
Client/Server synchronize PDU – syncs user identifiers between Client and Server
Client/Server Control PDU: send the following PDU to indicate the shared control over a session.
Client Control PDU (request/grant Control): The Client sends a request for control while the Server grants the request.
Persistent Key List PDU/PDUs: an optional stage where the Client sends a list of keys, each identifying the cached bitmap, allowing more graphical output from the Server to the Client.

9. Data Exchange
After all the stages and PDU exchanges, the primary data is sent between the Client and the Server, where the Server will input data and graphic data. RDP connection is now ready to be used by the end-user.





.
