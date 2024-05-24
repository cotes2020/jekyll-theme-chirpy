---
layout: post
title: OSI Layers
---
# Demystifying Network Communication: OSI Layers and TCP/IP Model Explained

When you open your web browser and type "google.com," have you ever wondered what happens behind the scenes to establish a connection and load the webpage? In this article, we'll take you through the intricate process step by step, using both the OSI (Open Systems Interconnection) model and the TCP/IP model as our guide.

## OSI Model and a Mnemonic: "Please Do Not Threw Sausage Pizza Away"

To help remember the OSI layers in order, you can use the mnemonic: **"Please Do Not Threw Sausage Pizza Away."** Each letter corresponds to a layer: Physical, Data Link, Network, Transport, Session, Presentation, and Application.

Now, let's delve into the detailed steps of the OSI model:

1. **Physical Layer:**
   - The journey begins with the physical connection between your device and the network medium, such as an Ethernet cable or Wi-Fi.

2. **Data Link Layer:**
   - Data is divided into frames, each carrying a header and a trailer with source and destination MAC addresses.
   - Your device's network interface card (NIC) adds its MAC address to the source header.

3. **Network Layer:**
   - Your device's IP address is added to the packet header.
   - Routing protocols determine the appropriate next-hop router's IP address.

4. **Transport Layer:**
   - A communication session is established using TCP (Transmission Control Protocol).
   - TCP segments the data and assigns sequence numbers for reassembly.

5. **Session Layer:**
   - Establishes, maintains, and manages the session between your device and the server.
   - Handles session synchronization, checkpointing, and recovery.

6. **Presentation Layer:**
   - Data is translated, encrypted, or compressed if necessary.
   - Ensures data compatibility between sender and receiver.

7. **Application Layer:**
   - Your web browser initiates an HTTP (Hypertext Transfer Protocol) request to the Google server.
   - The request includes the URL "google.com."

# TCP/IP Model, Scapy Example, Handling Responses, and References
## TCP/IP Model Steps

1. **Network Interface Layer (OSI Physical and Data Link Layers):**
   - Your device's network hardware (NIC) prepares data frames for transmission over the physical medium.

2. **Internet Layer (OSI Network Layer):**
   - The data frame is encapsulated in an IP packet.
   - Source and destination IP addresses are added to the IP header.

3. **Transport Layer (OSI Transport Layer):**
   - TCP segments are created and assigned sequence numbers.
   - Source and destination port numbers are added to the TCP header.

4. **Application Layer (OSI Application, Presentation, and Session Layers):**
   - Your web browser's application initiates an HTTP GET request for "google.com."

5. **Internet Layer (OSI Network Layer, Continued):**
   - The IP packet is routed to the destination IP address using routers.
   - Routers forward the packet based on routing tables.

6. **Transport Layer (OSI Transport Layer, Continued):**
   - TCP ensures reliable data delivery through acknowledgments and retransmissions.

7. **Network Interface Layer (OSI Data Link Layer, Continued):**
   - Data frames are transmitted through the network medium, following Ethernet or Wi-Fi protocols.

8. **Physical Layer (OSI Physical Layer):**
   - Data is transmitted as electrical signals over the physical medium.

As data traverses these layers, it undergoes encapsulation and decapsulation, with each layer adding or removing specific headers. This meticulous process guarantees accurate transmission and reception across the network. Ultimately, your browser successfully connects to "google.com," and the webpage loads seamlessly.

To illustrate the encapsulation process further, let's consider an example using Scapy, a Python library for packet manipulation.

## Scapy Example: Network and Transport Layers

```python
from scapy.layers.inet import IP, TCP

# Create an IP packet
ip_packet = IP(src="192.168.1.2", dst="8.8.8.8")

# Create a TCP segment
tcp_segment = TCP(sport=12345, dport=80)

In this example, we've demonstrated the construction of an IP packet and a TCP segment using Scapy. The IP packet encapsulates the TCP segment, replicating the network and transport layers' encapsulation process.

As the process continues, each layer contributes to the successful transmission of data, demonstrating the complexity and elegance of network communication.
# Handling Responses
Network Interface Layer (OSI Data Link Layer, Continued):

The destination device receives the data frame and extracts the encapsulated IP packet.
Internet Layer (OSI Network Layer, Continued):

The IP packet is examined, and the destination IP address is identified.
Transport Layer (OSI Transport Layer, Continued):

TCP decodes the incoming data, acknowledging the receipt of segments.

Application Layer (OSI Application, Presentation, and Session Layers):

The web browser processes the HTTP response from the Google server.
The response travels through the layers in reverse, ensuring that the data is accurately received, processed, and presented to you.

# Responses from Routers
When the IP packet is routed to the destination IP address, it encounters routers along the way. Each router examines the packet's headers and makes forwarding decisions based on routing tables.

To illustrate the responses received after sending packets to routers, let's consider a Scapy example:
```
from scapy.layers.inet import IP, ICMP
from scapy.sendrecv import sr1

# Create an IP packet with a destination IP address of a router
ip_packet = IP(src="192.168.1.2", dst="192.168.1.1")

# Create an ICMP request (ping) packet
icmp_packet = ICMP()

# Send the packet and receive a response
response = sr1(ip_packet / icmp_packet)
```

In this example, we send an ICMP ping packet to a router with the destination IP address "192.168.1.1." The sr1 function sends the packet and waits for a response. The response may include information about the router's status, latency, and other details.

By following these models, understanding responses, and exploring the interactions with routers, you've gained insights into the intricate world of network communication. The next time you browse the web, you'll have a deeper appreciation for the complex processes that enable seamless connectivity and access to online resources.

# References
OSI Model - (Wikipedia)[https://en.m.wikipedia.org/wiki/OSI_model]
Internet Protocol Suite - (Wikipedia)[https://en.m.wikipedia.org/wiki/Internet_protocol_suite]
Mahmoud Swelam, Security Engineer. LinkedIn Profile: (Mahmoud Swelam)[https://www.linkedin.com/in/mahmoud-swelam-36b08725a]
