

## Layer 3: The Network Layer (Packets)

Examples of devices: routers and multilayer switches.

Handles Logical addressing and routing od traffic: uses logical addressing to make forwarding decisions.
- The lowest layer for protocol software.
- A variety of routed protocols (like, AppleTalk and IPX) have their own logical addressing schemes
- the most widely deployed routed protocol is Internet Protocol (IP).
  - IP address: nonpersistent address assigned via software and changed as needed.
  - IPv4 address: 32 bits.
  - IPv6 address: 128 bits.
- A less popular Layer 3 protocol is Novell’s Internetwork Packet Exchange (IPX), which has its own format for Layer 3 addressing. (Although Novell developed, most modern Novell networks use IP as their Layer 3 protocol.)

Switching:
- Not only in Layer 2, but also exists at Layer 3.
- making decisions about how data should be forwarded, 3 common switching techniques:
- Packet switching:
  - With packet switching, a data stream is divided into packets.
  - Each packet has a Layer 3 header, which includes a source and destination Layer 3 address.
- Circuit switching:
  - Circuit switching dynamically brings up a dedicated communication link between two parties in order for those parties to communicate.
  - Example: making a phone call from your home to a business, traditional landline servicing your phone, the telephone company’s switching equipment interconnects your home phone with the phone system of the business you’re calling. This interconnection (circuit) only exists for the duration of the phone call.
- Message switching:
  - usually not suit for real-time applications, because of the delay involved.
  - a data stream is divided into messages.
  - Each message is tagged with a destination address, and the messages travel from one network device to another network device on the way to their destination.
  - Because these devices might briefly store the messages before forwarding them, a network using message switching sometimes called a store-and-forward network.

Metaphorically, you could visualize message switching like routing an e-mail message, where the e-mail message might be briefly stored on an e-mail server before being forwarded to the recipient.

Router:
- discovery and selection:
  - how to reach various network addresses.
  - maintain a routing table, have its routing table populated via:
  - manual configuration (entering static routes),
  - dynamic routing protocol (example, RIP, OSPF, or EIGRP),
  - or simply directly connected to certain networks.
  - send packet based on the packet’s destination network address.
- Connection services:
  - provided connection services for flow control and error control (like the data link layer)
  - Can improve the communication reliability, even the data link’s LLC sublayer not perform connection services.

The following functions are performed by connection services at the network layer:
- Flow control (congestion control):
  - Helps prevent a sender send data faster than the receiver is capable to receive the data.
- Packet reordering:
  - Allows packets to be placed in the appropriate sequence as they are sent to the receiver. (This might be necessary, because some networks support load-balancing, where multiple links are used to send packets between two devices. Because multiple links are used, packets might arrive out of order.)
