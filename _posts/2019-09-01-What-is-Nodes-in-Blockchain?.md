---
title: What are Nodes in Blockchian?
author: Dhruv Doshi
date: 2019-09-01 08:33:00 +0800
categories: [Blockchain, Cryptocurrency]
tags: [Blockchain]
math: true
mermaid: true
# image:
#   path: /blogs/Blockchain.jpg
#   width: 800
#   height: 500
#   alt: Representation of Blockchain through Image.
  
---

A Node is a part of cryptocurrency needed to make most of the popular tokens like Bitcoin or Dogecoin function. It's a fundamental part of the blockchain network, which is the `decentralised ledger that is used to maintain a cryptocurrency.`

The explanation can vary depending on the protocol. For example, a resident network may comprise a file server, three laptops and a fax machine. In this case, the network has five nodes, each equipped with a unique MAC address to identify them.

`What is a node in blockchain?`<br>
The term *“node”* is being used chiefly about blockchain, a decentralized digital ledger that records all cryptocurrency transactions and `makes the information available to everyone via a connected device.` What this means is that every transaction has to be chronologically recorded and distributed to a series of connected devices. These devices are called nodes. These nodes communicate within the network and transfer information about transactions and new blocks.

It is a critical component of the blockchain infrastructure. It helps maintain the `security and integrity of the network.` A blockchain node's main purpose is to verify each batch of network transactions, called blocks. Each node is distinguished from others by a unique identifier.

`What do nodes do?`<br>
When a miner attempts to add a new block of transactions to the blockchain, it broadcasts the block to all the nodes on the network. Based on the block’s legitimacy (validity of signature and transactions), `nodes can accept or reject the block.` When a node accepts a new block of transactions, it saves and stores it on top of the blocks it already has stored. In short, here is what nodes do:

1. Nodes check if a block of transactions is valid and accept or reject it.
2. Nodes save and store blocks of transactions (storing blockchain transaction history).
3. Nodes broadcast and spread this transaction history to other nodes that may need to synchronize with the blockchain (need to be updated on transaction history).

`What are the types of nodes?`<br>
There are basically two types of nodes: `full nodes and lightweight nodes.`

 - `Full nodes support and provide security` to the network. These nodes download a blockchain's entire history to observe and enforce its rules.

 - `Each user in the network is a lightweight node.` The lightweight node has to connect to a full node to be able to participate.

Many volunteers run full Bitcoin nodes to help the Bitcoin ecosystem. As of now, there are roughly `12,130 public nodes` running on the Bitcoin network. Other than the public nodes, there are many hidden nodes (non-listening nodes). These nodes usually run behind a firewall.

`Miners' nodes`<br>
There is also a third type of node: Miner nodes. The term “Bitcoin miners” has now become familiar. These miners are classified as nodes. The miner may work alone (solo miner) or in groups (pool miner). A solo miner uses his full node. Only the administrator can run a full node in a mining pool, which can be referred to as a pool miner's full node.

`The difference between a miner and a node`<br>
A miner must run a full node to select valid transactions to form a new block. Without a complete node, it cannot determine what proposed transactions are valid according to the current blockchain’s transaction history (if all balances involved in the transactions are sufficient to perform the proposed transactions) because it does not have access to the entire blockchain history. Therefore, a miner is always also a full node. A node, however, is not necessarily simultaneously a miner. A device can run a full node by receiving, storing, and broadcasting all transaction data (much like a server) without creating new blocks of transactions. In this case, it functions more like a passing point with a directory, whereas a miner is the same but simultaneously tries to create new blocks of transactions.

`Listening nodes (supernodes)`<br>
Moreover, finally, a sub-category called listening nodes. A listening node, essentially, is a publicly visible full node. It communicates with any node that decides to establish a connection with it. A reliable super node typically runs simultaneously, transmitting blockchain history and transaction data to multiple nodes.
