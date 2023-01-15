---
title: Time Stamping in Blockchain and Cryptocurrencies?
author: Dhruv Doshi
date: 2019-09-02 08:33:00 +0800
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

Trusted timestamping is the `process of securely keeping track of the creation and modification time of a document.` Security here means that no one—not even the owner of the document—should be able to change it once it has been recorded, provided that the timestamp's integrity is never compromised.

Decentralization is one of the fundamental aspects of technology blockchain, And of course, that implies that anyone from any part of the world can be added to the network and can operate in it. This, in turn, causes there to be no universal time code. This is because we can connect from any time zone. The timestamp is a timestamp, which is calculated according to different parameters.

The temporal parameter, or timestamp, is based on an ` quick adjustment that uses a median of the timestamps returned by all nodes of the network.` This is due to the decentralized form and seeks to keep the nodes of the network as well synchronized as possible.

We must also bear in mind that the timestamps of the blocks are not exact. This is because they do not necessarily have to be in order. However, they still offer a relative precision of between one and two hours, which gives a margin of validity. Basically, `all the nodes are connected to the same time slot.` For this, the reference is taken UTC-0 (London local time), where UTC It is in Spanish Coordinated Universal Time. From this, the network nodes coordinate the time in which they work. After storing this data, the local node calculates the displacement time between the UTC strip and the local time.

This adjusts between the time of the local node with the displacement of all the nodes connected to the network. `This allows the network time to be adjusted constantly`. This avoids manipulation and usually does with little time variations concerning the time slot. This is done because there may be many hourly rates and repetitions, and other problems could occur. Therefore, a universal timestamp creation system was developed for all nodes. This system considers the jet lag that could exist between the nodes.

Implementing a timestamp makes the block it is `impossible to be repeated in the future`, since, in addition to the time, the date of creation of the block is also stored, therefore, there is no possibility that it will be repeated hash that happened a week, two months ago, or a year ago.

<br>

`What is blockchain timestamp used for?`</br>
One of the primary uses of a timestamp is to establish the parameters of the mining process. This is because these timestamps allow nodes to `correctly adjust the mining difficulty to be used for each block generation period.` Timestamps help the network determine how long it takes to extract blocks for a certain period and adjust the mining difficulty parameter.

This, of course, can open the door for miners to manipulate time to lessen the difficulty. Nevertheless, Satoshi Nakamoto foresaw this and programmed the network so that nodes ignore blocks that are outside a ` specific time range based on their own internal clock time.` As a result, if a miner tried to do this, he would lose all his mining work.

On the other hand, in the whitepaper From Bitcoin, Nakamoto explains that another functionality of the timestamp is to create a mechanism to `avoid double-spending.` In this regard, Nakamoto wrote the following:

>> `*For our purposes, the last transaction is what counts, so we won't mind other subsequent double-spending attempts.*`
