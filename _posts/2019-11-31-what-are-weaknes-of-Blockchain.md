---
title: What is the weakness of Blockchain?? (Everything isn't perfect!!)
author: Dhruv Doshi
date: 2019-11-31 11:33:00 +0800
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

`1. There is no customer protection on the blockchain` Blockchain technology operates as a push-based settlement system. This means the individual holds power over the resource they want to verify on the blockchain. This could be in cryptos, certificate authentication, land titles, etc. The problem with this is if a transaction goes sour after it has already been verified on the blockchain, the only feasible way of returning the transaction is if the parties agree to reverse it. Using a  centralized system like a bank. However, there is a procedure in place to be able to dispute trades after they are complete. Some trade technologies that settle on a blockchain have used an arbiter system to fix this problem, an example of this is the Open Bazaar P2P  trade network. This way trade occurs between two people and one impartial moderator.

`2. Settlement on a blockchain is slow` A  cost of settling a transaction on the blockchain is that all the nodes in the network need to come to an agreement that the transaction is valid. This is a far slower process than having a bank verify your transaction in an instant. Transactions can be made instantaneously, however until the block in which the transaction is inserted has been verified, it is classified as untrustworthy. In the time between a lodged transaction is made and when the block settles, a bad actor could launch fraudulent transactions to trick the network into what is known as double spend. A  very exciting upcoming technology that could solve this problem is the lightning network. This solution acts as layer 2 of blockchain technology; it can be applied to any public blockchain. It will enable instantly verified transactions for a fraction of the cost of today’s settlement.

`3. Miners can be selfish `The mining process on the blockchain is an innovation that uses game theory economics to incentivize people to commit computer power for securing the network for a profit. The downside of this is generally miners won’t care about settling as many transactions as possible; they will make the most money by finding and verifying a block in the fastest way possible. This leads to a problem of miners finding empty blocks and validating. There is also another problem known as Selfish Mining, which is a situation where a miner or mining pool finds and validates a block and does not publish and distribute a valid solution to the rest of the network.

`4. The growing blockchain size ` With  every new block, a blockchain grows. This can be an issue because each node that is validating the network needs to store the entire history of the blockchain to be a participant. This is a hard enough problem with the bitcoin blockchain where the transaction size is only a  few bytes, the total blockchain size as of January 2017 is 98GB. Given that at the same time in 2016 the size was 50GB, and the use of the blockchain is continuing to increase, this is a growing concern. One of the biggest debates in the bitcoin space is if the block size should be increased. If a blockchain has bigger blocks the blockchain size will increase faster, thus weeding out the solo miners eventually. This is a big issue because the health of a blockchain network is partially dependent on the number of nodes in the network, and the spread of those nodes across the world. The counterargument for this issue is that with sufficient advancement of technology hard disk space will be very cheap in the future and will stay ahead of the blockchain size. The debate is ongoing.

`5. Eventually, a settlement on the blockchain will not be cheap ` On any public blockchain, space in a block is a finite resource.  Necessarily as the network is utilized more the amount of transactions that will want to settle in a block will exceed the storage capacity.  Public blockchain networks have a solution built-in for this which is that transactions with a higher miner fee attached will get precedence to be included in a block. This makes sense because the miners want to maximize their profit so that they will include transactions with the highest fees first. This is not a bug, but a feature. If it were free to settle on the blockchain, there would be far too many ways of attacking the blocks with dust transactions and clogging up the network. Originally the bitcoin blockchain had no block size limit; this was eventually set to  1MB to avoid a Sybil Attack on the network. All of these problems have potential solutions that can be implemented as a  fix. In my view, blockchains will eventually have layers of centralization like the lightning network. However, this is not a bad thing so long as there is a sufficient amount of encryption to protect the privacy of the people who want to use the centralized layers of the network.

