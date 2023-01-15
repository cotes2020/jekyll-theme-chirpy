---
title: What is Automatic Swaps in Blockchain?
author: Dhruv Doshi
date: 2019-09-08 11:33:00 +0800
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

An atomic swap is a smart contract technology that enables the exchange of one cryptocurrency for another without using centralized intermediaries, such as exchanges.

Atomic swaps can take place directly between blockchains of different cryptocurrencies, or they can be conducted off-chain, away from the main blockchain. They first came into prominence in September 2017, when an atomic swap between Decred and Litecoin was conducted. 

Since then, other startups and decentralized exchanges have allowed users the same facility. For example, Lightning Labs, a startup that uses bitcoin’s lightning network for transactions, has conducted off-chain swaps using the technology.

Cryptocurrencies and decentralized exchanges, such as 0x and Altcoin.io, have also incorporated the technology. 


**`How do atomic swaps work?`**<br>
Atomic swap protocols are designed in a way that prevents any of the involved parties from cheating. To understand how they work, let’s imagine that Alice wants to trade her Litecoins (LTC) for Bob’s Bitcoins (BTC).

First, Alice deposits her LTC into a contract address that acts like a safe. When this safe is created, Alice also generates a key to access it. She then shares a cryptographic hash of this key with Bob. Note that Bob can’t access the LTC yet because he only has the hash of the key and not the key itself.
Next, Bob uses the hash provided by Alice to create another safe contract address, in which he deposits his BTC. To claim the BTC, Alice is required to use that same key and, by doing so, she reveals it to Bob (thanks to a special function called hash lock). This means that as soon as Alice claims the BTC, Bob can claim the LTC and the swap is complete.
The term ‘atomic’ relates to the fact that these transactions either happen entirely or not at all. If any of the parties give up or fail to do what they are supposed to, the contract is cancelled, and the funds are automatically returned to their owners.

Atomic swaps can happen in two different ways: on-chain and off-chain. On-chain atomic swaps happen on either of the currency’s networks (in this case, either the Bitcoin or Litecoin blockchain). Off-chain atomic swaps, on the other hand, take place on a secondary layer. This kind of atomic swap is usually based on bidirectional payment channels, similar to the ones used in the Lightning Network.
Technically speaking, most of these trustless trading systems are based on smart contracts that use multi-signatures and Hash Timelock Contracts (HTLC).

*`Hash Timelock Contracts (HTLC)`*<br>
While Hash Timelock Contracts (HTLC) are an important part of the Bitcoin Lightning Network, they are also one of the key components that makes atomic swaps possible. As the name suggests, they are based on two key functions: a hash lock and a timelock.
A hash lock is what prevents funds from being spent unless a piece of data is revealed (Alice’s key in the previous example). Timelock is a function that ensures the contract can only be executed within a predefined timeframe. Consequently, the use of HTLCs removes the need for trust because they create a specific set of rules that prevent atomic swaps from executing partially.


*`Advantages`*<br>
The biggest advantages of atomic swaps are all related to their decentralized nature. By removing the need for a centralized exchange or any other kind of mediator, cross-chain swaps can be executed by two (or more) parties without requiring them to trust each other. There is also an increased level of security because users don’t need to give their funds to a centralized exchange or third party. Instead, the trades can happen directly from users’ wallets.
Also, this form of peer-to-peer trading has much lower operational costs as trading fees are either very low or absent. Lastly, atomic swaps make it possible for trades to happen very quickly, with higher degrees of interoperability. In other words, altcoins can be swapped directly without making use of Bitcoin or Ethereum as an intermediary coin.


*`Limitations`*<br>
There are a few conditions that need to be met for an atomic swap to take place, and these may likely present obstacles for the technique to be widely adopted. For instance, to perform an atomic swap, the two cryptocurrencies need to be based on blockchains that share the same hashing algorithm (e.g., SHA-256 for Bitcoin). They also need to be compatible with HTLC and other programmable functionalities.
Other than that, atomic swaps bring up concerns about user's privacy. That’s because on-chain swaps and transactions can be quickly tracked on a blockchain explorer, making it easy to link the addresses. A short-term answer to this problem is to use privacy-focused cryptocurrencies as a way to reduce exposure. Still, many developers are experimenting with the use of digital signatures in atomic swaps as a more reliable solution.


