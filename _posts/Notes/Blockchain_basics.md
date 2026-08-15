----------

### Blockchain basics

----------

- What is bitcoin and blockchain? Bitcoin utilizes the blockchain technology.The bitcoin paper authored by `Satoshi Nakamoto` described that Bitcoin will be facilitated by peer to peer transactions over a decentralized networked with cryptography.
- A few years later, Ethereum was created by Vitalik Buterin and others which builds over Blockchain basics with additional capabilities.With Ethereum, you can create decentralized transactions, organizations and agreement without a centralized intermediary and was achieved with smart contracts.`Smart contracts` are a set of instructions executed in a decentralized way without the need for a centralized or third party intermediary.
- Smart contracts functionality is the difference between  `Ethereum` and `Bitcoin`.Technically Bitcoin does have smart contracts but they're intentionally turing incomplete.

----------

### The Smart Contract Problem

----------

- Smart contract face a simillar issue, they cannot interact with data from the real world which is known as the `Oracle Problem`.Blockchain works on data within their ecosystem and to make the compute real world data, real world data are required.
- Oracles serve this purpose. They are devices or services that provide data to blockchains or run external computation. To maintain decentralization, it's necessary to use a decentralized Oracle network rather than relying on a single source. This combination of on-chain logic with off-chain data leads to hybrid smart contracts.

-------------

### Examples of Oracle

-------------

- Chainlink-: is a decentralized Oracle network that allows smart contracts access external data.

-------------

### Layer 2 Solutions

------------

- Layer 2 solutions were created to address issues in blockchain tech.It involves other blockchain hooking into the main blockchain.2 types of L2 Solutions-:
  - Optimistic Rollups-: Optimism, Arbitrum
  - Zero Knowledge ROllups-: Zksync Polygon ZK Evm

------------

### Common Terms

------------

- Blockchain-: Blockchain is a digital ledger that records transaction in decentralized manner across different computers.Each block records all transactions and every new chain is linked to the old chain making it tamper resistant.
- Oracle-: It grants smart contracts access to external data.They act as bridges between the blockchain and the outside world allowing them compute real-world data.
- Layer 2-:These solutions ensures that transactions are done off the main chain and final state is done on the main chain.
- Dapp(Decentralized Application)-: It is an application that runs on a decentralized network e.g the blockchain. It is powered by a smart contract on a decentralized network.Dapps can serve various purposes e.g finance to gaming.A good example is `Uniswap`.
- Smart Contracts-: It is a self-executing agreement with the agreement within the code which is executed as soon as the pre determinined conditions are met, without the need for intermidiaries.
- Hybrid Smart Contracts-: They comine on-chain data and data provided by Oracles.It allows smart contract to interact with data off their chain.e.g A smart contract for insurance data
- Ethereum/EVM  (Ethereum Virtual Machine)-: Ethereum is a blockchain platform known for its smart contract functionality. The Ethereum Virtual Machine (EVM) is its computation engine that executes smart contracts. Ethereum allows developers to build decentralized applications and is the basis for many web3 projects.

------------

### Web3 

-------------

- It is the internet powered by blockchain and smart contracts.nlike the previous versions of the web, web3 is permissionless and relies on decentralized networks rather than centralized servers. This ushers in an era of censorship-resistant and transparent agreements and transactions, often called an ownership economy.

-------------

### The purpose of Smart Contracts

-----------

- Our entire interactions rely solely on contracts which means an agreement between individuals.Normal contracts required trust which can be set aside but not in the case of smart contracts.They are deployed on a decentralized network with visible terms for everyone to see and executed when predetermined terms are set.
- They do this by representing 'promises' as code on the blockchain. This code is executed by a decentralized collective, such that no single entity can alter the agreemeent in any way! The agreement and its terms are public knowledge and will automatically execute without human intervention.
- Although not all platforms are decentralized as claimed, A good example is the `SBF FIX company` which presented itself as a WEB3 company but a web2 one under the hood which used cryptocurrency without smart contracts.


------------

### Features of Smart Contracts

------------

- It contains certain features that distinguishes it from traditonal contracts.Features-:
 - Decentralization-: They do not rely on a central intermediary.Instead, they run on a blockchain managed on thousand of individuals known as node operators which makes it decentralized.
 - Transperency and Flexibility-: It is inherent in blockchain network since node operators can see everything going on.There is no room for unfair or hidden deals. This transparency ensures that everyone has access to the same information and plays by the same rules.Also, transactions are not tied toy our real identity.
 - Security and Immutability-: Once a smart contract is deployed, it cannot be altered or tampered with. This immutability ensures that the terms of the contract are set in stone. This is a stark contrast to centralized systems where a server or database can be hacked, and data can be altered. The decentralized nature of blockchain makes hacking nearly impossible since an attacker would have to take control of more than half the nodes, which is significantly more challenging than compromising a single centralized server.Additionally, the data on a blockchain is resilient. In a traditional system, if your computer and backups fail, you lose all your data. In contrast, in a blockchain, your data is replicated across thousands of nodes. Even if several nodes were to go down, your data would remain secure as long as there is at least one copy of the blockchain.
 - Elimination of Counterparty Risk-: It eliminates the risk of trust in transactions.Once a smart contract is deployed, it cannot be tweaked.None of the parties can tweak it based on greed to ensure it is enforced as intended.

------------

### Application of Smart Contracts

------------

- Defi(Decentralized Finance)-: It allows users to interact with the financial market without a central authority.With smart contracts, users have transparent access to financial markets and can engage with sophisticated financial products efficiently and securely. We will provide practical examples of how to build and interact with DeFi protocols in upcoming lessons.
- DAOs(Decentralized Autonomous Organizations)-:AOs are governed entirely by smart contracts and operate in a decentralized manner. This structure offers benefits such as transparent governance, efficient engagement, and clear rules. DAOs are an evolution in politics and governance
- Non-Fungible Tokens-: They are known as digital art or digital assets.

-----------

### Setting up Testnet on Tenderly

------------

- Visit [Tenderly](https://tenderly.co/?mtm_campaign=partner&mtm_kwd=cyfrin)
- Create account and click on `Virtual Testnet` on dashboard
- Settings-:

![image](https://github.com/user-attachments/assets/92f54036-5a42-4847-82e2-32cc26b7f150)

![image](https://github.com/user-attachments/assets/d822db39-b5fe-4de9-aef0-bd7cca242822)

-----------

### Working  with Testnets

-----------

- To send a token on a testnet, a `faucet` is required to get free tokens.

![image](https://github.com/user-attachments/assets/815bf05b-010f-4e1a-a0af-0372c55f74ed)

- Recommended [faucet](https://faucets.chain.link)
- Understanding Transaction Details
You can view transaction details on [Sepolia Etherscan](https://sepolia.etherscan.io). Important details include:
- Transaction hash: A unique identifier for the transaction.
- Status: Whether the transaction was successful.
- From: The address that sent the transaction.
- To: The address that received the transaction.
- Value: The amount of ETH sent (in this case, 0.1 Test ETH).
 
-----------

### Transaction and Gas Fees

----------

- The transaction fee is the amount rewarded to the block producer for processing the transaction. It is paid in Ether or GWei. The gas price, also defined in either Ether or GWei, is the cost per unit of gas specified for the transaction. The higher the gas price, the greater the chance of the transaction being included in a block.Gas price is not to be confused with gas. While gas refers to the computational effort required to execute the transaction, gas price is the cost per unit of that effort.

------------

### Role of nodes in Blockchain

------------

- Blockchains are run by a group of different nodes, sometimes referred to as miners or validators, depending on the network. These miners get incentivized for running the blockchain by earning a fraction of the native blockchain currency for processing transactions. For instance, Ethereum miners get paid in Ether, while those in Polygon get rewarded in MATIC, the native token of Polygon. This remuneration encourages people to continue running these nodes.
- A gas is a computational complexity unit.The total transaction fee can be calculated by multiplying the gas used with the gas price in Ether (not GWei). Therefore, `Transaction fee = gasPrice * gasUsed`.


-----------

### Blockchain mode of operation

-----------

- Understanding hash functions-: Hash is a fixed length of string that serves to identify a piece of data. Eth uses its own form of hashing algorithm(keccak 256) which belongs to sha256 family.
- Understanding Block-: The blockchain is a collection of "blocks". A `block` is divided into `block`,`nonce` and `data`.All three are then run through the hash algorithm, producing the hash for that block. As a result, even a minor change in the data leads to an entirely different hash, hence, invalidating the block.In essence, mining involves the computational trial and error process of finding an acceptable value to produce a hash which typically follows a certain pattern, such as starting with four zeros. The value found, which satisfies this criterion, is known as the 'nonce'.

-----------

### Inherent immutability 

-----------

- In a blockchain, which is essentially a sequence of blocks, each block is comprised of the previous elements - a block number, a nonce and data - as well as the hash of the previous block.What this means in practice is that any changes to data, in any block of the chain, will invalidate every proceeding block, until they are recalculated, or re-mined.

![image](https://github.com/user-attachments/assets/6abd1680-07b2-47b1-aadd-6442c1b66e94)

- The first block is known as the `Genesis Block`.

--------------

### Decentralized Distribution

--------------

- Now, if a single entity were to control the blockchain, they could conceivably change any data they want, and then re-mine, or re-validate subsequent blocks. This is bad.The crux of blockchain's power lies in its decentralization or distributed nature. Under this system, multiple entities or "peers" run the blockchain technology, each holding equal weight and power. In the event of disparity between the blockchains run by different peers (due to tampering or otherwise), the majority hash wins, as the majority of the network agrees on it.
- Interplay of transactions in blockchain-: Until now we've been considering the data passed in a block to be a random string of text, but the reality is - this data can be anything. In the token and coinbase sections of this demo you can see how each block is comprised of a number of transactions that all get hashed together. Any edits to any of these transactions is going to invalidate the chain!


--------------

### Public and Private Keys

-------------

- The private key is then passed through an algorithm (the Elliptic Curve Digital Signature Algorithm for Ethereum and Bitcoin) to create the corresponding public key. Both the private and public keys are central to the transaction process. However, while the private key must remain secret, the public key needs to be accessible to everyone.

-------------

### How does transaction signing occur?

--------------

- When we sign a transaction on the blockchain, we're digitally signing some data with our private key. The hashing algorithm used makes it impossible for something to derive your private key from a message signature.
- It is like passing a private key to the blockchain-:

![image](https://github.com/user-attachments/assets/13b15622-b549-4cf1-88e6-8e1c7cd4e29a)

- This signing method allows anyone to verify the validity of a transaction by comparing the message signature to a user's public key!
- In addition to this, Ethereum addresses are derived from public keys by hashing a user's public keys with the Keccak256 algorithm.

--------------

### Transaction and Gas

--------------

- Important terms-:

```
Wei:  1,000,000,000 Wei  = 1 Gwei (Gigawei)
Gwei: 1,000,000,000 Gwei = 1 Eth
```

![image](https://github.com/user-attachments/assets/346b38bb-a924-4f8f-a45f-43d66acdf893)

   - Transaction Fee: This is calculated as Total Gas Used * Gas Price where Gas Used represents the computational units required to perform the work and Gas Price is comprised of a Base and Priority Fee.
   - Gas Limit: This is the maximum amount of gas allowed for the transaction. This can be set by the user prior to sending a transaction.

- Base Gas Fee: The base fee of a transaction, represented in Gwei. Remember, this is cost per gas.
- There are a couple important points to note regarding the Base Fee.The fee is burnt as of EIP-1559. Burning serves to remove the value from circulation, combating inflation on the protocol. The amount burnt can be seen beneath the Base Fee in the image above.The fee is dynamic, under EIP-1559, if a block is more than 50% full, the Base Gas Fee is increased for the next block. Likewise, if a block is less than 50% full, the fee decreases. This serves to balance network demand and capacity.
- Max Gas Fee: This is the maximum cost per cast the transaction has been configured to allow. This can again be configured prior to sending a transaction.
- Max Priority Fee: Again, configurable prior to sending a transaction, this represents the maximum tip we're willing to give miners. This incentivizes the inclusion of our transaction within a block.
- Block Confirmations: These are he number of blocks which have been mined or validated which have been confirmed to contain your transaction. The more confirmations the more sure we can be of the transaction's validity.

--------------

### Blockchain overview

--------------

- Traditionally, when you run an application be it a website or something that connects to a server you are interacting with a centralized entity. Blockchain runs on independent nodes.In our previous example, each of the Peers was representative of an independent node operator. The term node typically refers to a single instance of a decentralized system, Peer A would be a node. This network, this combination of these nodes interacting with each other is what creates a blockchain. What makes these networks so potent, is that anybody can join.
- In the traditional world applications are run by centralized entities and if that entity goes down or is malicious or decides that they want to shut off - they just can.Blockchains, by contrast, don't have this problem. If one node or one entity that runs several nodes goes down, since there are so many other independent nodes running, it doesn't matter, the blockchain and the system will persist so long as there is at least one node always running. Luckily for us, the most popular chains like Bitcoin and Ethereum have thousands and thousands of nodes. Malicious nodes are kicked from the network, or even punished in some cases. Majority rules when it comes to the blockchain.

- Consensus-: It includes  `Proof of work` and `Proof of Stake`, they both fall under `consensus`.Consensus is defined as the mechanism used to reach an agreement on the state or a single value on the blockchain especially in a decentralized system.
- It is a decentralized protocol that can be broken into two pieces.
  - A chain selection algorithm
  - A sybil resistance mechanism
- Mining, or Proof of Work, is a sybil resistance mechanism. This is what Bitcoin currently uses.
- Proof of Work is known as a sybil resistance mechanism because it defines a way to figure out who is the block author or which node did the work to mine a block. Sybil resistance is a blockchain's ability to defend against users creating a large number of pseudo-anonymous identities to gain a disproportionately advantageous influence over said system.Types-:
  - Proof of work
  - Proof of Stake

---------------

- Proof of Work-: Proof of work is a system of sybil resistance used in many blockchains, in its essence a miner needs to go through a very computationally heavy process (mining) to find the block's answer. As a result, it doesn't matter how many additional nodes you're running, each node is obligated to do this work in order to receive a reward. The playing field is kept fair.Proof of Work needs to be combined with a chain selection rule to create consensus.
- A `chain selection rule` is implemented as a means to determine which blockchain is the real blockchain. Bitcoin (and prior to the merge, Ethereum), both use something called `Nakamoto Consensus`. This is a combination of Proof of Work (Etherum has since switched to Proof of Stake) and the longest chain rule.
- Proof of Work also serves as a means to determine who receives transaction fees as we discussed earlier. These transaction fees are paid by whomever initiates the transaction. In a Proof of Work system, every node is competing against eachother to solve the block problem first. The first node to solve the problem gets paid the transaction fees accumulated in the block they mine. In addition to this, miners are also paid a block reward, the block reward is given by the blockchain itself.

-----------------

### Blockchain Attacks

------------------

- Sybil Attack-: When a user creates a number of pseudo-anonymous accounts to try to influence a network.
- 51% attack - Occurs when a single entity possesses both the longest chain and majority network control. This would allow the entity to fork the chain and bring the network onto the entities record of events, effectively allowing them to validate anything.


------------------

- Proof of Work does come with drawbacks. For example, Proof of Work consumes a LOT of electricity. When you have thousands of nodes all working as hard as they can to solve a block problem the energy consumption is HUGE and as such, so is the potential environmental impact.

------------------

### Proof of Stake

------------------

- Proof of Stake nodes put up some collateral that they are going to behave honestly aka they stake. If a node is found to be misbehaving, it's stake is slashed.This serves as a very effective sybil resistance mechanism because for each account, the validator needs to put up more stake and misbehaving risks losing all that collateral.

------------------

### Layer 1 and 2

-------------------

- Layer 1 solutions: This refers to base layer blockchain implementations like Bitcoin or Ethereum.
- Layer 2 solutions: These are applications added on top of a layer one, like `Chainlink` or `Arbitrum`.

-------------------

### Blockchain Layers

------------------

- A `Layer 1(L1)` is the base layer of the blockchain that allows nodes to reach consensus. It operates without any additional plugins and is often referred to as the settlement layer.Examples of L1 chains include Bitcoin, BNB Chain, Solana, and Avalanche. In this course, we primarily focus on Ethereum, which serves as the hub of the Ethereum ecosystem. Applications directly deployed on Ethereum, like Uniswap, are not considered L2s but rather dApps on L1.
- `Layer 2(L2)` are built outside the L1 blockchain that hooks back into it.There are different types of Layer 2, for example Chainlink, a decentralized Oracle networks and event indexing networks like The Graph, which enable applications to access on-chain data. But the most popular type of L2 is the rollup, or L2 chain.
- Rollups-: It is a L2 scaling solutions that increases the number of transactions on Ethereum by bundling it into one, reducing gas costs.Rollups help solve the blockchain trilemma, which states that a blockchain can only achieve two out of three properties: decentralization, security, and scalability. In the case of Ethereum, scalability is sacrificed as it can only process approximately 15 transactions per second. Rollups, on the other hand, aim to enhance scalability without compromising security or decentralization.

------------------

### How Rollup works

----------------

- When a user submits a transaction to a rollup, an operator (a node or entity responsible for processing transactions) picks it up, bundles it with other transactions, compresses them, and submits the batch back to the L1 blockchain. This process allows for efficient handling of transactions as gas costs associated with the transaction, are split among all the users that submitted the transactions in the batch.

-----------------

### Types of Rollups

------------------

- Optimistic rollups
- Zero-Knowledge rollups

-----------------

### Optimistic rollups

-----------------

- They assume that off-chain transactions are valid by default. Operators propose the valid state of the rollup chain, and during a challenge period, other operators can challenge potentially fraudulent transactions by computing a fraud proof.This fraud proof process involves the operator engaging in a call and response interaction with another operator to identify and isolate a specific computational step. This specific step is then executed on the Layer 1 blockchain: if the result differs from the original state, it indicates that the transaction was fraudulent. When the fraud proof succeeds, the rollup will re-execute the entire batch of transactions correctly, and the operator responsible for including the incorrect transaction will be penalized, usually by losing staked tokens (slashing).
- Zero Knowledge-: ZK rollups use validity proofs, known as zk proofs, to verify transaction batches. In this process, the prover (operator) generates a zk proof to show that their inputs (the transactions) satisfy this equation. A verifier (an L1 contract) then checks this proof to ensure that the output matches the expected result. The solution that the prover uses to demostrate that their input satisfies the mathematical equation in the zk proof is commonly referred as the witness.

------------------

### Centralized Sequencer I

------------------

- In blockchain and cryptocurrency networks, the role of a sequencer is crucial for ordering and bundling transactions. Sequencers are operators that are responsible for organizing how transactions are processed. In many roll-up solutions, sequencers are centralized, controlled by a single entity.

-------------------

### Rollup Stages

--------------------

- Stage 0-: In this initial stage, the rollup's governance is largely in the hands of the operators and a security council, ensuring that critical decisions and actions are overseen by a trusted group. The open-source software allows for the reconstruction of the state from L1 data, ensuring transparency and accessibility. Users in this stage have an exit mechanism that allows them to leave the rollup within seven days. However, this often requires actions from an entity/operator.
- Stage 1-: In this stage, governance evolves to be managed by smart contracts, although the security council still plays an important role (e.g. solving bugs). At this stage, the proof system becomes fully functional, enabling decentralized submission of validity proofs. The exit mechanism is improved, allowing users to exit independently without needing operator coordination.
- Stage 2-: n this final stage, the rollup achieves full decentralization with governance entirely managed by smart contracts, removing the need for operators or council interventions in everyday operations. The proof system at this stage is permissionless and the exit mechanism is also fully decentralized. The security council's role is now strictly limited to addressing any errors that occur on-chain, ensuring that the system remains fair without being overly reliant on centralized entities.

---------------------

------------------
