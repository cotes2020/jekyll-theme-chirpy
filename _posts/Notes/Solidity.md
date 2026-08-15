-------------

### Solidity Basics

-------------

- Installing solcjs

```bash
sudo npm install --global solc
```

- Specifying the version

```sol
//version
pragma solidity 0.8.30; //use only 0.8.30
//use versions between 0.8.19 and 0.9.0 (excluded)
pragma solidity ^0.8.19;
pragma solidity >=0.8.19 <0.9.0;
```
- SPDX License Identifier, use
```sol
//SPDX-License-Identifier: MIT
//solidity version
pragma solidity ^0.8.30;
```

- Running with `solcjs`-:

```bash
solcjs --bin main.sol
```
![image](https://github.com/user-attachments/assets/d14f8c95-e6a8-442d-94ec-f2d99164bdbc)

- Code-:

```sol
//SPDX-License-Identifier: MIT
//solidity version
pragma solidity ^0.8.30;

contract simpleStorage {
    //code goes thus
}
```
- Bool-:

```sol
bool hasFavoriteNumber = true; //true or false
```
- Basic types-:

```sol
   bool hasFavoriteNumber = true;
    uint256 favoriteNumber = 88;
    string favoriteNumberInText = "eighty-eight";
    int256 favoriteInt = -88;
    address myAddress = 0xAB1b7206aa6840C795aB7A6AE8b15417B7E63A8D;
    bytes32 favoriteBytes32 = "cat";
```

- create a function with `function`-:

```sol
//SPDX-License-Identifier: MIT
pragma solidity ^0.8.1;

contract Storage{
    uint256 digit = 100;
    function store(uint256 _digit) public {
        digit = _digit;
    }
}
```

-  Appending the public keyword next to a variable will automatically change its visibility and it will generate a getter function (a function that gets the variable's value when called).

```sol
uint256 public digit = 1000;
```

- Visibility for variables and function-:

```
public
private
internal
external
```

- Pure and view functions,-:

```sol
//SPDX-License-Identifier: MIT
pragma solidity ^0.8.1;

contract Storage{
    uint256 digit = 100;
    function store(uint256 _digit) public {
        digit = _digit;
    }
    //view
    function retrieve() public view returns {
        return digit;
    }
    //pure
    function retrieve() public pure returns {
        return 7;
    }
}
```
- To store function in memory, use `memory` -:

```solidity
function createZombie(string memory _name,uint _dna) public {

}
```
- Return a value in a function with `return`, specify like this

```solidity
string greeting = "What's up dog";

function sayHello() public returns (string memory) {
  return greeting;
}
```

- Function modifiers
- The below function doesn't actually change state in solidity e.g it doesn't chnage values or write anything.It is set with `view`

```solidity
function sayHello() public view returns (string memory){
}
```
- Solidity also contains pure functions meaning you are not accessing any data fron the app e.g

```solidity
function _multiplyNumbers(uint a, uint b) private pure returns (uint memory) {
  return a * b;
}
```
- 
 
--------------

### Arrays and Structs

--------------

- Creating a list of `uint256` numbers-:

```
uint256[] digits = [0,10,90];
```

- The issue with this data structure above is that we cannot link the owner with its favorite value. One solution is to establish a new type using the struct keyword, named Person, which is made of two attributes: a favorite number and a name.

```sol
struct Person {
   string name;
   uint256 age;
}
```
- Using it-:

 ```sol
Person public ade = Person("Ade",100);
```
- Array of struct-: Creating individual variables that represent several people might become a tedious task, due to the repetitive steps of the process. Instead of manually instantiating a variable for each person, we can combine the two concepts we just learned about: arrays and structs.

```sol
Person[] public people;
Person[3] public three_people;
```

- Pushing new values to an `array of structs` with a function.

```sol
//function push to three_people array
    function push_value(uint256 _negativeNumber, string memory _name) public{
        people.push(Person(_negativeNumber,_name));
  }
```

-------------

### Memory Storage

---------------

- How does solidity handle data storage? It uses locations like this-:

```
Calldata
Memory
Storage
Stack
Code
Logs
```

- `Calldata` and `memory` are temporary storage locations during function execution.calldata is read-only, used for function inputs that can't be modified. In contrast, memory allows for read-write access, letting variables be changed within the function. To modify calldata variables, they must first be loaded into memory.
- Memory-:

```sol
string memory name = "Ade";
```
-----------------

### Keccak256 and Typecasting

-----------------

- Ethereum's `keccak256` is the version of `SHA3`.It maps an input into a 256-bit hexadecimal number.It expects a parameter of type `bytes`.It means we have to pack `string` to `bytes`.

```solidity
keccak256(abi.encodePacked("deadbeef"))
```

-  Typecasting involves converting between data types.For example-:

```solidity

uint a = 6;
uint8 b = 9;

uint8 c = b * uint8(a);
```
-------------------

### Events
--------------------

- Events are a way for your contract to communicate that something happened on the blockchain to your app front-end, which can be 'listening' for certain events and take action when they happen.

```solidity
// declare the event
event IntegersAdded(uint x, uint y, uint result);

function add(uint _x, uint _y) public returns (uint) {
  uint result = _x + _y;
  // fire an event to let the app know the function was called:
  emit IntegersAdded(_x, _y, result);
  return result;
}
```

- Javascript implementation-:

```js
YourContract.IntegersAdded(function(error, result) {
  // do something with the result
})
```
- Status (chapter 15)-:

```solidity
pragma solidity >=0.5.0 <0.6.0;

contract ZombieFactory {

    event NewZombie(uint zombieId,string name,uint dna);
    
    uint dnaDigits = 16;
    uint dnaModulus = 10 ** dnaDigits;

    struct Zombie {
        string name;
        uint dna;
    }

    Zombie[] public zombies;

    function _createZombie(string memory _name, uint _dna) private {
        uint id = zombies.push(Zombie(_name, _dna)) - 1; // 2. Store the result of `zombies.push(...) - 1` in a `uint` called `id`
        emit NewZombie(id, _name, _dna);
    }

    function _generateRandomDna(string memory _str) private view returns (uint) {
        uint rand = uint(keccak256(abi.encodePacked(_str)));
        return rand % dnaModulus;
    }

    function createRandomZombie(string memory _name) public {
        uint randDna = _generateRandomDna(_name);
        _createZombie(_name, randDna);
    }

}
```

------------------

### Intermediate Solidity

--------------------

- New data types `mapping` and `address`
- Each account has an `address` which you can think of as a bank account which looks like this `0x0cE446255506E92DF41614C46F1d6df9Cc969183`. Mappings are another way of storing data
- A mapping might look like this-:

```solidity
mapping (address => uint) public accountBalance;
mapping (uint => string ) userIdToName;
```

-  It is typically a key-value pair for storing and looking up data.

---------------------

### Msg.sender

---------------------

- There are certain global functions available to all functions. One of them is `msg.sender` which belongs to the address of the person that called a function in a contract.

>Note: In Solidity, function execution always needs to start with an external caller. A contract will just sit on the blockchain doing nothing until someone calls one of its functions. So there will always be a msg.sender.

- Example-:

```solidity
mapping (address => uint ) public contractAddresses;

function setAddressValue(uint memory _number) public {
    contractAddresses[msg.sender] = _number;
}
```

- Increasing counter in solidity-:

```sol
uint number = 0;
number++;
```

- `Require` is used to ensure a line of code throws an error if the condition is not fulfilled.

```solidity
require(abi.encodePacked('a') == abi.encodePacked('b'));
```
-------------

### Inheritance

--------------

- Inheritance prevents a long contract by breaking it into several contracts inheriting attributes from the each other.Example-:

```solidity
contract Doge {
}
//is extends a subclass to base class
contract baseDoge is Doge {

}
```
- Importing a contract from another file, use-:

```solidity

import "./someContract.sol"

contract Doge is someContract {
}
```

-  `Storage v Memory` data location-:
-  `Storage` data are stored on the blockchain while `Memory` values are stored in the `RAM` or the `hard-disk`.
-  Although, you don't need to specify for `storage` because state variables are stored in it by default and permanently written to the blockchain, while variables stored in function are stored in the `memory` and disappears after a function gets called.

```solidity
function feedAndMultiply(uint _zombieId, uint _targetDna) public {
    require(msg.sender == zombieToOwner[_zombieId]);
    Zombie storage myZombie = zombies[_zombieId];
    // start here
  }
```
- Access a struct's data with `struct.key`.e.g

```
myZombie.dna
```
- More on function visibility-:
- Soldity has two more types of visibility: `internal` and `external`.
- `internal` is like private but it is accessible by a contract that inherits the contract.
- `external` is similar to public, except that these functions can ONLY be called outside the contract — they can't be called by other functions inside that contract.
- Interacting with other contracts, we can interact with other contracts on a blockchain with an `interface`.
- `interface` is declared like this without the curly braces-:

```solidity
contract KittyInterface {
  function getKitty(uint256 _id) external view returns (
    bool isGestating,
    bool isReady,
    uint256 cooldownIndex,
    uint256 nextActionAt,
    uint256 siringWithId,
    uint256 birthTime,
    uint256 matronId,
    uint256 sireId,
    uint256 generation,
    uint256 genes
  );
}
```

- Using an interface-:

```
address ckAddress = 0x06012c8cf97BEaD5deAe237070F9587f8E7A266d;
  // Initialize kittyContract here using `ckAddress` from above
  KittyInterface kittyContract = KittyInterface(ckAddress);
```

-  Handling multiple return values-:
```solidity
function multipleReturns() internal returns(uint a, uint b, uint c) {
  return (1, 2, 3);
}

function processMultipleReturns() external {
  uint a;
  uint b;
  uint c;
  // This is how you do multiple assignment:
  (a, b, c) = multipleReturns();
}

// Or if we only cared about one of the values:
function getLastReturnValue() external {
  uint c;
  // We can just leave the other fields blank:
  (,,c) = multipleReturns();
}
```
- `if` statements-:

```solidity
if () {
}
```

--------------

### Advanced Concepts

---------------

- Contracts are immutable in solidity which means once updated, it can never be updated anymore.
-  `constructor()` is a constructor, which is an optional special function. It will get executed only one time, when the contract is first created.
Function Modifiers: modifier onlyOwner(). Modifiers are kind of half-functions that are used to modify other functions, usually to check some requirements prior to execution
- Example of a modifier-:

```solidity
modifier onlyOwner() {
    require(isOwner());
    _;
  }
```
-  how it is called in a function-:

```solidity
function tryModifier(uint256 _number) external onlyOwner {
}
```
- Time units-:
> The variable `now` will return the current Unix timestamp.Solidity also contains the time units `seconds`, `minutes`, `hours`, `days`, `weeks` and `years`. These will convert to a uint of the number of seconds in that length of time.
> Note: The uint32(...) is necessary because now returns a uint256 by default. So we need to explicitly convert it to a uint32.


---------------

### Passing struct as arguments

----------------

- You can pass a storage pointer to a struct as an argument to a private or internal function. This is useful, for example, for passing around our Zombie structs between functions.

```solidity
function _doStuff(Zombie storage _zombie) internal {
  // do stuff with _zombie
}
```

- Stopped at lesson 5 -:

```solidity
pragma solidity >=0.5.0 <0.6.0;

import "./zombiefactory.sol";

contract KittyInterface {
  function getKitty(uint256 _id) external view returns (
    bool isGestating,
    bool isReady,
    uint256 cooldownIndex,
    uint256 nextActionAt,
    uint256 siringWithId,
    uint256 birthTime,
    uint256 matronId,
    uint256 sireId,
    uint256 generation,
    uint256 genes
  );
}

contract ZombieFeeding is ZombieFactory {

  KittyInterface kittyContract;

  function setKittyContractAddress(address _address) external onlyOwner {
    kittyContract = KittyInterface(_address);
  }

  // 1. Define `_triggerCooldown` function here
  function _triggerCooldown(Zombie storage _zombie) internal {
    _zombie.readyTime = uint32(now + cooldownTime);
  }

  // 2. Define `_isReady` function here
  function _isReady(Zombie storage _zombie) internal view returns (bool){
    return (_zombie.readyTime <= now );
  }
  function feedAndMultiply(uint _zombieId, uint _targetDna, string memory _species) public {
    require(msg.sender == zombieToOwner[_zombieId]);
    Zombie storage myZombie = zombies[_zombieId];
    _targetDna = _targetDna % dnaModulus;
    uint newDna = (myZombie.dna + _targetDna) / 2;
    if (keccak256(abi.encodePacked(_species)) == keccak256(abi.encodePacked("kitty"))) {
      newDna = newDna - newDna % 100 + 99;
    }
    _createZombie("NoName", newDna);
  }

  function feedOnKitty(uint _zombieId, uint _kittyId) public {
    uint kittyDna;
    (,,,,,,,,,kittyDna) = kittyContract.getKitty(_kittyId);
    feedAndMultiply(_zombieId, kittyDna, "kitty");
  }

}
```

----------------------

### Public functions & security

----------------------

- An important security practice is to examine all your public and external functions, and try to think of ways users might abuse them. Remember — unless these functions have a modifier like onlyOwner, any user can call them and pass them any data they want to.

- Function modifiers-: They can also take arguments.

```solidity
modifier aboveLevel(uint _level, uint _zombieId) {
    require(zombies[_zombieId].level >= _level);
    _;
  }
```

- More on storage location for data-:
> Note: calldata is somehow similar to memory, but it's only available to external functions.
- View functions don't cost gas.

> `view` functions don't cost any gas when they're called externally by a user.This is because view functions don't actually change anything on the blockchain – they only read the data. So marking a function with view tells web3.js that it only needs to query your local Ethereum node to run the function, and it doesn't actually have to create a transaction on the blockchain (which would need to be run on every single node, and cost gas).

- Storage is expensive in solidity (particularly writes).
- Declaring arrays in memory-:

```solidity
 uint[] memory result = new uint[](ownerZombieCount[_owner]);
    // Start here
    return result;
```
- For loops-:

```solidity
 for (uint i = 0; i < zombies.length; i++) {
      if (zombieToOwner[i] == _owner) {
        result[counter] = i;
        counter++;
      }
    }
```
-------------------
- Payable-:
- This modifier function makes Solidity and Ethereum dope. It can recieve Ether.
> When you call an API function on a normal web server, you can't send US dollars along with your function call — nor can you send Bitcoin.But in Ethereum, because the money (Ether), the data (transaction payload), and the contract code itself all live on Ethereum, it's possible for you to call a function and pay money to the contract at the same time.

- Here, msg.value is a way to see how much Ether was sent to the contract, and ether is a built-in unit.
```solidity
 function levelUp(uint _zombieId) external payable {
    require(msg.value == levelUpFee);
    zombies[_zombieId].level++;
  }
```

-  Withdraws-:
  - After you send Ether to a contract, it gets stuck in it except you sent a contract to withdraw it.
 > It is important to note that you cannot transfer Ether to an address unless that address is of type address payable. But the _owner variable is of type uint160, meaning that we must explicitly cast it to address payable.
 > Once you cast the address from uint160 to address payable, you can transfer Ether to that address using the transfer function, and address(this).balance will return the total balance stored on the contract. So if 100 users had paid 1 Ether to our contract, address(this).balance would equal 100 Ether.
> You can use transfer to send funds to any Ethereum address. For example, you could have a function that transfers Ether back to the msg.sender if they overpaid for an item:
```solidity
uint itemFee = 0.001 ether;
msg.sender.transfer(msg.value - itemFee);
```
- Example-:

```solidity
function withdraw() external onlyOwner {
    address payable _owner = address(uint160(owner()));
    _owner.transfer(address(this).balance);
  }
```

- Random Numbers-:
- The best randomness hash function in solidity is the `keccak256` hash function.

```solidity
 uint randNonce = 0;
  function randMod(uint _modulus) internal returns (uint) {
    randNonce++;
    return uint(keccak256(abi.encodedPacked(now,msg.sender,randNonce))) % _modulus;
  }
```
-------------------

### Refactoring Code Logic

-------------------

-  Using two modifers for a function

```solidity
function changeDna(uint _zombieId, uint _newDna) external aboveLevel(20, _zombieId) ownerOf(_zombieId) {
    zombies[_zombieId].dna = _newDna;
  }
```

- 

--------------------

### ERC721 & Crypto-Collectibles

--------------------

- A token in Ethereum basically behaves like a smart contract that follow some rules, it implements a set of functions that all other contracts follows, such as-:

```solidity
transferFrom(address _from, address _to, uint256 _amount) and balanceOf(address _owner)
```

- Internally, the smart contract has a mapping of `mapping( address => uint256) balances`, that keeps track of balances for each address. So basically a token is just a contract that keeps track of who owns how much of that token, and some functions so those users can transfer their tokens to other addresses.

---------------------

### ERC721 Standard

---------------------

- ERC721 standard-:

```solidity
contract ERC721 {
  event Transfer(address indexed _from, address indexed _to, uint256 indexed _tokenId);
  event Approval(address indexed _owner, address indexed _approved, uint256 indexed _tokenId);

  function balanceOf(address _owner) external view returns (uint256);
  function ownerOf(uint256 _tokenId) external view returns (address);
  function transferFrom(address _from, address _to, uint256 _tokenId) external payable;
  function approve(address _approved, uint256 _tokenId) external payable;
}
```

- A contract can also inherit multiple contracts.e.g

```solidity
contract Sathoshi is  Sensei,Blackie {

}
```
---------------------

### ERC721 implementation

---------------------

- Transfer logic-: ERC721 has rwo different methods of transferring tokens.

>function transferFrom(address _from, address _to, uint256 _tokenId) external payable;
and
function approve(address _approved, uint256 _tokenId) external payable;
function transferFrom(address _from, address _to, uint256 _tokenId) external payable;

- Preventing Overflows(Contract security enhancements: Overflows and Underflows)-:

```solidity
import "./safemath.sol"

contract ZombieFactory is Ownable {

  // 2. Declare using safemath here
  using SafeMath for uint256;

```

- Using Safemath for our code-: To prevent overflows and underflows, we can look for places in our code where we use +, -, *, or /, and replace them with add, sub, mul, div.
- Declaring for `uint32` and `uint16`-:

```solidity
using SafeMath32 for uint32;
using SafeMath16 for uint16;
```

-  Comments in Solidity-:

```solidity
//Single line comments
/* Multi line comments
lolzz*/
```

- Natspec standard-:

```solidity
/// @title  A contract that manages transferring zombie ownership
/// @uthor Sensei__
/// @dev Compliant with OpenZeppelin's implementation of the ERC721 spec draft
```

---------------------

### Deploying DAPPS

---------------------

- The Ethereum network is made up of nodes, with each containing a copy of the blockchain. When you want to call a function on a smart contract, you need to query one of these nodes and tell it:

- The address of the smart contract
- The function you want to call, and
- The variables you want to pass to that function.
- Ethereum nodes only speak a language called JSON-RPC, which isn't very human-readable. A query to tell the node you want to call a function on a contract looks something like this:

```json
{"jsonrpc":"2.0","method":"eth_sendTransaction","params":[{"from":"0xb60e8dd61c5d32be8058bb8eb970870f07233155","to":"0xd46e8dd67c5d32be8058bb8eb970870f07244567","gas":"0x76c0","gasPrice":"0x9184e72a000","value":"0x9184e72a","data":"0xd46e8dd67c5d32be8d46e8dd67c5d32be8058bb8eb970870f072445675058bb8eb970870f072445675"}],"id":1}
```

- Web3.js hides these nasty queries below the surface, so you only need to interact with a convenient and easily readable JavaScript interface. Using web3.js

```js
CryptoZombies.methods.createRandomZombie("Vitalik Nakamoto 🤔")
  .send({ from: "0xb60e8dd61c5d32be8058bb8eb970870f07233155", gas: "3000000" })
```

- Add this to your project-:

```html
<script language="javascript" type="text/javascript" src="web3.min.js"></script>
```

- 

---------------------
