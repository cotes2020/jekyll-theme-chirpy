---
layout: post
title: "12-iOS签名机制"
date: 2020-05-04 20:51:00.000000000 +09:00
categories: [逆向工程]
tags: [逆向工程, iOS签名机制, Certificate]
---

## 学习路线

```swift
加密解密 --> 单向散列函数 --> 数字签名 --> 证书 --> iOS签名机制
// 加密解密: 对称密码(DES、3DES、AES)、公钥密码(RSA)
// 单向散列函数: MD4、MD5、SHA-1、SHA-2、SHA-3
```

## 常见英文

+ `encrypt`: 加密
+ `decrypt`: 解密
+ `plaintex`t: 明文
+ `ciphertext`: 密文

## 密码类型

+ **对称密码** `Symmetric Cryptography`

  + 对称密码中，加密用的密钥和解密用的密钥是相同的。
  + 常见的对称密码算法有
    + `DES (Data Encryption Statdard)`
      + `DES`是一种将64bit明文加密成`64bit`密文的对称密码算法，密钥长度是`56bit`。
      + 规格上来说，密钥长度是`64bit`，但每隔`7bit`会设置一个用于错误检查的bit，因此密钥长度实质上市`56bit`。
      + 由于`DES`每次只能加密`64bit`的数据，遇到比较大的数据，需要对`DES`加密进行迭代。
      + 目前已经可以在短时间内被破解，所以不建议使用。
    + `3DES`
      + `3DES`，将`DES`重复3次所得到的的一种密码算法，也叫做`3重DES`。
      + 目前还被一些银行等机构使用个，但处理速度不高，安全性逐渐暴露出问题。
    + `AES (Advanced Encryption Standard)`
      + 取代`DES`成为新标准的一种对称加密算法。
      + `AES`的密钥长度有`128bit`、`192bit`、`256bit`三种。
      + 在2000年是选择Rijindael算法作为AES的实现
      + 目前`AES`已经逐步取代`DES`、`3DES`，成为首选的对称密码算法。
      + 一般来说，我们也不应该去使用任何自制的密码算法，而是应该使用`AES`。
  + 密钥配送问题
    + 事先共享密钥
    + 密钥分配中心
    + `Diffie-Hellman`密码交换
    + `公钥密码`

+ **公钥密码** (非对称密码)

  + `公钥密码`中，加密用的密钥和解密用的密钥是不相同的。
  + 公钥密码中，密钥分为加密密钥、解密密钥，他们并不是同一个密钥。
  + 公钥密码也被称为`非对称密码(Asymmetric Cryptography)`
  + 在公钥密码中
    + 加密密钥，一般是公开的，因此该密钥称为`公钥(public key)`
    + 解密密钥，由消息接收者自己保管的，不能公开，因此也称为`私钥(private key)`
    + 公钥和私钥是一 一对应的，是不能单独生成的，一对公钥和密钥统称为`密钥对(key pair)`
    + 由公钥加密的密文，必须使用与该公钥对应的私钥才能解密
    + 由私钥加密的密文，必须使用与该私钥对应的公钥才能解密

  ![crypto01](/assets/images/reverse/crypto01.png)

  + 解决密钥配送问题
    + 由消失的接收者生成一对公钥、私钥。
    + 将公钥发给消息的发送者。
    + 消息的发送者使用公钥加密消息。

  ![crypto02](/assets/images/reverse/crypto02.png)

+ **RSA**
  + 目前使用最广泛的公钥密码算法是RSA。
  + `RSA`的名字，有三位开发者，即`Ron Rivest`、`Adi Shamir`、`Leanard Adleman`的姓氏首字母组成。
  + `RSA`解决配送问题，对称密码解决传送问题。

## 混合密码系统(Hybrid Cryptosystem)

+ 对称密码的缺点

  + 不能很好解决密钥配送问题。

+ 公钥密码的缺点

  + 加密解密速度比较慢

+ 混合密码系统，是将对称密码和公钥密码的有事相结合的方法

  + 解决了公钥密码速度慢的问题。
  + 并通过公钥密码解决了对称密码的密钥配送问题

+ 网络上的密码通信所有的`SSL/TLS`都运用了混合密码系统

+ 混合密码 -- 加密

  + 会话密钥（session key）
    + 为本次通信随机生成的临时密钥
    + 作为对称密码的密钥，用于加密消息，提高速度
  + 加密步骤（发送消息）
    + 1.首先，消息发送者要拥有消息接收者的公钥
    + 2.生成会话密钥，作为对称密码的密钥，加密消息
    + 3.用消息接收者的公钥，加密会话密钥
    + 4.将前2步生成的加密结果，一并发给消息接收者
  + 发送出去的内容包括
    + 用会话密钥加密的消息（加密方法：对称密码）
    + 用公钥加密的会话密钥（加密方法：公钥密码）

  ![crypto03](/assets/images/reverse/crypto03.png)

+ 混合密码--解密

  + 解密步骤(收到消息)
    + 1.消息接收者用自己的私钥解密出会话密钥
    + 1.再用第1步解密出来的会话密钥，解密消息

  ![crypto04](/assets/images/reverse/crypto04.png)

+ 混合密码--加密解密流程

  + Alice -- 发消息给  --> Bob

  + 发送过程，加密过程

    > 1.Bob先生成一对公钥、私钥
    >
    > 2.Bob把公钥共享给Alice
    >
    > 3.Alice随机生成一个会话密钥（临时密钥）
    >
    > 4.Alice用会话密钥加密需要发送的消息（使用的是对称密码加密）
    >
    > 5.Alice用Bob的公钥加密会话密钥（使用的是公钥密码加密，也就是非对称密码加密）
    >
    > 6.Alice把第4、5步的加密结果，一并发送给Bob

  + 接收过程，解密过程

    > 1.Bob利用自己的私钥解密会话密钥（使用的是公钥密码解密，也就是非对称密码解密）
    >
    > 2.Bob利用会话密钥解密发送过来的消息（使用的是对称密码解密）

## 单向散列函数(One-way hash function)

+ 单向散列函数，可以根据根据消息内容计算出散列值
+ 散列值的长度和消息的长度无关，无论消息是1bit、10M、100G，单向散列函数都会计算出固定长度的散列值

![oneway01](/assets/images/reverse/oneway01.png)

+ **单向散列函数的特点**

  + 根据任意长度的消息，计算出固定长度的散列值
  + 计算速度快，能快速计算出散列值
  + 消息不同，散列值也不同
  + 具备单向性

  ![oneway02](/assets/images/reverse/oneway02.png)

+ 单向散列函数

  + 单向散列函数，又被称为消息摘要函数(message digest function)，哈希函数

  + 输出的散列值，也被称为消息摘要(message digest)、指纹(fingerprint)

  + 常见的几种单向散列函数

    + `MD4`、`MD5`

    + 产生`128bit`的散列值，`MD`就是`Message Digest`的缩写，目前已经不安全

    + Mac终端上默认可以使用`md5命令`

      ```
      $ md5 1.txt
      或者
      $ md5 -s "123"
      ```

    + `SHA-1`

      + 产生`160bit`的散列值，目前已经不安全

    + `SHA-2`

      + `SHA-256`、`SHA-384`、`SHA-512`，散列值长度分别是`256bit`、`384bit`、`512bit`

    + `SHA-3`

      + 全新标准

+ 如何防止数据被篡改

  ![oneway03](/assets/images/reverse/oneway03.png)

## 数字签名

+ 在数字签名技术中，有以下2种行为

  + 生成签名
    + 由消息的发送者完成，通过“签名密钥”生成
  + 验证签名
    + 由消息的接收者完成，通过“验证密钥”验证

+ 思考

  + 如何能保证这个签名是消息发送者自己签的？
  + 用消息发送者的`私钥`进行签名

  ![oneway04](/assets/images/reverse/oneway04.png)

+ 数字签名和公钥密码

  + 数字签名，其实就是将公钥密码反过来使用

  |            | 私钥                 | 公钥                       |
  | ---------- | :------------------- | :------------------------- |
  | 公钥密码   | 接受者解密时使用     | 发送者加密时使用           |
  | 数字签名   | 签名者生成签名时使用 | 验证者验证签名时使用       |
  | 谁持有密钥 | 个人持有             | 只要需要，任何人都可以持有 |

+ 数字签名的过程

  ![oneway05](/assets/images/reverse/oneway05.png)

+ 数字签名的过程 -- 改进

  ![oneway06](/assets/images/reverse/oneway06.png)

  ![oneway07](/assets/images/reverse/oneway07.png)

+ 如果有人篡改了文件内容或者签名内容，会是什么结果？

  + 签名验证失败，证明内容会篡改
  + 数字签名的作用不是为了保证机密性，仅仅是为了能够识别内容有没有被篡改

+ 数字签名的作用

  + 确认消息的完整性
  + 识别消息是否被篡改
  + 防止消息发送人否认

+ 数字签名无法解决的问题

  + 用于验证签名的公钥必须属于真正的发送者
  + 如果遭遇了中间人攻击，那么`公钥将是伪造的`、`数字签名将失效`
  + 所以在验证签名之前，首先得先验证公钥的合法性
  + 如何验证公钥的合法性？`证书`

  ![oneway08](/assets/images/reverse/oneway08.png)

## 证书(Certificate)

+ 证书，联想的是驾驶证、毕业证、英语四六级证等等，都是由权威机构认证的

+ 密码学中的证书，全称叫`公钥证书(Public-key Certificate，PKC`，跟驾驶证类似

  + 里面有姓名、邮箱等个人信息，以及此人的公钥
  + 并由认证机构(Certificate Authority，CA)施加数字签名

+ `CA`就是能够认定“公钥确实属于此人”并能够生成数字签名的个人或者组织

  + 有国际性组织、政府设立的组织
  + 有通过提供认证服务来盈利的企业
  + 个人也可以成立认证机构

+ 证书的利用

  ![oneway09](/assets/images/reverse/oneway09.png)

+ 证书的注册和下载

  ![oneway10](/assets/images/reverse/oneway10.png)

## iOS签名机制

+ iOS签名机制的作用

  + 保证安装到用户手机上的APP都是经过Apple官方允许的

+ 不管是真机调试，还是发布APP，开发者都需要经过一系列复杂的步骤

  + 生成`CertificateSigningRequest.certSigningRequest`文件
  + 获得`ios_development.cer` \ `ios_distribution.cer`证书文件
  + 注册`device`、添加`App ID`
  + 获得`*.mobileprovision`文件

+ 对于真机调试，现在的Xcode已经自动帮开发者做了以上操作

+ **iOS签名机制 – 流程图**

  ![oneway11](/assets/images/reverse/oneway11.png)

+ **iOS签名机制 – 生成Mac设备的公私钥**

  + `CertificateSigningRequest.certSigningRequest`文件
  + 就是Mac设备的公钥

  ![oneway12](/assets/images/reverse/oneway12.png)

+ **iOS签名机制 – 获得证书**

  ![oneway13](/assets/images/reverse/oneway13.png)

+ **iOS签名机制 – 获得证书**

  + `ios_development.cer`、`ios_distribution.cer`文件
  + 利用Apple后台的私钥，对Mac设备的公钥进行签名后的证书文件

+ **iOS签名机制 – 生成mobileprovision**

  ![oneway14](/assets/images/reverse/oneway14.png)

  ![oneway15](/assets/images/reverse/oneway15.png)

+ **iOS签名机制 – 安全检测**

  ![oneway16](/assets/images/reverse/oneway16.png)

+ **iOS签名机制 - AppStore**

  + 如果APP是从AppStore下载安装的，你会发现里面是没有`mobileprovision`文件的
  + 它的验证流程会简单很多，大概如下所示

  ![oneway17](/assets/images/reverse/oneway17.png)
