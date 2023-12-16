---
title: NetworkProtocol SSL/TLS Handshake
# author: Grace JyL
date: 2018-05-18 11:11:11 -0400
description:
excerpt_separator:
categories: [15NetworkSec, NetworkProtocol]
tags: [NetworkSec, NetworkProtocol, SSL, TLS]
math: true
# pin: true
toc: true
image: /assets/img/note/tls-ssl-handshake.png
---

# NetworkProtocol SSL/TLS Handshake

<<<<<<< HEAD
=======
- [NetworkProtocol SSL/TLS Handshake](#networkprotocol-ssltls-handshake)
  - [overall](#overall)
  - [4 SSL protocols:](#4-ssl-protocols)
  - [TLS Handshake Protocol:](#tls-handshake-protocol)
    - [Phase I of Handshake Protocol:](#phase-i-of-handshake-protocol)
    - [Phase II of Handshake Protocol:](#phase-ii-of-handshake-protocol)
    - [Phase III of Handshake Protocol](#phase-iii-of-handshake-protocol)
    - [Phase IV of Handshake Protocol](#phase-iv-of-handshake-protocol)
  - [DH算法的握手阶段](#dh算法的握手阶段)


---

## overall

>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
基本的运行过程

SSL/TLS协议的基本思路是采用公钥加密法，cliemt先向服务器端索要公钥，然后用公钥加密信息，服务器收到密文后，用自己的私钥解密。

（1）如何保证公钥不被篡改？
解决方法：将公钥放在数字证书中。只要证书是可信的，公钥就是可信的。
（2）公钥加密计算量太大，如何减少耗用的时间？
解决方法：每一次对话（session），客户端和服务器端都生成一个"对话密钥"（session key），用来加密信息。由于"对话密钥"是对称加密，所以运算速度非常快，而服务器公钥只用于加密"对话密钥"本身，这样就减少了加密运算的消耗时间。


握手之后的对话使用"对话密钥"加密（对称加密），服务器的公钥和私钥只用于加密和解密"对话密钥"（非对称加密），无其他作用。


因此，SSL/TLS协议的基本过程是这样的：
<<<<<<< HEAD
![Pasted Graphic](https://i.imgur.com/Lm3v5Bv.png)
=======

![Pasted Graphic](https://i.imgur.com/Lm3v5Bv.png)

>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
1. 客户端向服务器端索要并验证公钥。
2. 双方协商生成"对话密钥"。
3. 双方采用"对话密钥"进行加密通信。
上面过程的前两步，又称为"握手阶段"（handshake）。



---

## 4 SSL protocols:

![Pasted Graphic 15](https://i.imgur.com/FA5uvrx.png)

![Pasted Graphic 5](https://i.imgur.com/goZHVwC.png)

![Pasted Graphic 6](https://i.imgur.com/s3Tqk8q.png)
<<<<<<< HEAD
=======


>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
---

## TLS Handshake Protocol:
SSL缺省只进行server端的认证，客户端的认证是可选的。

![Pasted Graphic 16](https://i.imgur.com/5ZQdE3b.png)

![Pasted Graphic](https://i.imgur.com/gH1827k.png)

![A-SSL-TLS-handshake](https://i.imgur.com/ddmS4bp.png)

![Screen Shot 2020-09-24 at 11.36.15](https://i.imgur.com/YlJQa8r.png)

---

### Phase I of Handshake Protocol:

![Pasted Graphic 17](https://i.imgur.com/jyjhio1.png)

1. ClientHello 客户端发出请求
   - 传输过程中必须使用同一套加解密算法才能保证数据能够正常的加解密。
   - 客户端需要将本地支持的加密套件(Cipher Suite)的列表传送给服务端。
	   - TSL version 支持的最高version
		   - 从低到高依次: SSLv2 < SSLv3 < TLSv1 < TLSv1.1 < TLSv1.2
		   - 当前基本不再使用低于 TLSv1 的版本;
	   - Client random number 随机数 random_C，用于后续的密钥的生成
	   - Session ID
	   - cipher suites  客户端支持的加密套件 列表
		   - 每个加密套件对应前面 TLS 原理中的四个功能的组合：
		   - 认证算法 Au (身份验证)、
		   - 密钥交换算法 KeyExchange(密钥协商)、
		   - 对称加密算法 Enc (信息加密)
		   - 和信息摘要 Mac(完整性校验);
	   - compression methods 支持的压缩算法 列表，用于后续的信息压缩传输;
	   - 扩展字段 extensions，支持协议与算法的相关参数以及其它辅助信息等，常见的 SNI 就属于扩展字段，后续单独讨论该字段作用。
2. SeverHello 服务器回应
   - 服务端返回协商的信息结果，包括
	   - 选择使用的协议版本 version，
	   - 随机数 random_S
	   - Session ID
	   - 选择的加密套件 cipher suite，
	   - 选择的压缩算法 compression method
	   - …
3. 客户端和服务端都需要使用这两个随机数来产生Master Secret。

After Phase I, the client and server know the following:
❏ Version of SSL
❏ The two random numbers for key generation, 稍后用于生成”对话密钥”
❏ Session ID
❏ Cipher set: The algorithms for key exchange, message authentication, and encryption
❏ The compression method

---

### Phase II of Handshake Protocol:

![Pasted Graphic 18](https://i.imgur.com/BefWYDk.png)

客户端回应:
1. ServerCertificate 服务器端配置对应的证书链，用于身份验证与密钥交换;
   - 在接收到Client Hello之后，服务端将自己的证书发送给客户端。
     - 证书是对于服务端的一种认证。
   	 - 需要申请，由专门的数字证书认证机构(CA)严格审核后颁发的电子证书。
   - 颁发证书的同时会产生一个私钥和公钥。
     - 私钥由服务端自己保存，不可泄漏。
     - 公钥则是附带在证书的信息中，可以公开的。
   - 证书本身也附带一个证书电子签名
     - 用来验证证书的完整性和真实性，
     - 可以防止证书被串改。
     - 证书有有效期。
2. ServerKeyExchange
   - 在服务端向客户端发送的证书中没有提供足够的信息（证书公钥）的时候，还可以向客户端发送一个 ServerKeyExchange.
   - servercertificate 没有携带足够的信息时，发送给客户端以计算 pre-master，如基于 DH 的证书，公钥不被证书中包含，需要单独发送;
3. ClientCertificateRequest
   - 此外，对于非常重要的保密数据，服务端还需要对客户端进行验证，以保证数据传送给了安全的合法的客户端。
   - client收到后，首先需要向服务端发送client的证书，验证client的合法性。
   - 向client发出 CerficateRequest 消息，要求client发送证书对客户端的合法性进行验证。
   - 比如，金融机构往往只允许认证客户连入自己的网络，就会向正式客户提供USB密钥，里面就包含了一张客户端证书。
4. ServerHelloDone 通知客户端 server_hello 信息发送结束;
   - 最后server会发送一个ServerHelloDone消息给client，表示ServerHello消息结束了。

After Phase II,
❏  The server is authenticated to the client.
❏  The client knows the public key of the server if required.

4 cases in Phase II

![Pasted Graphic 19](https://i.imgur.com/t8WCx6i.png)

---

### Phase III of Handshake Protocol

![Pasted Graphic 20](https://i.imgur.com/iXftcaY.png)

1. Certificate client对server的证书进行检查:
   - 证书链的可信性 trusted certificate path，方法如前文所述;
	   - 证书是否吊销 revocation，有两类方式:
		   - 离线 CRL 与在线 OCSP，
		   - 不同的客户端行为会不同;
	   - 有效期 expirydate，证书是否在有效时间范围;
	   - 域名 domain，核查证书域名是否与当前的访问域名匹配，匹配规则后续分析;
   - 如果证书不是可信机构颁布、或者证书中的域名与实际域名不一致、或者证书已经过期，就会向访问者显示一个警告，由其选择是否还要继续通信。
   - 如果证书没有问题，client从服务器证书中取出服务器的公钥。
   - client_certificate与certificate_verify_message
   - If 服务器要求验证客户端，采用client的私钥加密的一段基于已经协商的通信信息得到数据，服务器可以采用对应的公钥解密并验证;
2. ClientKeyExchange
   - 合法性验证通过之后，客户端使用加密算法(如RSA, Diffie-Hellman)产生一个48个字节的Key, PreMaster Secret/Key，用证书公钥加密，发送给服务器;
   - 该随机数用服务器公钥加密，防止被窃听
   - 此时客户端已经获取全部的计算协商密钥需要的信息：两个明文随机数 random_C 和 random_S 与自己计算产生的 Pre-master，计算得到协商密钥session secret ;
   - enc_key=Fuc(random_C, random_S, Pre-Master)
3. CertificateVerify
   - 编码改变通知，表示随后的信息都将用双方商定的加密方法和密钥发送
   - client握手结束通知，表示client的握手阶段已经结束。这一项同时也是前面发送的所有内容的hash值，用来供服务器校验

After Phase III,
❏ The client is authenticated for the server.
❏ Both the client and the server know the pre-master secret.

4 cases in Phase III

![Pasted Graphic 21](https://i.imgur.com/iSwLeFi.png)

---

### Phase IV of Handshake Protocol

![Pasted Graphic 22](https://i.imgur.com/767YJ6p.png)

Client:
1. ChangeCipherSpec
   - ChangeCipherSpec是一个独立的协议，体现在数据包中就是一个字节的数据，用于告知服务端，客户端已经切换到之前协商好的加密套件（Cipher Suite）的状态，准备使用之前协商好的加密套件加密数据并传输了。
   - 在ChangecipherSpec传输完毕之后，客户端会使用之前协商好的加密套件和SessionSecret加密一段 Finish 的数据传送给服务端，此数据是为了在正式传输应用数据之前对刚刚握手建立起来的加解密通道进行验证。
2. Finished encrypted_handshake_message，
   - client结合之前所有通信参数的hash值与其它相关信息生成一段数据，采用协商密钥 session secret 与算法进行加密，然后发送给服务器用于数据与握手验证;

Server:
1. 服务器
   - 用私钥解密加密的 Pre-master 数据，基于之前交换的两个明文随机数 random_C 和 random_S，计算得到协商密钥session secret :
	   - enc_key=Fuc(random_C, random_S, Pre-Master);
   - 计算之前所有接收信息的hash值,然后解密客户端发送的encrypted_handshake_message,验证数据和密钥正确性;
2. ChangeCipherSpec
   - change_cipher_spec, 验证通过之后，服务器同样发送 ChangeCipherSpec 以告知客户端后续的通信都采用协商的密钥与算法进行加密通信;
3. Finished encrypted_handshake_message
   - 服务器也结合所有当前的通信参数信息生成一段数据并采用协商密钥 SessionSecret 与算法加密一段 Finish 消息发送给客户端，以验证之前通过握手建立起来的加解密通道是否成功。
4. 如果客户端和服务端都能对Finish信息进行正常加解密且消息正确的被验证，则说明握手通道已经建立成功，接下来，双方可以使用上面产生的Session Secret对数据进行加密传输了。
5. 客户端计算所有接收信息的 hash 值，并采用协商密钥解密 encrypted_handshake_message，验证服务器发送的数据和密钥，验证通过则握手完成;
6. 握手结束
7. 加密通信, 开始使用协商密钥与算法进行加密通信。

After Phase IV, the client and server are ready to exchange data.

---

注意：
alter message 用于指明在握手或通信过程中的状态改变或错误信息，一般告警信息触发条件是连接关闭，收到不合法的信息，信息解密失败，用户取消操作等，收到告警信息之后，通信会被断开或者由接收方决定是否断开连接。

---

## DH算法的握手阶段
- 整个握手阶段都不加密（也没法加密），都是明文。因此，如果有人窃听通信，他可以知道双方选择的加密方法，以及三个随机数中的两个。整个通话的安全，只取决于第三个随机数（Premaster secret）能不能被破解。
- 虽然理论上，只要服务器的公钥足够长（比如2048位），那么Premaster secret可以保证不被破解。但是为了足够安全，我们可以考虑把握手阶段的算法从默认的RSA算法，改为Diffie-Hellman算法（简称DH算法）。
- 采用DH算法后，Premaster secret不需要传递，双方只要交换各自的参数，就可以算出这个随机数。

![Pasted Graphic 7](https://i.imgur.com/0TPEhzQ.png)

上图中，第三步和第四步由传递Premaster secret变成了传递DH算法所需的参数，然后双方各自算出Premaster secret。这样就提高了安全性。
