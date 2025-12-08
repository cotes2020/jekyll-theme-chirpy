---
title: "Protobuf协议优化研究与Node.js使用"
date: 2020-07-11
permalink: /2020-07-11-node-protobuf/
categories: ["C工作实践分享"]
---

Protocol Buffers 是一种轻便高效的结构化数据存储格式，可以用于结构化数据串行化，或者说序列化。它很适合做数据存储或 RPC 数据交换格式。


特点：语言无关、平台无关、可扩展序列化结构。


场景：数据存储和交换。


优点：定义简单、(反)序列化速度快、支持多种数据结构


## 编码优化


### Varint 编码:不定长二进制整型编码


![007S8ZIlgy1gixl5y05y6j30jy06rgm4.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-07-11-node-protobuf/007S8ZIlgy1gixl5y05y6j30jy06rgm4.jpg)


第一位的 0 或者 1 代表：是否是数字的尾部；剩下的 7 位用于记录数字的二进制。


例如，数字 299 在 int32 下是：


```text
00000000 00000000 00000001 00101011
```


编码后是：


```text
10101011 00000010
```


**优点很明显：节省了 2 个字节。**


对于小于$2^{28}$的 int32 或者 int64，Varint 起到压缩效果。对于大数字，位数反而更多。但是，**正常情况下小数字使用频率远远高于大数字**。


### Zigzag: 有符号数编码优化


原因：对于负数，是以补码形式存储（数大），占用位数多。


作用：将有符号整数映射为无符号整数，例如 -1 => 1；原本无符号的整数，变成之前的 2 倍，例如 1 => 2。


![007S8ZIlgy1gixl6jtdc1j30fw0d43yr.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-07-11-node-protobuf/007S8ZIlgy1gixl6jtdc1j30fw0d43yr.jpg)


## 传输优化


protobuf 是以下面的形式传输的。优点如下：

- Field 之间没有区分符号，节省字节
- Key 是 filed_id

![007S8ZIlgy1gixl79jgogj30ci04gmxd.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-07-11-node-protobuf/007S8ZIlgy1gixl79jgogj30ci04gmxd.jpg)


第 2 点比较难理解。比如一个 proto:


```protobuf
message demo {
  number age = 1;
}
```


对于 filed age 来说，field id 是 1。但是传输的时候不会传输 Key（也就是"age"），**而是根据 field id 以及数据类型进行位运算，生成一个整数。规则如下**：


```javascript
(field_id << 3) | wire_type
```


![007S8ZIlgy1gixl7lqq2tj30qn0brq41.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-07-11-node-protobuf/007S8ZIlgy1gixl7lqq2tj30qn0brq41.jpg)


消费端接收到数据流后，可以很快通过位运算解出 filed（id+value）。然后根据 id，再去读取 proto 定义文件（**生产端和消费端共有**），将其处理成对应的 key。


流程如上，优化了网络传输数据量，更多逻辑放在本地。


## Node 中使用 protobuf


### .proto 定义


`person.proto`:


```protobuf
package person; // 名字空间
syntax = "proto3";

message PersonMessage {
    required string name = 1;
    optional string email = 2;
    // 枚举
    enum SexType {
        UNKNOWN = 0;
        MALE = 1;
        FEMALE = 2;
    }
    required SexType sex = 3 [default = UNKNOWN]; // 默认值

    // 嵌套message
    message LocationType {
        required string country = 1;
    }
    optional LocationType location = 4;
}
```


`student.proto` :


```protobuf
syntax = "proto3";

import "./person.proto"; // 模块引用

package student;

message StudentMessage {
    required string school = 1;
    required person.PersonMessage teacher = 2;
}
```


### 加载、验证、编/解码


```javascript
const protobuf = require('protobufjs');

main();

function main() {
    const payload = {
        school: 'szu',
        teacher: {
            name: 'dongyuanxin',
            sex: 2,
            location: {
                country: 'zh-cn',
            },
        },
    };

    protobuf.load('./student.proto').then((root) => {
        const AwesomeMessage = root.lookupType('student.StudentMessage');

        // step1：校验是否合法
        let verified = AwesomeMessage.verify(payload);
        if (verified) {
            // verified 存放不合法信息
            throw new Error(verified);
        }

        // step2: 工厂模式创建编译/解码器
        let message = AwesomeMessage.create(payload);
        console.log(message);

        // step3: 编译为2进制
        let buffer = AwesomeMessage.encode(message).finish();
        console.log(buffer);
        // step4: 解码
        let decoded = AwesomeMessage.decode(buffer);
        console.log(decoded);
    });
}
```


## 参考链接

- [IBM：Google Protocol Buffer 的使用和原理](https://www.ibm.com/developerworks/cn/linux/l-cn-gpb/)
- [在 NodeJS 中玩转 ProfoBuf](https://imweb.io/topic/570130a306f2400432c1396c)
- [浅析 Protobuf 整形编码方式：Varint 与 Zigzag](https://juejin.im/post/6844904025578553351)

