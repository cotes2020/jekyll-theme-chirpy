---
title: "腾讯云云开发短信验证码实战"
date: 2020-05-01
permalink: /2020-05-01-tcb-verification-code/
categories: ["C工作实践分享"]
---
最近支持了云开发的自定义短信验证码登录功能。


整篇文章不涉及业务，单独以短信验证码相关逻辑，带你上手云开发，前后端一把梭。


## 环境准备

- 前往腾讯云控制台，开通云开发
- 打开云开发设置-匿名登录
- 前往腾讯云控制台，开通 SMS
- 打开 SMS，创建并审核通过短信模版

## 架构设计


### 云数据库


前往 CloudBase 控制台，创建 tcb-sms-auth 集合。集合字段信息如下：


```text
expiration<number>: 验证码过期时间
phone<string>: 手机号
smsCode<string>: 验证码

```


除了 expiration 字段，还需要一个多余的字段来防止验证码对同一手机，在规定时间内，重复发送。但是腾讯云 SMS 自带频控管理，所以不在数据库中添加这个字段。


### 云函数


支持 3 种 Action：

- `send(phone)`: 向手机号 phone 发送随机验证码
- `verify(phone, smsCode)`: 检验手机验证码是否正确
- `clear()`: 定时任务清空手机验证码（前往 cloudbase 控制台-云函数-设置定时 corn）

整体架构设计如下：所有的服务都封装在 services 目录下；index.js 是入口文件，解析 C 端传入的参数，从而调用对应的


service。


![007S8ZIlly1gegbuahzkwj30iy0eadhd.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-01-tcb-verification-code/007S8ZIlly1gegbuahzkwj30iy0eadhd.jpg)


### 发送随机验证码


流程如下：

1. 查询云数据库，清空 phone 之前的验证码。保证在同一时刻，对同一个 phone，只有一个 smsCode 有效
2. 生成随机 6 位验证码，并将其存入云数据库

```text
/**
 * 生成验证码并存储到云数据库，发送短信
 *
 * @param {string} phone
 * @param {object} ctx
 */
async function sendSmsCode(phone, ctx) {
    const { db } = ctx;
    // 1. 移除之前的验证码
    const res = await db
        .collection(config.collection)
        .where({ phone })
        .get();
    if (res.data.length) {
        await db
            .collection(config.collection)
            .where({ phone })
            .remove();
    }
    // 2. 生成验证码
    const smsCode = randomStr(6);
    const period = 3 * 60 * 1000;
    const doc = {
        phone, // 电话号码
        smsCode, // 短信验证码
        expiration: Date.now() + period // 过期时间
    };
    await db.collection(config.collection).add(doc);
    // 3. 发送短信
    await sendSms({
        phone,
        smsCode
    });
}

```

1. 调用腾讯云 SMS 服务，向 phone 发送 smsCode

```text
/**
 * 发送短信
 *
 * @param {object} params
 */
function sendSms(params) {
    const { phone, smsCode } = params;
    const {
        TENCENTCLOUD_SECRETID,
        TENCENTCLOUD_SECRETKEY,
        TENCENTCLOUD_SESSIONTOKEN
    } = process.env;
    // 具体拼接请见：<https://github.com/TencentCloud/tencentcloud-sdk-nodejs/blob/master/examples/sms/v20190711/SendSms.js>
    const cred = new Credential(
        TENCENTCLOUD_SECRETID,
        TENCENTCLOUD_SECRETKEY,
        TENCENTCLOUD_SESSIONTOKEN
    );
    const client = new smsClient(cred, "ap-guangzhou");
    const req = new models.SendSmsRequest();
    req.SmsSdkAppid = config.SmsSdkAppid;
    req.Sign = config.Sign;
    req.ExtendCode = "";
    req.SenderId = "";
    req.SessionContext = "";
    req.PhoneNumberSet = [`+86${phone}`];
    req.TemplateID = config.TemplateID; // 模版类似：验证码是{1},有效期{2}
    req.TemplateParamSet = [smsCode, 3]; // 3代表3分钟
    return new Promise((resolve, reject) => {
        client.SendSms(req, (err, res) => {
            // 用于日志
            console.log(">>> sms err is", err);
            console.log(">>> sms res is", res);
            if (err) {
                err.code = "SMS_REQUEST_FAIL";
                return reject(err);
            }
            const json = JSON.parse(res.to_json_string());
            if (json.SendStatusSet[0].Code.toLowerCase() === "ok") {
                return resolve();
            }
            const error = new Error(json.SendStatusSet[0].Message);
            error.code = json.SendStatusSet[0].Code;
            return reject(error);
        });
    });
}

```


### 检验验证码有效性


利用聚合搜索，查询符合以下条件的数据库字段：

- phone 和 smsCode 匹配 C 端传入
- expiration 小于/等于当前时间戳

```text
/**
 * 验证验证码是否和云数据库中一致
 *
 * @param {string} phone
 * @param {string} smsCode
 * @param {object} ctx
 * @return {Promise<boolean>}
 */
async function verifySmsCode(phone, smsCode, ctx) {
    const { db, visitTime } = ctx;
    const _ = db.command;
    const res = await db
        .collection(config.collection)
        .where({
            phone,
            smsCode,
            expiration: _.gte(visitTime)
        })
        .get();
    return !!res.data.length;
}

```


### 清空过期验证码


查询 expiration 过期的所有记录，直接删除数据库记录


```text
/**
 * 清空过期验证码
 *
 * @param {object} ctx
 */
async function clearSmsCode(ctx) {
    const { db, visitTime } = ctx;
    const _ = db.command;
    await db
        .collection(config.collection)
        .where({
            expiration: _.lt(visitTime)
        })
        .remove();
}

```


### C 端消费


基于 tcb-js-sdk，通过匿名登录，调用短信验证码的云函数。请参考[tcb-js-sdk 文档](https://docs.cloudbase.net/api-reference/web/functions.html#callfunction)


