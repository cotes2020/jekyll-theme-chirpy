---
layout: post
title: Screeps 回坑
date: 2022-10-29 10:12:00 +0800
Author: Sokranotes
tags: [recording, Screeps]
comments: true
categories: recording
toc: true
typora-root-url: ..

---

# Screeps 回坑步骤

```shell
// 换源
npm config set registry https://registry.npm.taobao.org
// 清空缓存
npm cache clear --force
// 自动补全（声明文件）
npm install @types/screeps @types/lodash@3.10.1
// 安装rollup
npm install -D rollup
// 安装插件
npm install rollup-plugin-clear rollup-plugin-screeps rollup-plugin-copy -D
// 根目录下创建文件.secret.json
// .secret.json文件中填入如下内容，根据实际情况修改
{
    "main": {
        "token": "你的 screeps token 填在这里",
        "protocol": "https",
        "hostname": "screeps.com",
        "port": 443,
        "path": "/",
        "branch": "default"
    },
    "local": {
        "copyPath": "你要上传到的游戏路径，例如 C:\\Users\\DELL\\AppData\\Local\\Screeps\\scripts\\screeps.com\\default"
    }
}
```



## 参考资料

1. [Screep 中文教程目录](https://www.jianshu.com/p/5431cb7f42d3) —— [HoPGoldy](https://www.jianshu.com/u/3ee5572a4346)