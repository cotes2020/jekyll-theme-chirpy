---
title: Lab - Burpsuite - Intruder 发现敏感目录
date: 2020-11-20 11:11:11 -0400
description: SQL injection
categories: [Lab, Burpsuite]
# img: /assets/img/sample/rabbit.png
tags: [Lab, Burpsuite]
---

[toc]

---

# Burpsuite - Intruder 发现敏感目录


1. 在burpsuite的proxy栏目中，找到对WackoPicko路径的请求报，右键选择‘Send to intruder’
2. 切换到Intruder栏目下的Position选项
   - 被 `§` 包裹着的字段，高亮显示的
   - 这些字段是Intruder在每次请求中都会更改的字段，单击Clear按钮清空所有被`§`包裹着的字段。

3. 在url的最后一个`/`后随意加一个字段
   - 然后选中它，并单击Add按钮，让这个被选中的字符成为一个修改点

![wbmc=](https://i.imgur.com/iCpl5ZH.png)

4. 切换到Payload选项下
   - 由于只设置了一个修改点
   - 所以只需要根据默认配置生成一个攻击载荷列表即可
   - 将攻击载荷的类型设置为simple list
   - 然后载入一个外部的攻击列表。

5. 单击Load，选择选择`/user/share/wordlists/dirb/small.txt`(可以手动添加 or 加载目录字典)

6. 然后点击Start attack按钮开始向服务器发送请求，
   - 200是存在且可访问的文件或目录的响应代码，
   - 重定向为300，
   - 错误范围为400和500。


Position中的type的其它类型：

- **Sniper**
  - 将一组攻击载荷分别替换每一个修改点上，
  - 每个替换后的值都是不同的。

- **Battering ram**：
  - 和Sniper一样，它也使用一组攻击载荷，
  - 但是不同的是它在每一次修改中会把所有修改点都替换成一样的值。

- **Pitchfork**
  - 将多个攻击载荷集中的每一项依次替换不同的修改点，
  - 当我们对已知的某些数据进行测试时会发现它很有用，比如对一组用户名和密码进行测试。

- **Cluster bomb**：
  - 测试多个攻击载荷的所有排列组合。
