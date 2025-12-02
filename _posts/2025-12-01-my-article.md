---
title: "文章标题（Title）"
date: 2025-01-01 10:00:00 +0800
categories: [This_Demo, Tutorial]   # 分类（支持多级）
tags: [Demo]               				# 标签（不限数量）
description: "本篇文章的简短摘要，用于 SEO 与文章预览。"
author: 你的名字（可选）
pin: false                              # 是否置顶（true / false）

# 封面图（社交媒体预览图最佳尺寸 1200x630）
image:
  path: /assets/img/sample-cover.jpg     # 站点内路径
  alt: "封面图描述"
  lqip: /assets/img/lqip/sample.jpg      # （可选）低质量占位图

# 是否启用目录（左侧浮动目录）
toc: true
toc_label: "目录"
toc_icon: "list-ul"

# 文章版权 / 转载声明（可选）
copyright:
  license: CC BY-NC-SA 4.0
  holder: "你的名字"
---

<!--
如果需要摘要，放在这里，两段之间空一行。
这一段内容会在首页或搜索中作为 excerpt 显示。
-->

本文主要介绍……

<!-- more -->

---

## 一级标题示例

这里写正文内容……

### 二级标题示例

你可以正常写 Markdown 内容：

- 列表
- 加粗 **bold**
- 斜体 *italic*
- 链接 [Link](https://example.com)

---

## 插入代码块示例

\`\`\`python
def hello():
    print("Hello Chirpy!")
\`\`\`

---

## 插入图片示例

![示例图片 alt](/assets/img/demo.png)

你也可以设置宽度：

![Alt text](/assets/img/demo.png){: width="60%" }

---

## 引用示例

> 这是一段引用文本。

---

## 表格示例

| 名称 | 数值 |
| ---- | ---- |
| A    | 123  |
| B    | 456  |

---

## 脚注示例

这是一个脚注[^1]。

[^1]: 这里是脚注内容。

---

## MathJax（数学公式）示例

行内公式：\( a^2 + b^2 = c^2 \)

块级公式：

$$
\int_0^{1} x^2 \, dx = \frac{1}{3}
$$

---

## 最后一段

如果你愿意，我还能为你：

- 自动生成 tag / category 封面
- 提供文章批量创建脚本
- 自动生成图床链接或本地 assets 目录
- 给出封面图模板（1200x630）

需要哪个？我可以继续帮你生成。