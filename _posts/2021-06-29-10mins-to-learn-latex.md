---
title: 十分钟学会使用Latex
author: yuanjian
date: 2019-06-29 23:30:00 +0800
categories: [Tutorial]
tags: [skill]
pin: true
math: true
mermaid: true
image:
  path: https://i9.ytimg.com/vi/NqbkCPf2DU8/mqdefault.jpg?v=60dbede7&sqp=CMzqjq8G&rs=AOn4CLC0Cgt4SMkg9qC_eERbNk7Bqo9oHw
  alt: learn to use Latex in 10 minutes
---

这篇笔记简要地记录了Latex的基本使用技巧。

![Latex Simple Example](https://uploads-ssl.webflow.com/5d88b6baa39a48a9706c0dc5/60dbf776ce7a830faf2b0c9b_basic-latex2.png)

建议大家先观看10分钟的Youtube以后再阅读本文，视频在这里：

{% include embed/youtube.html id='NqbkCPf2DU8' %}

大家可以通过上面十分钟的Youtube视频了解如何在Overleaf上写Latex的文档。在这篇笔记中，我会为大家整理Latex的一些基本语法，方便大家使用。首先是一个基本的Latex文档结构，如下所示。

```latex
\documentclass[conference]{IEEEtran}
\usepackage{graphicx} % 用于插入图片
\usepackage{physics} % 用于插入公式

\chead{\thepage} %定义页眉 c表示中部 显示页数
\rfoot{\today}  %定义页脚 r表示右侧 显示今天的日期

\begin{document}
\title{Optimizing Error-Bounded Lossy Compression for Scientific Datasets with Pointwise Prerequisites}
\author{\IEEEauthorblockN{Yuanjian Liu} \\
University of Chicago, Chicago, IL, USA}}
\maketitle

\section{Introduction}
Hello, we can start writing here!
\section{Related Work}
\section{Method}
\section{Evaluation}
\bibliographystyle{IEEEtran}
\bibliography{reference}
\end{document}
```

视频中提到的cheating sheet，也就是各种数学符号的Latex表达式，可以在这个wiki中找到。展示的公式案例的文本形式在下面：

```latex
13.38 \textbf{Question:} \\
Let $a_n, b_n>0$ and $a_n \to \infty$. Prove: If $a_n=\Theta(b_n)$, then $\ln a_n \sim \ln b_n$.

\medskip\noindent
\textbf{Source:} None\\
\textbf{Answer:}\\
$a_n=\Theta(b_n) \implies (\exists c, d>0, N_0 \in N)
(\forall n>N_0)(0 \leq d|b_n|  \leq |a_n| \leq c|b_n|)$.
Because $\ln x$ is monotonically increasing,
we have $\ln d|b_n| \leq \ln |a_n| \leq \ln c|b_n|$. As we know $a_n, b_n>0$，
then we have $\ln d + \ln b_n \leq \ln a_n \leq \ln c + \ln b_n$.
Because $a_n \to \infty$, $\exists N_1>N_0$, when $n>N_1$, $a_n>1$,
therefore $\ln a_n>0$. So we have $\frac{\ln d + \ln b_n }{\ln a_n} \leq 1$
and $\frac{\ln c + \ln b_n}{\ln a_n} \geq 1$. Taking the limit,
we have $1 \leq \lim_{n \to \infty} \frac{\ln b_n}{ \ln a_n} \leq 1$,
so $\lim_{n \to \infty} \frac{\ln b_n}{ \ln a_n}=1$.
In this way, we've proven that $\ln a_n \sim \ln b_n$.

```


引用的格式称为BibTeX，我们不需要手动输入，在ACM、IEEE以及各类学术期刊的网站上，一般每篇论文都可以导出BibTeX格式的参考文献，我们只需要修改第一行的名字就可以进行引用了。

![cite reference example](https://uploads-ssl.webflow.com/5d88b6baa39a48a9706c0dc5/60dbf79536e9bc762617e5a0_cite-reference.png)