---
title: 网络相关名词介绍
date: 2018-07-12 18:54:25
categories: iOS 
tags: 网络
---

<br>

## TCP/IP
互联网协议族，人们通常用 `TCP/IP` 来泛指整个互联网协议族，而不是单指这两种协议。[中文](https://zh.wikipedia.org/wiki/TCP/IP%E5%8D%8F%E8%AE%AE%E6%97%8F) / [英文](https://en.wikipedia.org/wiki/Internet_protocol_suite)

<br>

## IP
IP 是网际协议 (Internet Protocol) 的缩写。[中文](https://zh.wikipedia.org/wiki/%E7%BD%91%E9%99%85%E5%8D%8F%E8%AE%AE) / [英文](https://en.wikipedia.org/wiki/Internet_Protocol)

<br>

## TCP
TCP 是传输控制协议 (Transmission Control Protocol) 的缩写，TCP 是基于 IP 层的协议。建立起一个TCP连接需要经过“三次握手”：请求，确认，建立连接。[中文](https://zh.wikipedia.org/wiki/%E4%BC%A0%E8%BE%93%E6%8E%A7%E5%88%B6%E5%8D%8F%E8%AE%AE) / [英文](https://en.wikipedia.org/wiki/Transmission_Control_Protocol)

<br>

## UDP
用户数据报文协议（英语：User Datagram Protocol，缩写为UDP）[中文](https://zh.wikipedia.org/wiki/%E7%94%A8%E6%88%B7%E6%95%B0%E6%8D%AE%E6%8A%A5%E5%8D%8F%E8%AE%AE) / [英文](https://en.wikipedia.org/wiki/User_Datagram_Protocol)

<br>

## HTTP
HTTP，是超文本传输协议 (Hypertext Transfer Protocol) 的缩写，使用 HTTP 的 web 服务器会监听 80 端口。[中文](https://zh.wikipedia.org/wiki/%E8%B6%85%E6%96%87%E6%9C%AC%E4%BC%A0%E8%BE%93%E5%8D%8F%E8%AE%AE) / [英文](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol)

<br>

## HTTPS
HTTPS，基于 TLS 的 HTTP 请求就是 HTTPS，使用 HTTPS 的 web 服务器会监听 443 端口。
而 HTTP 又基于 TCP。TCP 连接就要执行三次握手，然后到了 TLS 层还会再握手三次。估算一下，建立一个 HTTPS 连接的耗时至少是创建一个 HTTP 连接的两倍。[中文](https://zh.wikipedia.org/wiki/%E8%B6%85%E6%96%87%E6%9C%AC%E4%BC%A0%E8%BE%93%E5%AE%89%E5%85%A8%E5%8D%8F%E8%AE%AE) / [英文](https://en.wikipedia.org/wiki/HTTPS)

<br>

## SSL
SSL（Secure Sockets Layer）是网景公司设计的安全传输协议，有 1.0 / 2.0 / 3.0 三个版本，但有设计缺陷。 [中文](https://zh.wikipedia.org/wiki/%E5%82%B3%E8%BC%B8%E5%B1%A4%E5%AE%89%E5%85%A8%E6%80%A7%E5%8D%94%E5%AE%9A) / [英文](https://en.wikipedia.org/wiki/Transport_Layer_Security)

<br>

## TLS
TLS（Transport Layer Security）安全传输层协议，1.0 版本基于 SSL 3.0 开发，后续移除了对 SSL 的兼容，安全性更高。尽量使用 TLS 1.2 或更新版本。[中文](https://zh.wikipedia.org/wiki/%E5%82%B3%E8%BC%B8%E5%B1%A4%E5%AE%89%E5%85%A8%E6%80%A7%E5%8D%94%E5%AE%9A) / [英文](https://en.wikipedia.org/wiki/Transport_Layer_Security)




<br>

## 证书锁定
证书锁定 (Certificate Pinning)，不仅要验证证书的有效性，还需要确定证书和其持有者是否匹配，可以防止“中间人攻击”。
AFNetworking 框架中可以通过 AFSecurityPolicy 来设置。

<br>

## URI
URI（Uniform Resource Identifier，统一资源标识符）URI的最常见的形式是统一资源定位符（URL）。[中文](https://zh.wikipedia.org/wiki/%E7%BB%9F%E4%B8%80%E8%B5%84%E6%BA%90%E6%A0%87%E5%BF%97%E7%AC%A6) / [英文](https://en.wikipedia.org/wiki/Uniform_Resource_Identifier)
<br>

## URL
URL (Uniform Resource Locator，统一资源定位符)。 [中文](https://zh.wikipedia.org/wiki/%E7%BB%9F%E4%B8%80%E8%B5%84%E6%BA%90%E5%AE%9A%E4%BD%8D%E7%AC%A6) / [英文](https://en.wikipedia.org/wiki/URL)


<br>
<br>
<br>