---
title: URL encoding and decoding using Scala
description: A quick hands-on example of ecoding and decoding urls in Scala.
tags: ["scala", "jvm"]
category: ["programming", "tutorial"]
date: 2017-04-03
permalink: '/scala/url-encoding-and-decoding-using-scala/'
counterlink: 'url-encoding-and-decoding-using-scala/'
---


### Introduction
Sometime it is required to encode or decode the URL in projects, especially when URL is supposed to be sent over the network. As the URLs often contain the characters that are outside UTF-8 character set, the URL must be converted to a valid character set. An example is to convert space character ” ” to “+” sign.

### EncodeURL Method
We are going to use `java.net.URLEncoder` class to encode our URL. It is a utility class that converts `String` to `application/x-www-form-urlencoded` MIME format. The class converts all the unsafe characters with “%” followed by two hexadecimal digits.

```scala
def encodeUrl(url: String): String =
  try {
    URLEncoder.encode(url, "UTF-8")
  } catch {
    case exception: UnsupportedEncodingException =>
      "Problems while encoding " + exception.getMessage
  }
```

The method `URLEncoder.encode` requires two parameters:

- First one is the input url, while
- The second one is supported characterset
  
The method provides the output as encoded url

### DecodeURL Method
We can decode our URL using `java.net.URLDecoder`. The `decode` method converts `application/x-www-form-urlencoded` MIME format to `String`. The `Decoder` class is reverse of Encoder. The hexadecimal digits that are followed by “%” are converted to the characters.

```scala
def decodeUrl(url: String): String = {
  val decodedUrl: Either[String, String] = try {
    Left(URLDecoder.decode( url, "UTF-8" ))
  } catch {
    case exception: UnsupportedEncodingException =>
      Right("Problems while encoding " + exception.getMessage)
  }
  
  decodedUrl match {
    case Left(dUrl) => if(url == dUrl) url else decodeUrl(dUrl)
    case Right(exceptionMsg) => exceptionMsg
  }
}
```
Similar to the `encode` method, `URLDecoder.decode` also accepts two parameters

- First one is the input url, while
- The second one is supported characterset 
  
But the `decodeURL` method is slightly different from `encodeURL` method. The url may be encoded multiple time, hence, `decodeURL` must keep decoding the URL until the URL get rid of all hexadecimal characters.

If you have a look at the above code, we are using tail recursion. We will keep passing the url to `decodeURL` method until the url and `decodedURL` are same.

### Output

- EncodeURL example 
  
`encodeUrl("https://gaur4vgaur.github.io/scala/url-encoding-and-decoding-using-scala")`

Result
`https%3A%2F%2Fgaur4vgaur.github.io%2Fscala%2Furl-encoding-and-decoding-using-scala`

- DecodeURL example
  
`decodeUrl("https%253A%252F%252Fgaur4vgaur.github.io%252Fscala%252Furl-encoding-and-decoding-using-scala")`

This URL above is encoded twice, still we get the expected result
`https://gaur4vgaur.github.io/scala/url-encoding-and-decoding-using-scala`

__NOTE__: Imports for above example

```scala
import java.net.URLEncoder
import java.io.UnsupportedEncodingException
import java.net.URLDecoder
```

