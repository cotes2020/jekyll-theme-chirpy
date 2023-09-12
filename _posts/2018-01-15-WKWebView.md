---
title: WKWebView 的简单使用
date: 2018-01-15 16:03:25
categories: iOS
tags: UI
---

## 开发中一般只使用以下 4 个方法：

``` objc
// 决定是否发送请求 (类似 UIWebView 的 webView:shouldStartLoadWithRequest:navigationType:)
webView:decidePolicyForNavigationAction:decisionHandler:

// 页面开始加载 (类似 UIWebView 的 webViewDidStartLoad:)
webView:didStartProvisionalNavigation:

// 页面加载完成之后调用 (类似 UIWebView 的 webViewDidFinishLoad:)
webView:didFinishNavigation:

// 页面加载失败时调用 (类似 UIWebView 的 webView:didFailLoadWithError:)
webView:didFailProvisionalNavigation:withError:
```

<br>

## WKNavigationDelegate 代理方法：
``` objc
// 决定是否发送请求 (类似 UIWebView 的 webView:shouldStartLoadWithRequest:navigationType:)
- (void)webView:(WKWebView *)webView decidePolicyForNavigationAction:(WKNavigationAction *)navigationAction decisionHandler:(void (^)(WKNavigationActionPolicy))decisionHandler {
    NSLog(@"decidePolicyForNavigationAction");
    // 如果参数是 WKNavigationResponsePolicyCancel，则 webView 直接终止加载 url，不会有后续回调
    decisionHandler(WKNavigationActionPolicyAllow);
}

// 页面开始加载 (类似 UIWebView 的 webViewDidStartLoad:)
- (void)webView:(WKWebView *)webView didStartProvisionalNavigation:(null_unspecified WKNavigation *)navigation {
    NSLog(@"didStartProvisionalNavigation");
}

// 接收到认证询问 (有可能会多次调用？可能一次都不调用)
- (void)webView:(WKWebView *)webView didReceiveAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge completionHandler:(void (^)(NSURLSessionAuthChallengeDisposition disposition, NSURLCredential *__nullable credential))completionHandler {
    NSLog(@"didReceiveAuthenticationChallenge");
    completionHandler(NSURLSessionAuthChallengePerformDefaultHandling, nil);
}

// 在收到响应后，决定是否接收内容
- (void)webView:(WKWebView *)webView decidePolicyForNavigationResponse:(WKNavigationResponse *)navigationResponse decisionHandler:(void (^)(WKNavigationResponsePolicy))decisionHandler {
    NSLog(@"decidePolicyForNavigationResponse");
    // 如果参数是 WKNavigationResponsePolicyCancel，则 webView 不会接收内容，直接收到 didFailProvisionalNavigation 回调
    decisionHandler(WKNavigationResponsePolicyAllow);
}


// 当内容开始返回时调用
- (void)webView:(WKWebView *)webView didCommitNavigation:(null_unspecified WKNavigation *)navigation {
    NSLog(@"didCommitNavigation");
}

// 页面加载完成之后调用 (类似 UIWebView 的 webViewDidFinishLoad:)
- (void)webView:(WKWebView *)webView didFinishNavigation:(null_unspecified WKNavigation *)navigation {
    NSLog(@"didFinishNavigation");
}

// 页面加载失败时调用 (类似 UIWebView 的 webView:didFailLoadWithError:)
- (void)webView:(WKWebView *)webView didFailProvisionalNavigation:(null_unspecified WKNavigation *)navigation withError:(NSError *)error {
    NSLog(@"didFailProvisionalNavigation");
}

// 接收到服务器跳转请求之后调用
- (void)webView:(WKWebView *)webView didReceiveServerRedirectForProvisionalNavigation:(null_unspecified WKNavigation *)navigation {
    NSLog(@"didReceiveServerRedirectForProvisionalNavigation");
}

// WKNavigation导航错误
- (void)webView:(WKWebView *)webView didFailNavigation:(null_unspecified WKNavigation *)navigation withError:(NSError *)error {
    NSLog(@"didFailNavigation");
}

// WKWebView终止
- (void)webViewWebContentProcessDidTerminate:(WKWebView *)webView {
    NSLog(@"webViewWebContentProcessDidTerminate");
}
```
